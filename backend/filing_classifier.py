"""
filing_classifier.py — BUY/SELL/HOLD classifier for SEC filings.

Loads the fine-tuned LoRA model from data/finetune/model/ if it exists.
Falls back to Claude Sonnet 4.6 if not yet trained.

Singleton — loaded once per process, kept in memory.

Usage:
  from backend.filing_classifier import FilingClassifier

  result = FilingClassifier.get().classify("AAPL")
  # {"signal": "SELL", "confidence": 0.78, "reason": "...", "source": "local", ...}
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger("raphi.filing_classifier")

_ROOT      = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ROOT / "data" / "finetune" / "model"
_CACHE: dict[str, tuple[dict, float]] = {}
_CACHE_TTL = 3600  # 1 hour


def _build_prompt(ticker: str, form: str, filed: str, text: str, financials: dict) -> str:
    fin_str = ", ".join(f"{k}={v}" for k, v in financials.items() if v is not None) or "unavailable"
    return (
        "You are a financial analyst. Analyze this SEC filing excerpt and classify the investment signal.\n\n"
        f"Ticker: {ticker}\n"
        f"Filing type: {form}\n"
        f"Filed: {filed}\n"
        f"Financials: {fin_str}\n\n"
        f"Filing excerpt:\n{text[:2500].strip()}\n\n"
        'Respond in JSON: {"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "one sentence"}'
    )


def _parse_response(raw: str) -> tuple[str, float, str]:
    """Extract (signal, confidence, reason) from model output. Tolerates partial JSON."""
    raw = raw.strip()
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            signal     = str(data.get("signal", "HOLD")).upper()
            signal     = signal if signal in ("BUY", "SELL", "HOLD") else "HOLD"
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.6))))
            reason     = str(data.get("reason", ""))[:300]
            return signal, confidence, reason
    except Exception:
        pass
    # Keyword fallback
    upper = raw.upper()
    for sig in ("BUY", "SELL", "HOLD"):
        if sig in upper:
            return sig, 0.55, raw[:200]
    return "HOLD", 0.5, "Could not parse classifier output"


class FilingClassifier:
    _instance: "FilingClassifier | None" = None
    _pipe = None

    @classmethod
    def get(cls) -> "FilingClassifier":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._pipe        = None
        self._base_model: str | None = None
        self._load_local_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_local_model(self) -> None:
        config_path = _MODEL_DIR / "raphi_finetune_config.json"
        if not config_path.exists():
            logger.info(
                "FilingClassifier: no fine-tuned model at %s — using Claude fallback. "
                "Run `python -m backend.finetune.label_builder` then "
                "`python -m backend.finetune.train` to build a local model.",
                _MODEL_DIR,
            )
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from peft import PeftModel

            with open(config_path) as f:
                cfg = json.load(f)
            base_model = cfg["base_model"]

            logger.info(
                "FilingClassifier: loading fine-tuned model (base=%s) …", base_model
            )
            tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            dtype = torch.float16 if device == "cuda" else torch.float32

            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map="auto" if device in ("cuda", "mps") else None,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, str(_MODEL_DIR))
            model.eval()

            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=120,
                temperature=0.05,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            self._base_model = base_model
            logger.info("FilingClassifier: local model ready (%s)", device)

        except Exception as exc:
            logger.warning(
                "FilingClassifier: failed to load local model (%s) — falling back to Claude", exc
            )
            self._pipe = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_latest_filing(self, ticker: str) -> tuple[str, str, str]:
        """Return (filing_text, form_type, filed_date). Text is '' on failure."""
        try:
            from backend.edgar_live import get_recent_filings, get_filing_text
            filings = get_recent_filings(ticker, forms=["10-Q", "10-K"], days=180, limit=1)
            if not filings:
                return "", "", ""
            f    = filings[0]
            text = get_filing_text(
                f["accession"], f["cik"],
                primary_doc=f.get("primary_doc"),
                max_chars=3000,
            ) or ""
            return text, f.get("form", "10-Q"), f.get("filed", "")
        except Exception as exc:
            logger.debug("Filing fetch failed for %s: %s", ticker, exc)
            return "", "", ""

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_local(self, ticker: str, form: str, filed: str, text: str, financials: dict) -> dict:
        prompt   = _build_prompt(ticker, form, filed, text, financials)
        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        try:
            output = self._pipe(messages)[0]["generated_text"]
            # pipeline returns full conversation; extract last assistant turn
            if isinstance(output, list):
                reply = output[-1].get("content", "") if output else ""
            else:
                reply = str(output)[len(prompt):]
        except Exception as exc:
            logger.warning("Local model inference failed (%s) — falling back to Claude", exc)
            return self._run_claude(ticker, form, filed, text, financials)

        latency_ms            = int((time.time() - t0) * 1000)
        signal, conf, reason  = _parse_response(reply)
        return {"signal": signal, "confidence": conf, "reason": reason,
                "source": "local", "ticker": ticker, "latency_ms": latency_ms}

    def _run_claude(self, ticker: str, form: str, filed: str, text: str, financials: dict) -> dict:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {
                "signal": "HOLD", "confidence": 0.5,
                "reason": "No local model and ANTHROPIC_API_KEY not set",
                "source": "default", "ticker": ticker, "latency_ms": 0,
            }

        import anthropic
        prompt = _build_prompt(ticker, form, filed, text, financials)
        t0     = time.time()
        try:
            client  = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = message.content[0].text if message.content else ""
        except Exception as exc:
            logger.warning("Claude classifier call failed: %s", exc)
            return {
                "signal": "HOLD", "confidence": 0.5,
                "reason": f"Classifier unavailable: {exc}",
                "source": "default", "ticker": ticker, "latency_ms": 0,
            }

        latency_ms           = int((time.time() - t0) * 1000)
        signal, conf, reason = _parse_response(reply)
        return {"signal": signal, "confidence": conf, "reason": reason,
                "source": "claude", "ticker": ticker, "latency_ms": latency_ms}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        ticker:       str,
        filing_text:  str | None  = None,
        financials:   dict | None = None,
    ) -> dict:
        """
        Classify a filing as BUY, SELL, or HOLD.

        Args:
            ticker:      Stock ticker symbol.
            filing_text: Pre-fetched filing text. Fetched from EDGAR if None.
            financials:  Fundamentals dict (pe_ratio, revenue_growth, market_cap, …).

        Returns:
            {signal, confidence, reason, source, ticker, latency_ms}
        """
        cache_key = f"{ticker}:{hash(filing_text or '')}"
        cached = _CACHE.get(cache_key)
        if cached is not None:
            result, ts = cached
            if time.time() - ts < _CACHE_TTL:
                return result

        financials = financials or {}
        form = filed = ""

        if filing_text is None:
            filing_text, form, filed = self._fetch_latest_filing(ticker)

        if not filing_text or len(filing_text) < 100:
            result = {
                "signal": "HOLD", "confidence": 0.5,
                "reason": "No filing text available for classification",
                "source": "default", "ticker": ticker, "latency_ms": 0,
            }
            _CACHE[cache_key] = (result, time.time())
            return result

        if self._pipe is not None:
            result = self._run_local(ticker, form, filed, filing_text, financials)
        else:
            result = self._run_claude(ticker, form, filed, filing_text, financials)

        _CACHE[cache_key] = (result, time.time())
        return result
