"""
label_builder.py — Build JSONL training data for the SEC filing classifier.

For each ticker:
  1. Fetch 10-Q and 10-K filings from EDGAR (3 years back)
  2. Compute event study: abnormal return (stock − SPY) + z-score + triple barrier
  3. Fetch filing text excerpt via edgar_live
  4. Write instruction-tuning JSONL to data/finetune/training_data.jsonl

Labeling method (industry-standard event study):
  Alpha      = log(stock_return) − β × log(SPY_return)  (beta-adjusted log-return)
  Beta       = cov(stock, SPY) / var(SPY) over 60-day pre-event window
  Sigma      = std(pre-filing differential log returns, 60 days) × √21
  Z-score    = alpha / sigma  (how unusual the move was vs this stock's own vol)

  Triple barrier (Lopez de Prado):
  BUY   — cumulative alpha crosses +1.5σ before day 21
  SELL  — cumulative alpha crosses −1.5σ before day 21
  HOLD  — neither barrier hit; z-score at day 21 is between −1.5σ and +1.5σ

Universe options (--universe flag):
  sp500      — S&P 500 (~500 tickers, ~35 min with 3 workers)
  nasdaq100  — Nasdaq-100 (~100 tickers, ~8 min)
  sec_all    — All SEC EDGAR filers (~13,000 tickers, ~6 hrs with 3 workers)
  default    — Built-in 40-ticker seed list (fastest, for testing)

Usage:
  python -m backend.finetune.label_builder                          # 40 tickers
  python -m backend.finetune.label_builder --universe sp500         # S&P 500
  python -m backend.finetune.label_builder --universe sec_all       # all stocks
  python -m backend.finetune.label_builder --tickers AAPL MSFT NVDA # custom list
  python -m backend.finetune.label_builder --universe sp500 --workers 5

Resume: safe to kill and re-run — completed tickers are checkpointed and skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger("raphi.finetune.label_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# yfinance is not thread-safe: concurrent yf.download calls for different tickers
# receive each other's columns, causing MISSING_COL → event=None → 0 written.
_yf_lock = threading.Lock()

_ROOT        = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUT = _ROOT / "data" / "finetune" / "training_data.jsonl"

_DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "JPM", "BAC", "GS",
    "XOM", "CVX", "JNJ", "UNH", "PFE", "WMT", "HD", "COST", "V", "MA",
    "AMD", "INTC", "QCOM", "AVGO", "MU", "CRM", "ORCL", "ADBE", "NOW", "SNOW",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "BA", "CAT", "GE", "MMM", "HON",
]


# ──────────────────────────────────────────────────────────────────────────────
# Universe fetchers
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_sp500() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        import pandas as pd
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info("S&P 500: fetched %d tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("S&P 500 fetch failed (%s) — using default list", exc)
        return _DEFAULT_TICKERS


def _fetch_nasdaq100() -> list[str]:
    """Fetch Nasdaq-100 tickers from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        # Find the table that has a Symbol or Ticker column
        for df in tables:
            for col in df.columns:
                if str(col).lower() in ("ticker", "symbol"):
                    tickers = df[col].dropna().str.replace(".", "-", regex=False).tolist()
                    if len(tickers) > 50:
                        logger.info("Nasdaq-100: fetched %d tickers from Wikipedia", len(tickers))
                        return tickers
        raise ValueError("ticker column not found")
    except Exception as exc:
        logger.warning("Nasdaq-100 fetch failed (%s) — using default list", exc)
        return _DEFAULT_TICKERS


def _fetch_sec_all() -> list[str]:
    """
    Fetch all tickers that file with the SEC (~13,000 companies).
    Source: SEC EDGAR company_tickers.json — authoritative, free, no rate limit.
    """
    try:
        import httpx
        resp = httpx.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "RAPHI/1.0 alankrit04jan@gmail.com"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        tickers = []
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).strip().upper()
            # Skip blank, ETFs that aren't equities, and very long symbols
            if ticker and len(ticker) <= 5 and ticker.isalpha():
                tickers.append(ticker)
        # Deduplicate preserving order
        seen: set[str] = set()
        unique = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        logger.info("SEC all: fetched %d unique tickers from EDGAR", len(unique))
        return unique
    except Exception as exc:
        logger.warning("SEC all-tickers fetch failed (%s) — using default list", exc)
        return _DEFAULT_TICKERS


def _tickers_from_watchlist() -> list[str]:
    settings_path = _ROOT / "backend" / "user_settings.json"
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                data = json.load(f)
            wl = data.get("watchlist", [])
            if wl:
                return [t.upper() for t in wl]
        except Exception:
            pass
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint (resume support)
# ──────────────────────────────────────────────────────────────────────────────

def _checkpoint_path(out_path: Path) -> Path:
    return out_path.with_suffix(".checkpoint.json")


def _load_checkpoint(out_path: Path) -> set[str]:
    """Return set of tickers already fully processed."""
    cp = _checkpoint_path(out_path)
    if cp.exists():
        try:
            data = json.loads(cp.read_text())
            done = set(data.get("done", []))
            logger.info("Resume: %d tickers already done (from checkpoint)", len(done))
            return done
        except Exception:
            pass
    return set()


def _mark_done(out_path: Path, ticker: str, lock: threading.Lock) -> None:
    cp = _checkpoint_path(out_path)
    with lock:
        data: dict = {"done": []}
        if cp.exists():
            try:
                data = json.loads(cp.read_text())
            except Exception:
                pass
        done = set(data.get("done", []))
        done.add(ticker)
        cp.write_text(json.dumps({"done": sorted(done)}, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Labeling helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EventReturn:
    alpha:       float  # beta-adjusted log abnormal return at label day
    zscore:      float  # alpha / (daily_vol × √21)
    label:       str    # BUY / SELL / HOLD
    barrier_hit: str    # "upper" / "lower" / "time"
    day_hit:     int    # trading day when barrier triggered (1–21)
    beta:        float  # stock's sensitivity to SPY in pre-event window


def _compute_event_return(ticker: str, filed_date: str) -> "EventReturn | None":
    """
    Event study: abnormal return + z-score + triple barrier labeling.

    Alpha      = log(stock/stock_start) − β × log(SPY/spy_start)  (beta-adjusted)
    Beta       = cov(stock_log, SPY_log) / var(SPY_log) over 60-day pre-event window
    Sigma      = std(pre-event differential log returns, 60 days) × √21
    Z          = alpha / sigma

    Why log returns: time-additive, no upward bias on large moves,
    theoretically correct for vol scaling via √T.

    Why 60-day vol window: 20 days is one calendar month — a single
    outlier move distorts the entire estimate. 60 days gives a stable
    daily vol estimate with enough observations to be meaningful.

    Why beta adjustment: a stock with beta=1.6 that moved +8% when SPY
    moved +5% gained only +0.3% of true alpha (8% − 1.6×5% = 0%). Without
    beta, we'd label that filing BUY when the stock just moved with the
    market. Beta removes this systematic sensitivity so only the
    stock-specific move is measured.

    Triple barrier (Lopez de Prado):
      Upper  +1.5σ → BUY
      Lower  −1.5σ → SELL
      Time   day 21 → label by z-score if neither barrier hit
    """
    import yfinance as yf
    import pandas as pd

    filing = date.fromisoformat(filed_date)

    # Need 61 pre-event prices (for 60 log returns) + 21 post-event prices.
    # 120 calendar days back gives ~85 trading days — enough margin for
    # holidays, gaps, and the 61-price requirement.
    start = (filing - timedelta(days=120)).isoformat()
    end   = (filing + timedelta(days=50)).isoformat()

    with _yf_lock:
        try:
            data = yf.download(
                [ticker, "SPY"],
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                actions=False,
            )
        except Exception as exc:
            logger.debug("yfinance download failed %s @ %s: %s", ticker, filed_date, exc)
            return None

    if data.empty or "Close" not in data:
        return None

    close = data["Close"]
    if ticker not in close.columns or "SPY" not in close.columns:
        return None

    # Align stock and SPY — drop any date where either has no data.
    aligned = close[[ticker, "SPY"]].dropna()
    if len(aligned) < 83:  # 61 pre-event prices + 1 event + 21 post-event
        return None

    # yfinance returns tz-aware index; filing dates are plain strings.
    # Comparing them crashes with TypeError — strip tz first.
    if getattr(aligned.index, "tz", None) is not None:
        aligned.index = aligned.index.tz_localize(None)

    stock_prices = aligned[ticker]
    spy_prices   = aligned["SPY"]

    # First trading day on or after filing date.
    # SEC filings can land on weekends/holidays which are not in the index.
    filing_ts = pd.Timestamp(filed_date)
    mask = stock_prices.index >= filing_ts
    if not mask.any():
        return None
    event_idx = int(mask.argmax())

    # Need 61 pre-event prices (= 60 log returns) and 21 post-event prices.
    # Returns None for recent IPOs or filings without enough price history.
    if event_idx < 61 or event_idx + 21 >= len(stock_prices):
        return None

    # Pre-event differential log return volatility over 60 trading days.
    # Vol of (stock log return − SPY log return) is the correct denominator
    # for the z-score. Using stock vol alone overstates the threshold because
    # correlated SPY moves cancel out — the differential is quieter than the
    # stock in isolation.
    # -inf guard: np.log of a zero price produces -inf, not NaN, so dropna()
    # misses it. np.isfinite() filter catches it before std() sees it.
    pre_stock = stock_prices.iloc[event_idx - 61 : event_idx]
    pre_spy   = spy_prices.iloc[event_idx - 61 : event_idx]

    pre_stock_log = np.log(pre_stock / pre_stock.shift(1)).dropna()
    pre_spy_log   = np.log(pre_spy   / pre_spy.shift(1)).dropna()

    pre_diff = (pre_stock_log - pre_spy_log).dropna()
    pre_diff = pre_diff[np.isfinite(pre_diff)]  # remove -inf/+inf from bad price data

    if len(pre_diff) < 30:
        return None
    daily_vol = float(pre_diff.std())
    if daily_vol <= 0 or not np.isfinite(daily_vol):
        return None

    # Beta = cov(stock, SPY) / var(SPY) over the pre-event window.
    # Aligning on the shared index ensures cov and var use identical dates.
    # Clamped [0, 3]: negative beta is genuine for <10 S&P stocks (gold miners,
    # VIX products) but near-impossible for equities — more likely bad data.
    # Beta > 3 is also bad data — no broad-market stock moves 3× SPY on average.
    # Fallback to 1.0 keeps the formula correct when SPY var is degenerate.
    aligned_log = pd.concat([pre_stock_log, pre_spy_log], axis=1).dropna()
    aligned_log.columns = ["stock", "spy"]
    spy_var = float(aligned_log["spy"].var())
    if spy_var <= 0 or not np.isfinite(spy_var):
        beta = 1.0
    else:
        raw_beta = float(aligned_log["stock"].cov(aligned_log["spy"])) / spy_var
        beta = max(0.0, min(raw_beta, 3.0))

    # 21-day horizon vol: daily_vol × √21 (i.i.d. log returns assumed)
    horizon_vol = daily_vol * np.sqrt(21)
    barrier     = 1.5 * horizon_vol

    stock_start = float(stock_prices.iloc[event_idx])
    spy_start   = float(spy_prices.iloc[event_idx])
    if stock_start <= 0 or spy_start <= 0:
        return None

    # Triple barrier walk — log returns throughout for consistency with vol.
    for day in range(1, 22):
        idx = event_idx + day
        if idx >= len(stock_prices):
            break
        s_price = float(stock_prices.iloc[idx])
        b_price = float(spy_prices.iloc[idx])
        if s_price <= 0 or b_price <= 0:
            continue
        cum_alpha = (
            np.log(s_price / stock_start)
            - beta * np.log(b_price / spy_start)
        )
        if cum_alpha >= barrier:
            return EventReturn(
                alpha=float(cum_alpha),
                zscore=float(cum_alpha / horizon_vol),
                label="BUY", barrier_hit="upper", day_hit=day, beta=beta,
            )
        if cum_alpha <= -barrier:
            return EventReturn(
                alpha=float(cum_alpha),
                zscore=float(cum_alpha / horizon_vol),
                label="SELL", barrier_hit="lower", day_hit=day, beta=beta,
            )

    # Time barrier — neither ±1.5σ hit within 21 days.
    stock_end = float(stock_prices.iloc[event_idx + 21])
    spy_end   = float(spy_prices.iloc[event_idx + 21])
    if stock_end <= 0 or spy_end <= 0:
        return None
    alpha  = np.log(stock_end / stock_start) - beta * np.log(spy_end / spy_start)
    zscore = alpha / horizon_vol
    return EventReturn(
        alpha=float(alpha),
        zscore=float(zscore),
        label=_label_from_zscore(zscore),
        barrier_hit="time",
        day_hit=21,
        beta=beta,
    )


def _label_from_zscore(zscore: float) -> str:
    """±1.5σ matches triple barrier thresholds — consistent labeling throughout."""
    if zscore >=  1.5: return "BUY"
    if zscore <= -1.5: return "SELL"
    return "HOLD"


def _build_prompt(ticker: str, form: str, filed: str, text: str, financials: dict) -> str:
    fin_str = ", ".join(f"{k}={v}" for k, v in financials.items() if v is not None) or "unavailable"
    return (
        "You are a financial analyst. Analyze this SEC filing and classify the historical price impact.\n\n"
        f"Ticker: {ticker}\n"
        f"Filing type: {form}\n"
        f"Filed: {filed}\n"
        f"Financials: {fin_str}\n\n"
        f"Filing excerpt:\n{text[:2500].strip()}\n\n"
        'Respond in JSON: {"signal": "BUY|SELL|HOLD", "label_strength": 0.0-1.0, "reason": "one sentence"}'
    )


def _build_response(signal: str, event: EventReturn) -> str:
    # label_strength: how far the z-score is from zero, scaled to [0.40, 0.95]
    label_strength = round(min(0.95, 0.40 + min(abs(event.zscore), 3.0) / 3.0 * 0.55), 2)
    how = (
        f"via {event.barrier_hit} barrier on day {event.day_hit}"
        if event.barrier_hit != "time"
        else "at 21-day horizon"
    )
    reasons = {
        "BUY":  f"Abnormal return z-score +{event.zscore:.2f}σ ({how}): stock outperformed SPY post-filing.",
        "SELL": f"Abnormal return z-score {event.zscore:.2f}σ ({how}): stock underperformed SPY post-filing.",
        "HOLD": f"Abnormal return z-score {event.zscore:.2f}σ: no directional signal vs SPY at 21-day horizon.",
    }
    return json.dumps({"signal": signal, "label_strength": label_strength, "reason": reasons[signal]})


# ──────────────────────────────────────────────────────────────────────────────
# Per-ticker processor (runs in worker thread)
# ──────────────────────────────────────────────────────────────────────────────

def _process_ticker(
    ticker:   str,
    out_path: Path,
    file_lock: threading.Lock,
    cp_lock:   threading.Lock,
) -> tuple[int, int]:
    """
    Process one ticker: fetch filings, label, write to JSONL.
    Returns (examples_written, examples_skipped).
    """
    from backend.edgar_live import get_recent_filings, get_filing_text
    from backend.market_data import MarketData

    written = skipped = 0
    md = MarketData()

    try:
        filings = get_recent_filings(ticker, forms=["10-Q", "10-K"], days=1095, limit=12)
    except Exception as exc:
        logger.warning("[%s] filings fetch failed: %s", ticker, exc)
        _mark_done(out_path, ticker, cp_lock)
        return 0, 0

    for filing in filings:
        filed       = filing.get("filed", "")
        form        = filing.get("form", "")
        accession   = filing.get("accession", "")
        cik         = filing.get("cik", "")
        primary_doc = filing.get("primary_doc", "")

        if not (filed and accession and cik):
            continue

        event = _compute_event_return(ticker, filed)
        if event is None:
            skipped += 1
            continue
        # Drop weak non-HOLD signals: a barrier hit with |z| < 0.5 is
        # ambiguous and noisy for training. Keep all time-barrier HOLDs
        # so the model has enough examples of "no directional signal."
        if event.barrier_hit != "time" and abs(event.zscore) < 0.5:
            skipped += 1
            continue

        time.sleep(0.35)  # EDGAR rate limit
        text = get_filing_text(accession, cik, primary_doc=primary_doc, max_chars=3000)
        if not text or len(text) < 200:
            skipped += 1
            continue

        try:
            detail     = md.stock_detail(ticker)
            financials = {
                "pe_ratio":       detail.get("pe_ratio")       if isinstance(detail, dict) else None,
                "revenue_growth": detail.get("revenue_growth") if isinstance(detail, dict) else None,
                "market_cap":     detail.get("market_cap")     if isinstance(detail, dict) else None,
            }
        except Exception:
            financials = {}

        signal = event.label
        record = {
            "messages": [
                {"role": "user",      "content": _build_prompt(ticker, form, filed, text, financials)},
                {"role": "assistant", "content": _build_response(signal, event)},
            ],
            "meta": {
                "ticker":      ticker,
                "form":        form,
                "filed":       filed,
                "outcome":     signal,
                "zscore":      round(event.zscore, 4),
                "alpha":       round(event.alpha, 4),
                "beta":        round(event.beta, 4),
                "barrier_hit": event.barrier_hit,
                "day_hit":     event.day_hit,
            },
        }

        with file_lock:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        written += 1
        logger.info("  + %s %s %s → %s (z=%.2f, day=%d)", ticker, form, filed, signal, event.zscore, event.day_hit)

    _mark_done(out_path, ticker, cp_lock)
    return written, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Main dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(tickers: list[str], out_path: Path, workers: int = 3) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip tickers already in checkpoint
    done_set  = _load_checkpoint(out_path)
    remaining = [t for t in tickers if t not in done_set]

    if not remaining:
        logger.info("All %d tickers already processed (checkpoint). Nothing to do.", len(tickers))
        return 0

    logger.info(
        "Processing %d tickers (%d already done, %d remaining) with %d workers",
        len(tickers), len(done_set), len(remaining), workers,
    )

    file_lock = threading.Lock()
    cp_lock   = threading.Lock()
    total_written = total_skipped = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_ticker, ticker, out_path, file_lock, cp_lock): ticker
            for ticker in remaining
        }
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            try:
                w, s = future.result()
                total_written  += w
                total_skipped  += s
            except Exception as exc:
                logger.warning("[%s] unhandled error: %s", ticker, exc)

            if completed % 10 == 0 or completed == len(remaining):
                logger.info(
                    "Progress: %d/%d tickers done | %d examples written so far",
                    completed, len(remaining), total_written,
                )

    logger.info(
        "Finished. %d examples written, %d skipped → %s",
        total_written, total_skipped, out_path,
    )
    return total_written


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SEC filing classifier training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Universe sizes and estimated run times (3 workers):
  default    ~40 tickers    ~5 min
  nasdaq100  ~100 tickers   ~12 min
  sp500      ~500 tickers   ~60 min
  sec_all    ~13000 tickers ~6 hrs

Safe to kill and re-run — completed tickers are checkpointed.
        """,
    )
    parser.add_argument(
        "--universe",
        choices=["default", "sp500", "nasdaq100", "sec_all"],
        default="default",
        help="Predefined ticker universe (ignored if --tickers is set)",
    )
    parser.add_argument("--tickers", nargs="*", help="Explicit ticker list (overrides --universe)")
    parser.add_argument("--out",     default=str(_DEFAULT_OUT), help="Output JSONL path")
    parser.add_argument("--workers", type=int, default=3,       help="Parallel worker threads (default: 3)")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe == "sp500":
        tickers = _fetch_sp500()
    elif args.universe == "nasdaq100":
        tickers = _fetch_nasdaq100()
    elif args.universe == "sec_all":
        tickers = _fetch_sec_all()
    else:
        tickers = _tickers_from_watchlist()
        if not tickers:
            logger.error(
                "No tickers found in watchlist. Add tickers via RAPHI settings "
                "or pass --universe sp500 / nasdaq100 / sec_all explicitly."
            )
            raise SystemExit(1)

    tickers = list(dict.fromkeys(tickers))  # deduplicate, preserve order
    logger.info("Universe: %s | %d tickers | workers: %d", args.universe, len(tickers), args.workers)
    build_dataset(tickers, Path(args.out), workers=args.workers)


if __name__ == "__main__":
    main()