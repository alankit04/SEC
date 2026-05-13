# backend/conviction_store.py
"""
conviction_store.py — Conviction Ledger data layer

Append-only JSONL store for RAPHI research output tracking.
Two files:
  convictions.jsonl  — one line per research output, write-once
  resolutions.jsonl  — one line per resolved lookback window
"""

import json
import logging
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf

logger = logging.getLogger("raphi.convictions")

BASE_DIR         = Path(__file__).parent.parent
LEDGER_DIR       = BASE_DIR / ".raphi_audit" / "conviction_ledger"
CONVICTIONS_FILE = LEDGER_DIR / "convictions.jsonl"
RESOLUTIONS_FILE = LEDGER_DIR / "resolutions.jsonl"

VALID_DIRECTIONS   = {"LONG", "SHORT", "NEUTRAL", "HOLD"}
VALID_TRENDS       = {"accelerating", "decelerating", "stable"}
VALID_SIGNAL_VIEWS = {"Positive", "Negative", "Neutral"}
VALID_SOURCES      = {"memo", "signal_query", "chat"}

# Noise band for NEUTRAL direction confirmation (within ±1.5% = CONFIRMED)
NEUTRAL_BAND_PCT = 1.5
# Noise band for SEC trend (within ±3% = INCONCLUSIVE)
SEC_NOISE_BAND_PCT = 3.0


def _ensure_dir() -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.chmod(0o700)


def _rand3() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=3))


def _fetch_vix() -> Optional[float]:
    try:
        data = yf.Ticker("^VIX").history(period="2d")
        if not data.empty:
            return round(float(data["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.warning("VIX fetch failed at conviction creation: %s", e)
    return None


def write_conviction(
    ticker: str,
    ml_direction: str,
    ml_probability: float,
    ml_model_version: str,
    sec_trend: Optional[str],
    sec_latest_revenue: Optional[float],
    sec_quarters_used: Optional[int],
    sec_next_filing_due: Optional[str],
    signal_view: str,
    conviction: str,
    source: str,
    entry_price: float,
) -> str:
    """
    Append one conviction to convictions.jsonl. Returns conviction_id.
    Call this after every validated research output.
    """
    _ensure_dir()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    conviction_id = f"cvx-{date_str}-{ticker.upper()}-{_rand3()}"

    vix = _fetch_vix()

    obj = {
        "id": conviction_id,
        "ticker": ticker.upper(),
        "date": now.isoformat(),
        "entry_price": entry_price,
        "ml": {
            "direction": ml_direction,
            "probability": ml_probability,
            "model_version": ml_model_version,
        },
        "sec": {
            "trend": sec_trend,
            "latest_revenue": sec_latest_revenue,
            "quarters_used": sec_quarters_used,
            "next_filing_due": sec_next_filing_due,
        },
        "signal_view": signal_view,
        "conviction": conviction,
        "source": source,
        "vix_at_creation": vix,
        "lookbacks_due": {
            "30d": (now + timedelta(days=30)).strftime("%Y-%m-%d"),
            "60d": (now + timedelta(days=60)).strftime("%Y-%m-%d"),
            "90d": (now + timedelta(days=90)).strftime("%Y-%m-%d"),
        },
    }

    with open(CONVICTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

    logger.info("Conviction written: %s", conviction_id)
    return conviction_id


def _read_convictions() -> dict:
    """Return dict[conviction_id → conviction_obj]."""
    if not CONVICTIONS_FILE.exists():
        return {}
    result = {}
    with open(CONVICTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                result[obj["id"]] = obj
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipped malformed conviction line")
    return result


def _read_resolved_set() -> set:
    """Return set of (conviction_id, lookback) that have been resolved."""
    if not RESOLUTIONS_FILE.exists():
        return set()
    resolved = set()
    with open(RESOLUTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                resolved.add((obj["conviction_id"], obj["lookback"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return resolved


def _append_resolution(res: dict) -> None:
    _ensure_dir()
    with open(RESOLUTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


def _fetch_price_spy_vix(ticker: str) -> tuple:
    """Fetch latest close for ticker, SPY, and VIX. Returns (price, spy, vix)."""
    def _close(sym: str) -> Optional[float]:
        try:
            data = yf.Ticker(sym).history(period="5d")
            if not data.empty:
                return round(float(data["Close"].iloc[-1]), 2)
        except Exception as e:
            logger.warning("Price fetch failed for %s: %s", sym, e)
        return None

    return _close(ticker), _close("SPY"), _close("^VIX")


def _eval_ml(direction: str, vs_entry_pct: float) -> str:
    """Compute CONFIRMED/CONTRADICTED for an ML direction vs actual price change."""
    if direction == "LONG":
        return "CONFIRMED" if vs_entry_pct > 0 else "CONTRADICTED"
    if direction == "SHORT":
        return "CONFIRMED" if vs_entry_pct < 0 else "CONTRADICTED"
    if direction in ("NEUTRAL", "HOLD"):
        return "CONFIRMED" if abs(vs_entry_pct) <= NEUTRAL_BAND_PCT else "CONTRADICTED"
    return "CONTRADICTED"


def check_pending() -> dict:
    """
    Resolve all pending ML lookback windows whose due dates have passed.
    Returns {resolved: int, still_pending: int, errors: list[str]}.
    Idempotent — safe to call on every liveRefresh cycle.
    """
    _ensure_dir()
    today = datetime.now(timezone.utc).date()
    convictions = _read_convictions()
    resolved_set = _read_resolved_set()

    resolved_count = 0
    still_pending  = 0
    errors: list = []

    for cid, conv in convictions.items():
        ticker      = conv["ticker"]
        entry_price = conv.get("entry_price", 0)
        ml_dir      = conv.get("ml", {}).get("direction", "")
        lookbacks   = conv.get("lookbacks_due", {})

        # ── ML price checks (30d, 60d, 90d) ──────────────────────────
        for window in ("30d", "60d", "90d"):
            if (cid, window) in resolved_set:
                continue
            due_str = lookbacks.get(window)
            if not due_str:
                continue
            due_date = datetime.strptime(due_str, "%Y-%m-%d").date()
            if today < due_date:
                still_pending += 1
                continue

            try:
                price_close, spy_close, vix_close = _fetch_price_spy_vix(ticker)
                if price_close is None or entry_price == 0:
                    errors.append(f"{cid}:{window} — price fetch returned None")
                    continue

                vs_entry = round((price_close - entry_price) / entry_price * 100, 2)
                vs_spy   = round(vs_entry - (spy_close or 0), 2) if spy_close else None

                ml_result = _eval_ml(ml_dir, vs_entry)

                res = {
                    "conviction_id": cid,
                    "lookback":      window,
                    "resolved_date": today.isoformat(),
                    "ml_result":     ml_result,
                    "price_at_check": price_close,
                    "vs_entry_pct":  vs_entry,
                    "vs_spy_pct":    vs_spy,
                    "vix_at_check":  vix_close,
                }
                _append_resolution(res)
                resolved_set.add((cid, window))
                resolved_count += 1

            except Exception as e:
                errors.append(f"{cid}:{window} — {e}")

        # ── SEC filing check ──────────────────────────────────────────
        if (cid, "sec") not in resolved_set:
            sec_due = conv.get("sec", {}).get("next_filing_due")
            if sec_due and datetime.strptime(sec_due, "%Y-%m-%d").date() <= today:
                sec_result = _check_sec_filing(cid, conv)
                if sec_result == "PENDING":
                    still_pending += 1
                elif sec_result == "ERROR":
                    errors.append(f"{cid}:sec — EDGAR data unavailable")
                else:
                    resolved_set.add((cid, "sec"))
                    resolved_count += 1
            elif sec_due:
                still_pending += 1

    return {"resolved": resolved_count, "still_pending": still_pending, "errors": errors}


def _sec_latest_revenue_after(ticker: str, after_date: str) -> Optional[tuple]:
    """
    Read local EDGAR XBRL data for ticker. Return (revenue, period_date) for the
    first quarter with period > after_date. Returns None if no new data found.
    """
    try:
        from sec_data import SECData
        sec = SECData(BASE_DIR)
        financials = sec.company_financial_entries(ticker)
        if not financials:
            return None

        revenue_tags = {
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "RevenueFromContractWithCustomer",
        }

        after = datetime.strptime(after_date, "%Y-%m-%d").date()
        candidates = []
        for entry in financials:
            tag = entry.get("tag", "")
            if tag not in revenue_tags:
                continue
            period = entry.get("period") or entry.get("ddate", "")
            try:
                period_date = datetime.strptime(period[:10], "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if period_date > after:
                val = entry.get("val") or entry.get("value")
                if val:
                    candidates.append((float(val), period[:10]))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]
    except Exception as e:
        logger.warning("SEC filing lookup failed for %s: %s", ticker, e)
        return None


def _check_sec_filing(conviction_id: str, conv: dict) -> str:
    """
    Check SEC trend for a conviction. Returns:
      "CONFIRMED" | "CONTRADICTED" | "INCONCLUSIVE" — and appends to resolutions.jsonl
      "PENDING"  — EDGAR data not yet available, nothing written
      "ERROR"    — unexpected failure, nothing written
    """
    ticker    = conv["ticker"]
    sec_info  = conv.get("sec", {})
    trend     = sec_info.get("trend")
    baseline  = sec_info.get("latest_revenue")
    conv_date = conv.get("date", "")[:10]

    if not trend or baseline is None:
        return "PENDING"

    result = _sec_latest_revenue_after(ticker, conv_date)
    if result is None:
        return "PENDING"

    actual_revenue, period_date = result
    delta_pct = round((actual_revenue - baseline) / baseline * 100, 2) if baseline else None

    if delta_pct is None:
        return "ERROR"

    if abs(delta_pct) <= SEC_NOISE_BAND_PCT:
        sec_result = "INCONCLUSIVE"
    elif trend == "accelerating" and delta_pct > SEC_NOISE_BAND_PCT:
        sec_result = "CONFIRMED"
    elif trend == "decelerating" and delta_pct < -SEC_NOISE_BAND_PCT:
        sec_result = "CONFIRMED"
    else:
        sec_result = "CONTRADICTED"

    _append_resolution({
        "conviction_id":     conviction_id,
        "lookback":          "sec",
        "resolved_date":     datetime.now(timezone.utc).date().isoformat(),
        "sec_result":        sec_result,
        "actual_revenue":    actual_revenue,
        "revenue_delta_pct": delta_pct,
        "period_date":       period_date,
    })

    return sec_result


def get_accuracy_stats(ticker: Optional[str] = None) -> dict:
    """
    Compute accuracy stats across all resolved convictions.
    Pending windows (no resolution yet) are excluded from the denominator.
    Returns a dict suitable for the /api/convictions/stats endpoint.
    """
    convictions = _read_convictions()
    if not convictions:
        return _empty_stats()

    # Build resolution lookup: {conviction_id → {lookback → resolution_obj}}
    resolutions: dict = {}
    if RESOLUTIONS_FILE.exists():
        with open(RESOLUTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cid = obj["conviction_id"]
                    lb  = obj["lookback"]
                    resolutions.setdefault(cid, {})[lb] = obj
                except (json.JSONDecodeError, KeyError):
                    pass

    if ticker:
        convictions = {k: v for k, v in convictions.items()
                       if v["ticker"].upper() == ticker.upper()}

    counts: dict = {
        w: {"confirmed": 0, "contradicted": 0}
        for w in ("30d", "60d", "90d", "sec")
    }
    pending_count = 0
    total         = len(convictions)
    vix_buckets: dict = {"low": [], "mid": [], "high": []}

    for cid, conv in convictions.items():
        conv_resolutions = resolutions.get(cid, {})
        has_any = bool(conv_resolutions)
        if not has_any:
            pending_count += 1

        vix = conv.get("vix_at_creation")
        bucket = None
        if vix is not None:
            bucket = "low" if vix < 15 else ("mid" if vix <= 25 else "high")

        for window in ("30d", "60d", "90d"):
            res = conv_resolutions.get(window)
            if not res:
                continue
            result_key = res.get("ml_result", "")
            if result_key == "CONFIRMED":
                counts[window]["confirmed"] += 1
                if bucket:
                    vix_buckets[bucket].append(res.get("vs_spy_pct", 0))
            elif result_key == "CONTRADICTED":
                counts[window]["contradicted"] += 1

        sec_res = conv_resolutions.get("sec")
        if sec_res:
            result_key = sec_res.get("sec_result", "")
            if result_key == "CONFIRMED":
                counts["sec"]["confirmed"] += 1
            elif result_key == "CONTRADICTED":
                counts["sec"]["contradicted"] += 1

    def _acc(window: str) -> Optional[float]:
        c = counts[window]["confirmed"]
        d = c + counts[window]["contradicted"]
        return round(c / d * 100, 1) if d > 0 else None

    def _regime_acc(bucket_name: str) -> Optional[float]:
        vals = vix_buckets[bucket_name]
        if not vals:
            return None
        return round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1)

    return {
        "ml_accuracy_30d":   _acc("30d"),
        "ml_accuracy_60d":   _acc("60d"),
        "ml_accuracy_90d":   _acc("90d"),
        "sec_accuracy":      _acc("sec"),
        "total_convictions": total,
        "pending_count":     pending_count,
        "total_resolved":    sum(counts[w]["confirmed"] + counts[w]["contradicted"]
                                 for w in ("30d", "60d", "90d", "sec")),
        "regime_low_acc":    _regime_acc("low"),
        "regime_mid_acc":    _regime_acc("mid"),
        "regime_high_acc":   _regime_acc("high"),
        "ticker_filter":     ticker,
    }


def _empty_stats() -> dict:
    return {
        "ml_accuracy_30d": None, "ml_accuracy_60d": None, "ml_accuracy_90d": None,
        "sec_accuracy": None, "total_convictions": 0, "pending_count": 0,
        "total_resolved": 0, "regime_low_acc": None, "regime_mid_acc": None,
        "regime_high_acc": None, "ticker_filter": None,
    }


def _window_status(resolution: Optional[dict], window: str, conv: dict) -> dict:
    """Format a single ML lookback window for the ledger table."""
    if resolution:
        result = resolution.get("ml_result", "")
        return {
            "status":        result,
            "vs_entry_pct":  resolution.get("vs_entry_pct"),
            "vs_spy_pct":    resolution.get("vs_spy_pct"),
            "resolved_date": resolution.get("resolved_date"),
        }
    due = conv.get("lookbacks_due", {}).get(window)
    return {"status": "PENDING", "due_date": due}


def _sec_window_status(resolution: Optional[dict], conv: dict) -> dict:
    """Format the SEC lookback window for the ledger table."""
    if resolution:
        return {
            "status":            resolution.get("sec_result", ""),
            "revenue_delta_pct": resolution.get("revenue_delta_pct"),
            "resolved_date":     resolution.get("resolved_date"),
        }
    due = conv.get("sec", {}).get("next_filing_due")
    if not due:
        return {"status": "N/A"}
    return {"status": "PENDING", "due_date": due}


def get_ledger(page: int = 1, ticker: Optional[str] = None) -> dict:
    """
    Full conviction history joined with resolutions.
    Sorted by date descending. Paginated at 50 per page.
    """
    PAGE_SIZE = 50
    convictions = _read_convictions()

    resolutions: dict = {}
    if RESOLUTIONS_FILE.exists():
        with open(RESOLUTIONS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    resolutions.setdefault(obj["conviction_id"], {})[obj["lookback"]] = obj
                except (json.JSONDecodeError, KeyError):
                    pass

    items = list(convictions.values())
    if ticker:
        items = [i for i in items if i["ticker"].upper() == ticker.upper()]

    items.sort(key=lambda x: x.get("date", ""), reverse=True)

    total      = len(items)
    start      = (page - 1) * PAGE_SIZE
    end        = start + PAGE_SIZE
    page_items = items[start:end]

    rows = []
    for conv in page_items:
        cid = conv["id"]
        res = resolutions.get(cid, {})
        rows.append({
            "id":              cid,
            "ticker":          conv["ticker"],
            "date":            conv.get("date", "")[:10],
            "ml_direction":    conv.get("ml", {}).get("direction"),
            "ml_prob":         conv.get("ml", {}).get("probability"),
            "sec_trend":       conv.get("sec", {}).get("trend"),
            "signal_view":     conv.get("signal_view"),
            "conviction":      conv.get("conviction"),
            "source":          conv.get("source"),
            "vix_at_creation": conv.get("vix_at_creation"),
            "lookbacks_due":   conv.get("lookbacks_due", {}),
            "windows": {
                "30d": _window_status(res.get("30d"), "30d", conv),
                "60d": _window_status(res.get("60d"), "60d", conv),
                "90d": _window_status(res.get("90d"), "90d", conv),
                "sec": _sec_window_status(res.get("sec"), conv),
            },
        })

    return {
        "rows":        rows,
        "total":       total,
        "page":        page,
        "page_size":   PAGE_SIZE,
        "total_pages": max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
    }
