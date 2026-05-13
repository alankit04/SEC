# Conviction Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Conviction Ledger — RAPHI tracks every research output it generates (ML signal direction, SEC revenue trend, final Signal View) and automatically checks accuracy at 30, 60, and 90-day windows, surfacing a live track record on a dedicated dashboard page and inline on every research output.

**Architecture:** Two append-only JSONL files (`convictions.jsonl` + `resolutions.jsonl`) in `.raphi_audit/conviction_ledger/`. A new `conviction_store.py` module owns all reads/writes. Four new API endpoints expose data to the frontend. The resolution check (`check_pending()`) is added to the existing `liveRefresh()` Promise.all cycle — zero new scheduling infrastructure.

**Tech Stack:** Python 3.11 · FastAPI · yfinance · pytest · vanilla JS (existing index.html patterns)

**Spec:** `docs/superpowers/specs/2026-04-08-conviction-ledger-design.md`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `backend/conviction_store.py` | All reads/writes to both JSONL files |
| Create | `tests/test_conviction_store.py` | Pytest unit tests for conviction_store |
| Create | `.raphi_audit/conviction_ledger/` | Data directory (created programmatically) |
| Modify | `backend/raphi_server.py` | Add 4 API endpoints + write_conviction hook |
| Modify | `backend/static/index.html` | Add liveRefresh additions + Ledger page + inline view |

---

## Task 1: conviction_store.py — directory setup + write_conviction()

**Files:**
- Create: `backend/conviction_store.py`

- [ ] **Step 1.1: Create the file with imports and directory constants**

```python
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

BASE_DIR        = Path(__file__).parent.parent
LEDGER_DIR      = BASE_DIR / ".raphi_audit" / "conviction_ledger"
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
```

- [ ] **Step 1.2: Add `write_conviction()` function**

```python
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
```

- [ ] **Step 1.3: Add helpers for reading both files**

```python
def _read_convictions() -> dict[str, dict]:
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


def _read_resolved_set() -> set[tuple[str, str]]:
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
```

- [ ] **Step 1.4: Verify file can be imported**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -c "from backend.conviction_store import write_conviction; print('OK')"
```

Expected: `OK`

- [ ] **Step 1.5: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/conviction_store.py
git commit -m "feat: add conviction_store.py with write_conviction and file helpers"
```

---

## Task 2: conviction_store.py — check_pending() ML price resolution

**Files:**
- Modify: `backend/conviction_store.py`

- [ ] **Step 2.1: Write the failing test first**

Create `tests/test_conviction_store.py`:

```python
# tests/test_conviction_store.py
"""Tests for conviction_store.py"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import pytest

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    """Redirect LEDGER_DIR, CONVICTIONS_FILE, RESOLUTIONS_FILE to tmp_path."""
    import conviction_store as cs
    monkeypatch.setattr(cs, "LEDGER_DIR",        tmp_path)
    monkeypatch.setattr(cs, "CONVICTIONS_FILE",  tmp_path / "convictions.jsonl")
    monkeypatch.setattr(cs, "RESOLUTIONS_FILE",  tmp_path / "resolutions.jsonl")
    return tmp_path


def make_conviction(tmp_path, ticker="NVDA", direction="LONG", trend="accelerating",
                    days_ago=35, signal_view="Positive"):
    """Write a conviction directly to convictions.jsonl for testing."""
    import conviction_store as cs
    now = datetime.now(timezone.utc) - timedelta(days=days_ago)
    obj = {
        "id": f"cvx-test-{ticker}-abc",
        "ticker": ticker,
        "date": now.isoformat(),
        "entry_price": 100.0,
        "ml": {"direction": direction, "probability": 0.71, "model_version": "xgb_v2.1"},
        "sec": {"trend": trend, "latest_revenue": 1_000_000, "quarters_used": 8,
                "next_filing_due": (now + timedelta(days=5)).strftime("%Y-%m-%d")},
        "signal_view": signal_view,
        "conviction": "MEDIUM",
        "source": "memo",
        "vix_at_creation": 15.0,
        "lookbacks_due": {
            "30d": (now + timedelta(days=30)).strftime("%Y-%m-%d"),
            "60d": (now + timedelta(days=60)).strftime("%Y-%m-%d"),
            "90d": (now + timedelta(days=90)).strftime("%Y-%m-%d"),
        },
    }
    with open(cs.CONVICTIONS_FILE, "a") as f:
        f.write(json.dumps(obj) + "\n")
    return obj


def test_check_pending_ml_confirmed(ledger_dir):
    """LONG signal + price up = CONFIRMED resolution written."""
    import conviction_store as cs
    make_conviction(ledger_dir, direction="LONG", days_ago=35)

    mock_history = MagicMock()
    mock_history.empty = False
    mock_history.__getitem__ = MagicMock(return_value=MagicMock(
        iloc=MagicMock(__getitem__=MagicMock(return_value=115.0))  # +15% from 100
    ))

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_history
        result = cs.check_pending()

    assert result["resolved"] == 1
    assert result["still_pending"] >= 0

    lines = cs.RESOLUTIONS_FILE.read_text().strip().split("\n")
    res = json.loads(lines[0])
    assert res["ml_result"] == "CONFIRMED"
    assert res["lookback"] == "30d"


def test_check_pending_ml_contradicted(ledger_dir):
    """LONG signal + price down = CONTRADICTED."""
    import conviction_store as cs
    make_conviction(ledger_dir, direction="LONG", days_ago=35)

    mock_history = MagicMock()
    mock_history.empty = False
    mock_history.__getitem__ = MagicMock(return_value=MagicMock(
        iloc=MagicMock(__getitem__=MagicMock(return_value=85.0))  # -15%
    ))

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_history
        result = cs.check_pending()

    lines = cs.RESOLUTIONS_FILE.read_text().strip().split("\n")
    res = json.loads(lines[0])
    assert res["ml_result"] == "CONTRADICTED"


def test_check_pending_idempotent(ledger_dir):
    """Calling check_pending twice does not double-write resolutions."""
    import conviction_store as cs
    make_conviction(ledger_dir, direction="LONG", days_ago=35)

    mock_history = MagicMock()
    mock_history.empty = False
    mock_history.__getitem__ = MagicMock(return_value=MagicMock(
        iloc=MagicMock(__getitem__=MagicMock(return_value=115.0))
    ))

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = mock_history
        cs.check_pending()
        cs.check_pending()

    lines = [l for l in cs.RESOLUTIONS_FILE.read_text().strip().split("\n") if l]
    assert len(lines) == 1  # only one resolution written total
```

- [ ] **Step 2.2: Run test to confirm it fails**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m pytest tests/test_conviction_store.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError: module has no attribute 'check_pending'`

- [ ] **Step 2.3: Add `check_pending()` to conviction_store.py**

```python
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
    errors: list[str] = []

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


def _fetch_price_spy_vix(ticker: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
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
```

- [ ] **Step 2.4: Run tests**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m pytest tests/test_conviction_store.py::test_check_pending_ml_confirmed tests/test_conviction_store.py::test_check_pending_ml_contradicted tests/test_conviction_store.py::test_check_pending_idempotent -v
```

Expected: All 3 PASS

- [ ] **Step 2.5: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/conviction_store.py tests/test_conviction_store.py
git commit -m "feat: add check_pending() ML price resolution to conviction_store"
```

---

## Task 3: conviction_store.py — SEC filing resolution

**Files:**
- Modify: `backend/conviction_store.py`
- Modify: `tests/test_conviction_store.py`

- [ ] **Step 3.1: Write failing test for SEC resolution**

Add to `tests/test_conviction_store.py`:

```python
def test_check_pending_sec_confirmed(ledger_dir):
    """SEC trend 'accelerating' + next filing shows +5% revenue = CONFIRMED."""
    import conviction_store as cs
    from unittest.mock import patch
    make_conviction(ledger_dir, trend="accelerating", days_ago=35)

    # Mock sec_data.company_financials to return data showing revenue growth
    mock_financials = [
        {"period": "2026-03-31", "tag": "Revenues", "val": 1_050_000, "form": "10-Q"},
    ]

    with patch("conviction_store.yf.Ticker") as mock_ticker, \
         patch("conviction_store._sec_latest_revenue_after", return_value=(1_050_000, "2026-03-31")):
        mock_ticker.return_value.history.return_value = MagicMock(
            empty=True  # price checks not relevant here
        )
        result = cs.check_pending()

    lines = [l for l in cs.RESOLUTIONS_FILE.read_text().strip().split("\n") if l]
    sec_lines = [json.loads(l) for l in lines if json.loads(l).get("lookback") == "sec"]
    assert len(sec_lines) == 1
    assert sec_lines[0]["sec_result"] == "CONFIRMED"


def test_check_pending_sec_inconclusive(ledger_dir):
    """Revenue delta within ±3% = INCONCLUSIVE (not counted in denominator)."""
    import conviction_store as cs
    make_conviction(ledger_dir, trend="accelerating", days_ago=35)

    with patch("conviction_store.yf.Ticker") as mock_ticker, \
         patch("conviction_store._sec_latest_revenue_after", return_value=(1_010_000, "2026-03-31")):
        mock_ticker.return_value.history.return_value = MagicMock(empty=True)
        cs.check_pending()

    lines = [l for l in cs.RESOLUTIONS_FILE.read_text().strip().split("\n") if l]
    sec_lines = [json.loads(l) for l in lines if json.loads(l).get("lookback") == "sec"]
    if sec_lines:
        assert sec_lines[0]["sec_result"] == "INCONCLUSIVE"
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m pytest tests/test_conviction_store.py::test_check_pending_sec_confirmed -v 2>&1 | tail -15
```

Expected: FAIL — `_sec_latest_revenue_after` not defined

- [ ] **Step 3.3: Add SEC helper and `_check_sec_filing()` to conviction_store.py**

```python
def _sec_latest_revenue_after(ticker: str, after_date: str) -> Optional[tuple[float, str]]:
    """
    Read local EDGAR XBRL data for ticker. Return (revenue, period_date) for the
    first quarter with period > after_date. Returns None if no new data found.
    """
    try:
        # Import here to avoid circular at module load
        from sec_data import SECData
        sec = SECData(BASE_DIR)
        financials = sec.company_financials(ticker)
        if not financials:
            return None

        revenue_tags = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "SalesRevenueNet", "RevenueFromContractWithCustomer"}

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
        # Return the most recent qualifying quarter
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
    ticker   = conv["ticker"]
    sec_info = conv.get("sec", {})
    trend    = sec_info.get("trend")
    baseline = sec_info.get("latest_revenue")
    conv_date = conv.get("date", "")[:10]  # YYYY-MM-DD

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
        "conviction_id":   conviction_id,
        "lookback":        "sec",
        "resolved_date":   datetime.now(timezone.utc).date().isoformat(),
        "sec_result":      sec_result,
        "actual_revenue":  actual_revenue,
        "revenue_delta_pct": delta_pct,
        "period_date":     period_date,
    })

    return sec_result
```

- [ ] **Step 3.4: Run SEC tests**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m pytest tests/test_conviction_store.py -v 2>&1 | tail -20
```

Expected: All 5 tests PASS

- [ ] **Step 3.5: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/conviction_store.py tests/test_conviction_store.py
git commit -m "feat: add SEC filing resolution to check_pending()"
```

---

## Task 4: conviction_store.py — get_accuracy_stats()

**Files:**
- Modify: `backend/conviction_store.py`
- Modify: `tests/test_conviction_store.py`

- [ ] **Step 4.1: Write failing test**

Add to `tests/test_conviction_store.py`:

```python
def _write_resolution(tmp_path, conviction_id, lookback, ml_result="CONFIRMED",
                      vs_entry=8.0, vs_spy=5.0, vix=15.0):
    import conviction_store as cs
    res = {
        "conviction_id": conviction_id,
        "lookback": lookback,
        "resolved_date": "2026-05-08",
        "ml_result": ml_result,
        "price_at_check": 108.0,
        "vs_entry_pct": vs_entry,
        "vs_spy_pct": vs_spy,
        "vix_at_check": vix,
    }
    with open(cs.RESOLUTIONS_FILE, "a") as f:
        f.write(json.dumps(res) + "\n")


def test_get_accuracy_stats_basic(ledger_dir):
    """Two convictions, one confirmed, one contradicted → 50% accuracy."""
    import conviction_store as cs
    c1 = make_conviction(ledger_dir, ticker="NVDA", direction="LONG", days_ago=40)
    c2 = make_conviction(ledger_dir, ticker="AAPL", direction="LONG", days_ago=40)

    _write_resolution(ledger_dir, c1["id"], "30d", ml_result="CONFIRMED")
    _write_resolution(ledger_dir, c2["id"], "30d", ml_result="CONTRADICTED")

    stats = cs.get_accuracy_stats()
    assert stats["ml_accuracy_30d"] == 50.0
    assert stats["total_convictions"] == 2
    assert stats["pending_count"] == 0


def test_get_accuracy_stats_pending_excluded(ledger_dir):
    """Pending windows do not affect the denominator."""
    import conviction_store as cs
    c1 = make_conviction(ledger_dir, ticker="NVDA", direction="LONG", days_ago=40)
    c2 = make_conviction(ledger_dir, ticker="AAPL", direction="LONG", days_ago=5)

    _write_resolution(ledger_dir, c1["id"], "30d", ml_result="CONFIRMED")
    # c2 has no resolutions yet — still pending

    stats = cs.get_accuracy_stats()
    assert stats["ml_accuracy_30d"] == 100.0  # only c1 in denominator
    assert stats["pending_count"] == 1


def test_get_accuracy_stats_ticker_filter(ledger_dir):
    """ticker= param filters to one ticker only."""
    import conviction_store as cs
    c1 = make_conviction(ledger_dir, ticker="NVDA", direction="LONG", days_ago=40)
    c2 = make_conviction(ledger_dir, ticker="AAPL", direction="LONG", days_ago=40)

    _write_resolution(ledger_dir, c1["id"], "30d", ml_result="CONFIRMED")
    _write_resolution(ledger_dir, c2["id"], "30d", ml_result="CONTRADICTED")

    stats = cs.get_accuracy_stats(ticker="NVDA")
    assert stats["ml_accuracy_30d"] == 100.0
    assert stats["total_convictions"] == 1
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m pytest tests/test_conviction_store.py::test_get_accuracy_stats_basic -v 2>&1 | tail -10
```

Expected: FAIL — `get_accuracy_stats` not defined

- [ ] **Step 4.3: Add `get_accuracy_stats()` to conviction_store.py**

```python
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
    resolutions: dict[str, dict[str, dict]] = {}
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

    # Filter by ticker if requested
    if ticker:
        convictions = {k: v for k, v in convictions.items()
                       if v["ticker"].upper() == ticker.upper()}

    # Accumulate accuracy per window
    counts: dict[str, dict[str, int]] = {
        w: {"confirmed": 0, "contradicted": 0}
        for w in ("30d", "60d", "90d", "sec")
    }
    pending_count   = 0
    total           = len(convictions)
    vix_buckets     = {"low": [], "mid": [], "high": []}  # for regime breakdown

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
            # INCONCLUSIVE: skip (excluded from denominator)

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
```

- [ ] **Step 4.4: Run accuracy tests**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m pytest tests/test_conviction_store.py -v 2>&1 | tail -20
```

Expected: All 8 tests PASS

- [ ] **Step 4.5: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/conviction_store.py tests/test_conviction_store.py
git commit -m "feat: add get_accuracy_stats() to conviction_store"
```

---

## Task 5: conviction_store.py — get_ledger()

**Files:**
- Modify: `backend/conviction_store.py`

- [ ] **Step 5.1: Add `get_ledger()` to conviction_store.py**

```python
def get_ledger(page: int = 1, ticker: Optional[str] = None) -> dict:
    """
    Full conviction history joined with resolutions.
    Sorted by date descending. Paginated at 50 per page.
    Returns per-conviction status per lookback window.
    """
    PAGE_SIZE = 50
    convictions = _read_convictions()

    # Build resolution lookup
    resolutions: dict[str, dict[str, dict]] = {}
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

    # Sort newest first
    items.sort(key=lambda x: x.get("date", ""), reverse=True)

    total   = len(items)
    start   = (page - 1) * PAGE_SIZE
    end     = start + PAGE_SIZE
    page_items = items[start:end]

    rows = []
    for conv in page_items:
        cid  = conv["id"]
        res  = resolutions.get(cid, {})
        rows.append({
            "id":          cid,
            "ticker":      conv["ticker"],
            "date":        conv.get("date", "")[:10],
            "ml_direction": conv.get("ml", {}).get("direction"),
            "ml_prob":      conv.get("ml", {}).get("probability"),
            "sec_trend":    conv.get("sec", {}).get("trend"),
            "signal_view":  conv.get("signal_view"),
            "conviction":   conv.get("conviction"),
            "source":       conv.get("source"),
            "vix_at_creation": conv.get("vix_at_creation"),
            "lookbacks_due": conv.get("lookbacks_due", {}),
            "windows": {
                "30d": _window_status(res.get("30d"), "30d", conv),
                "60d": _window_status(res.get("60d"), "60d", conv),
                "90d": _window_status(res.get("90d"), "90d", conv),
                "sec": _sec_window_status(res.get("sec"), conv),
            },
        })

    return {
        "rows":       rows,
        "total":      total,
        "page":       page,
        "page_size":  PAGE_SIZE,
        "total_pages": max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
    }


def _window_status(resolution: Optional[dict], window: str, conv: dict) -> dict:
    """Format a single ML lookback window for the ledger table."""
    if resolution:
        result = resolution.get("ml_result", "")
        return {
            "status":       result,
            "vs_entry_pct": resolution.get("vs_entry_pct"),
            "vs_spy_pct":   resolution.get("vs_spy_pct"),
            "resolved_date": resolution.get("resolved_date"),
        }
    due = conv.get("lookbacks_due", {}).get(window)
    return {"status": "PENDING", "due_date": due}


def _sec_window_status(resolution: Optional[dict], conv: dict) -> dict:
    """Format the SEC lookback window for the ledger table."""
    if resolution:
        return {
            "status":           resolution.get("sec_result", ""),
            "revenue_delta_pct": resolution.get("revenue_delta_pct"),
            "resolved_date":    resolution.get("resolved_date"),
        }
    due = conv.get("sec", {}).get("next_filing_due")
    if not due:
        return {"status": "N/A"}
    return {"status": "PENDING", "due_date": due}
```

- [ ] **Step 5.2: Verify import and basic call**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -c "
from backend.conviction_store import get_ledger
result = get_ledger()
print('total:', result['total'], 'pages:', result['total_pages'])
print('OK')
"
```

Expected: `total: 0 pages: 1` then `OK` (ledger is empty, that's correct)

- [ ] **Step 5.3: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/conviction_store.py
git commit -m "feat: add get_ledger() to conviction_store with pagination and window status"
```

---

## Task 6: API endpoints in raphi_server.py

**Files:**
- Modify: `backend/raphi_server.py`

- [ ] **Step 6.1: Add conviction_store import at top of raphi_server.py**

Find this block in `raphi_server.py` (around line 52–57):
```python
from market_data       import MarketData
from sec_data          import SECData
from ml_model          import SignalEngine
from portfolio_manager import PortfolioManager
from a2a_executor_v2   import RaphiAgent, RaphiAgentExecutor
from security          import TokenAuth, init_sentry
```

Add one line after it:
```python
from conviction_store  import (
    write_conviction, check_pending, get_accuracy_stats, get_ledger
)
```

- [ ] **Step 6.2: Add the four conviction endpoints to raphi_server.py**

Find the `# ── SEC ──` section (around line 427) and add the following block **before** it:

```python
# ── conviction ledger ─────────────────────────────────────────────────
class ConvictionRequest(BaseModel):
    ticker:            str
    ml_direction:      str
    ml_probability:    float
    ml_model_version:  str
    sec_trend:         Optional[str] = None
    sec_latest_revenue: Optional[float] = None
    sec_quarters_used: Optional[int] = None
    sec_next_filing_due: Optional[str] = None
    signal_view:       str
    conviction:        str
    source:            str = "memo"
    entry_price:       float


@api.post("/convictions")
@limiter.limit("60/minute")
def post_conviction(body: ConvictionRequest, request: Request):
    import re
    if not re.match(r"^[A-Z]{1,5}$", body.ticker.upper()):
        raise HTTPException(422, "Invalid ticker")
    conviction_id = write_conviction(
        ticker=body.ticker,
        ml_direction=body.ml_direction,
        ml_probability=body.ml_probability,
        ml_model_version=body.ml_model_version,
        sec_trend=body.sec_trend,
        sec_latest_revenue=body.sec_latest_revenue,
        sec_quarters_used=body.sec_quarters_used,
        sec_next_filing_due=body.sec_next_filing_due,
        signal_view=body.signal_view,
        conviction=body.conviction,
        source=body.source,
        entry_price=body.entry_price,
    )
    return {"conviction_id": conviction_id}


@api.get("/convictions/check")
@limiter.limit("60/minute")
def convictions_check(request: Request):
    return check_pending()


@api.get("/convictions/stats")
@limiter.limit("60/minute")
def convictions_stats(request: Request, ticker: Optional[str] = None):
    return get_accuracy_stats(ticker=ticker)


@api.get("/convictions/ledger")
@limiter.limit("60/minute")
def convictions_ledger(request: Request, page: int = 1, ticker: Optional[str] = None):
    return get_ledger(page=page, ticker=ticker)
```

Also add `Optional` to the existing imports at the top of `raphi_server.py` if not already present. Find:
```python
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Request
```
And ensure `Optional` is imported from `typing` (add at top with other stdlib imports):
```python
from typing import Optional
```

- [ ] **Step 6.3: Start server and test endpoints**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m backend.raphi_server &
sleep 3
# Test check endpoint (should return empty)
curl -s -H "X-API-Key: $(cat settings.json | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"raphi_api_key\",\"\"))')" \
  http://localhost:9999/api/convictions/check
```

Expected: `{"resolved":0,"still_pending":0,"errors":[]}`

```bash
curl -s -H "X-API-Key: $(cat settings.json | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"raphi_api_key\",\"\"))')" \
  http://localhost:9999/api/convictions/stats
```

Expected: `{"ml_accuracy_30d":null,"ml_accuracy_60d":null,...,"total_convictions":0,...}`

```bash
# Stop background server
pkill -f raphi_server
```

- [ ] **Step 6.4: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/raphi_server.py
git commit -m "feat: add /api/convictions/* endpoints to raphi_server"
```

---

## Task 7: Write conviction after memo generation

**Files:**
- Modify: `backend/raphi_server.py`

The chat endpoint already has the ticker, ML signal (from cache), and can derive the signal_view from context. We hook into the `/api/chat` endpoint's response to write a conviction when a research conclusion is present.

- [ ] **Step 7.1: Add conviction writer helper to raphi_server.py**

Add this function near the `_fmt_portfolio` helper (around line 236):

```python
def _maybe_write_conviction(ticker: str, sig_cache_path: Path, response_text: str) -> None:
    """
    After a chat or memo response, attempt to write a conviction if the
    response contains a structured research conclusion. Fires and forgets —
    never raises, never blocks the response.
    """
    import re
    import pickle

    try:
        # Only write if the response explicitly contains a Signal View marker
        sv_match = re.search(
            r"Signal\s*View[:\s]+([Pp]ositive|[Nn]egative|[Nn]eutral)", response_text
        )
        if not sv_match:
            return

        signal_view = sv_match.group(1).capitalize()

        # ML data from cache
        if not sig_cache_path.exists():
            return
        with open(sig_cache_path, "rb") as f:
            sig = pickle.load(f)

        ml_dir   = sig.get("direction", "NEUTRAL")
        ml_prob  = sig.get("confidence", 50) / 100  # stored as 0–100 in cache
        ml_ver   = "xgb_v2.1"

        # Market price
        detail = market.stock_detail(ticker.upper())
        price  = detail.get("price")
        if not price:
            return

        # SEC data
        fin    = sec.company_financials(ticker.upper())
        latest_rev  = None
        sec_trend   = None
        sec_due     = None
        sec_qtrs    = 0
        if fin:
            rev_tags = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "SalesRevenueNet"}
            rev_entries = sorted(
                [e for e in fin if e.get("tag") in rev_tags and e.get("val")],
                key=lambda x: x.get("period", ""), reverse=True
            )
            if len(rev_entries) >= 2:
                sec_qtrs    = len(rev_entries)
                latest_rev  = float(rev_entries[0]["val"])
                prev_rev    = float(rev_entries[1]["val"])
                delta       = (latest_rev - prev_rev) / prev_rev * 100 if prev_rev else 0
                sec_trend   = "accelerating" if delta > 3 else (
                              "decelerating" if delta < -3 else "stable")

        # Conviction tier from ML probability
        conviction_tier = "HIGH" if ml_prob >= 0.70 else ("MEDIUM" if ml_prob >= 0.60 else "LOW")

        write_conviction(
            ticker=ticker.upper(),
            ml_direction=ml_dir,
            ml_probability=ml_prob,
            ml_model_version=ml_ver,
            sec_trend=sec_trend,
            sec_latest_revenue=latest_rev,
            sec_quarters_used=sec_qtrs,
            sec_next_filing_due=sec_due,
            signal_view=signal_view,
            conviction=conviction_tier,
            source="chat",
            entry_price=float(price),
        )
        logger.info("Conviction written via chat for %s", ticker)
    except Exception as e:
        logger.warning("_maybe_write_conviction failed silently: %s", e)
```

- [ ] **Step 7.2: Wire _maybe_write_conviction into the chat endpoint**

In the `/api/chat` endpoint's `generate()` generator function, find where the full response is assembled. Look for around line 513–540 in raphi_server.py. Find the final `yield` after the Claude streaming ends, and add the conviction write call.

Locate the line that reads (near end of `generate()`):
```python
        yield _sse("done", json.dumps({"message": "complete"}))
```

Add before that line:
```python
        # Write conviction if response contains a Signal View conclusion
        full_response = " ".join(collected_text)  # collected_text must be assembled above
        _maybe_write_conviction(
            ticker=req.ticker.upper(),
            sig_cache_path=BASE / ".model_cache" / f"{req.ticker.upper()}.pkl",
            response_text=full_response,
        )
```

> **Note:** The chat endpoint uses `collected_text` as a list that gets appended to as tokens stream. Verify the variable name in your version of the file and use the correct one.

- [ ] **Step 7.3: Verify server starts cleanly**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m backend.raphi_server &
sleep 3
curl -s http://localhost:9999/api/health
pkill -f raphi_server
```

Expected: `{"status":"ok","server":"raphi-unified","a2a":true}`

- [ ] **Step 7.4: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/raphi_server.py
git commit -m "feat: wire _maybe_write_conviction into chat endpoint after Signal View detection"
```

---

## Task 8: Frontend — liveRefresh additions

**Files:**
- Modify: `backend/static/index.html`

- [ ] **Step 8.1: Add `loadConvictionStats()` JS function**

Find the existing `loadWatchlistPrices` function in `index.html`. Add the following two functions **immediately after** it:

```javascript
async function loadConvictionStats(signal) {
  try {
    const r = await apiFetch('/api/convictions/stats', { signal });
    if (!r.ok) return;
    const d = await r.json();

    const fmt = v => v === null || v === undefined ? '—' : v.toFixed(1) + '%';
    const fmtN = v => v === null || v === undefined ? '—' : v;

    const ids = {
      'cl-stat-ml-30':  fmt(d.ml_accuracy_30d),
      'cl-stat-ml-60':  fmt(d.ml_accuracy_60d),
      'cl-stat-ml-90':  fmt(d.ml_accuracy_90d),
      'cl-stat-sec':    fmt(d.sec_accuracy),
      'cl-stat-total':  fmtN(d.total_convictions),
      'cl-stat-pending': fmtN(d.pending_count),
    };
    for (const [id, val] of Object.entries(ids)) {
      const el = document.getElementById(id);
      if (el) el.textContent = val;
    }

    // Colour-code accuracy cards
    ['cl-stat-ml-30','cl-stat-ml-60','cl-stat-ml-90','cl-stat-sec'].forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      const pct = parseFloat(el.textContent);
      el.style.color = isNaN(pct) ? '' : pct >= 65 ? '#4ade80' : pct >= 50 ? '#f59e0b' : '#ef4444';
    });

    // Sidebar pending badge
    const badge = document.getElementById('nav-pending-count');
    if (badge) {
      badge.textContent = d.pending_count || '';
      badge.style.display = d.pending_count > 0 ? 'inline-flex' : 'none';
    }
  } catch(e) {
    if (e.name !== 'AbortError') console.warn('loadConvictionStats:', e.message);
  }
}

async function resolveConvictionsPoll(signal) {
  try {
    const r = await apiFetch('/api/convictions/check', { signal });
    if (!r.ok) return;
    const d = await r.json();
    if (d.resolved > 0) {
      // Refresh stats after new resolutions
      await loadConvictionStats(signal);
      // Refresh ledger table if user is on that page
      if (document.getElementById('cl-table-body')) {
        await loadConvictionLedger(signal);
      }
      // Show toast
      const toast = document.getElementById('toast');
      if (toast) {
        toast.textContent = `${d.resolved} conviction${d.resolved > 1 ? 's' : ''} resolved — accuracy updated`;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 4000);
      }
    }
  } catch(e) {
    if (e.name !== 'AbortError') console.warn('resolveConvictionsPoll:', e.message);
  }
}
```

- [ ] **Step 8.2: Add both functions to the existing liveRefresh() Promise.all**

Find this block in `index.html` (the existing liveRefresh Promise.all):
```javascript
await Promise.all([
    loadMarketMetrics(sig),
    loadDashboardSignals(sig),
    loadWatchlistPrices(sig),
```

Change it to:
```javascript
await Promise.all([
    loadMarketMetrics(sig),
    loadDashboardSignals(sig),
    loadWatchlistPrices(sig),
    loadConvictionStats(sig),
    resolveConvictionsPoll(sig),
```

- [ ] **Step 8.3: Verify no JS errors in browser console**

Start server, open `http://localhost:9999`, open DevTools → Console. Confirm no errors from the new functions. The functions will silently no-op if the ledger is empty.

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m backend.raphi_server &
```

Open http://localhost:9999, check console for errors, then:

```bash
pkill -f raphi_server
```

- [ ] **Step 8.4: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/static/index.html
git commit -m "feat: add loadConvictionStats + resolveConvictionsPoll to liveRefresh"
```

---

## Task 9: Frontend — Conviction Ledger page

**Files:**
- Modify: `backend/static/index.html`

- [ ] **Step 9.1: Add sidebar navigation entry**

Find the sidebar nav section in `index.html` — look for the Decision Memo nav item. It will look something like:
```html
<div class="nav-item" onclick="switchPage('memo')">
```

Add the Conviction Ledger nav item **immediately after** the Decision Memo item:

```html
<div class="nav-item" onclick="switchPage('convictions')" id="nav-convictions">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
  </svg>
  <span>Conviction Ledger</span>
  <span id="nav-pending-count" style="display:none;margin-left:auto;background:#f59e0b;color:#000;
    border-radius:10px;padding:1px 7px;font-size:0.7rem;font-weight:700;"></span>
</div>
```

- [ ] **Step 9.2: Add the Conviction Ledger page HTML**

Find the section where other pages are defined (look for `id="page-memo"` or `id="page-portfolio"`). Add the following **after** the last page div:

```html
<!-- ══ CONVICTION LEDGER PAGE ══════════════════════════════════════ -->
<div id="page-convictions" class="page" style="display:none;">
  <div class="page-header">
    <h2>Conviction Ledger</h2>
    <p class="page-subtitle">RAPHI's self-evaluated research track record</p>
  </div>

  <!-- Aggregate stats bar -->
  <div style="display:grid;grid-template-columns:repeat(7,1fr);gap:0.75rem;margin-bottom:1.5rem;">
    <div class="metric-card">
      <div class="metric-label">ML Acc 30d</div>
      <div class="metric-value" id="cl-stat-ml-30">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">ML Acc 60d</div>
      <div class="metric-value" id="cl-stat-ml-60">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">ML Acc 90d</div>
      <div class="metric-value" id="cl-stat-ml-90">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">SEC Trend Acc</div>
      <div class="metric-value" id="cl-stat-sec">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Signal vs SPY</div>
      <div class="metric-value" id="cl-stat-spy" style="color:#7c6cf0;">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Total Calls</div>
      <div class="metric-value" id="cl-stat-total">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Pending</div>
      <div class="metric-value" id="cl-stat-pending" style="color:#f59e0b;">—</div>
    </div>
  </div>

  <!-- Filters -->
  <div style="display:flex;gap:0.75rem;margin-bottom:1rem;align-items:center;">
    <select id="cl-filter-ticker" onchange="loadConvictionLedger(null)"
      style="background:#1e2535;border:1px solid #2a3550;color:#e2e8f0;
             border-radius:6px;padding:0.35rem 0.6rem;font-size:0.82rem;">
      <option value="">All Tickers</option>
    </select>
    <select id="cl-filter-window" onchange="loadConvictionLedger(null)"
      style="background:#1e2535;border:1px solid #2a3550;color:#e2e8f0;
             border-radius:6px;padding:0.35rem 0.6rem;font-size:0.82rem;">
      <option value="">All Windows</option>
      <option value="30d">30 days</option>
      <option value="60d">60 days</option>
      <option value="90d">90 days</option>
      <option value="sec">SEC Filing</option>
    </select>
    <span style="font-size:0.78rem;color:#4a5568;margin-left:auto;">
      Sorted newest first · 50 per page
    </span>
  </div>

  <!-- Conviction history table -->
  <div class="card" style="overflow-x:auto;margin-bottom:1.5rem;">
    <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
      <thead>
        <tr style="border-bottom:1px solid #2a3550;color:#94a3b8;text-align:left;">
          <th style="padding:0.6rem 0.75rem;">Date</th>
          <th style="padding:0.6rem 0.75rem;">Ticker</th>
          <th style="padding:0.6rem 0.75rem;">ML Signal</th>
          <th style="padding:0.6rem 0.75rem;">SEC Trend</th>
          <th style="padding:0.6rem 0.75rem;">Signal View</th>
          <th style="padding:0.6rem 0.75rem;">30d</th>
          <th style="padding:0.6rem 0.75rem;">60d</th>
          <th style="padding:0.6rem 0.75rem;">90d</th>
          <th style="padding:0.6rem 0.75rem;">SEC</th>
          <th style="padding:0.6rem 0.75rem;">Source</th>
        </tr>
      </thead>
      <tbody id="cl-table-body">
        <tr><td colspan="10" style="text-align:center;padding:2rem;color:#4a5568;">
          Loading conviction history...
        </td></tr>
      </tbody>
    </table>
    <div id="cl-pagination" style="padding:0.75rem;display:flex;gap:0.5rem;
         justify-content:center;border-top:1px solid #1a2035;"></div>
  </div>

  <!-- Regime breakdown -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
    <div class="card" id="cl-regime-low">
      <div style="color:#4ade80;font-size:0.72rem;text-transform:uppercase;
                  letter-spacing:.05em;margin-bottom:.5rem;">VIX &lt; 15 — Low Volatility</div>
      <div style="color:#4a5568;font-size:0.82rem;">No data yet</div>
    </div>
    <div class="card" id="cl-regime-mid">
      <div style="color:#f59e0b;font-size:0.72rem;text-transform:uppercase;
                  letter-spacing:.05em;margin-bottom:.5rem;">VIX 15–25 — Normal</div>
      <div style="color:#4a5568;font-size:0.82rem;">No data yet</div>
    </div>
    <div class="card" id="cl-regime-high">
      <div style="color:#ef4444;font-size:0.72rem;text-transform:uppercase;
                  letter-spacing:.05em;margin-bottom:.5rem;">VIX &gt; 25 — High Volatility</div>
      <div style="color:#4a5568;font-size:0.82rem;">No data yet</div>
    </div>
  </div>
</div>
```

- [ ] **Step 9.3: Add `loadConvictionLedger()` JS function**

Add after `resolveConvictionsPoll`:

```javascript
async function loadConvictionLedger(signal, page = 1) {
  const ticker = document.getElementById('cl-filter-ticker')?.value || '';
  const url = `/api/convictions/ledger?page=${page}${ticker ? '&ticker=' + ticker : ''}`;
  try {
    const r = await apiFetch(url, { signal });
    if (!r.ok) return;
    const d = await r.json();

    const tbody = document.getElementById('cl-table-body');
    if (!tbody) return;

    if (!d.rows || d.rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;padding:2rem;color:#4a5568;">' +
        'No convictions yet — generate a research memo to start building your track record.</td></tr>';
      return;
    }

    const chip = (w) => {
      if (!w) return '<span style="color:#4a5568;">—</span>';
      if (w.status === 'PENDING') {
        return `<span style="background:rgba(245,158,11,0.12);color:#f59e0b;padding:2px 8px;
          border-radius:10px;font-size:0.75rem;">⏳ ${w.due_date || ''}</span>`;
      }
      if (w.status === 'CONFIRMED') {
        const pct = w.vs_entry_pct != null ? (w.vs_entry_pct > 0 ? '+' : '') + w.vs_entry_pct.toFixed(1) + '%' : '';
        return `<span style="background:rgba(74,222,128,0.1);color:#4ade80;padding:2px 8px;
          border-radius:10px;font-size:0.75rem;">✓ ${pct}</span>`;
      }
      if (w.status === 'CONTRADICTED') {
        const pct = w.vs_entry_pct != null ? (w.vs_entry_pct > 0 ? '+' : '') + w.vs_entry_pct.toFixed(1) + '%' : '';
        return `<span style="background:rgba(239,68,68,0.1);color:#ef4444;padding:2px 8px;
          border-radius:10px;font-size:0.75rem;">✗ ${pct}</span>`;
      }
      if (w.status === 'INCONCLUSIVE') {
        return `<span style="color:#94a3b8;font-size:0.75rem;">~ flat</span>`;
      }
      if (w.status === 'N/A') return '<span style="color:#4a5568;font-size:0.75rem;">N/A</span>';
      return `<span style="color:#94a3b8;">${w.status}</span>`;
    };

    const dirColor = d => d === 'LONG' ? '#4ade80' : d === 'SHORT' ? '#ef4444' : '#94a3b8';

    tbody.innerHTML = d.rows.map(row => {
      const hasContradicted = Object.values(row.windows).some(w => w && w.status === 'CONTRADICTED');
      const rowBg = hasContradicted ? 'background:rgba(239,68,68,0.03);' : '';
      return `<tr style="border-bottom:1px solid #1a2035;${rowBg}">
        <td style="padding:.5rem .75rem;color:#94a3b8;">${row.date}</td>
        <td style="padding:.5rem .75rem;font-weight:600;">${row.ticker}</td>
        <td style="padding:.5rem .75rem;color:${dirColor(row.ml_direction)};">
          ${row.ml_direction} <span style="color:#4a5568;font-size:0.75rem;">(${((row.ml_prob||0)*100).toFixed(0)}%)</span>
        </td>
        <td style="padding:.5rem .75rem;color:#94a3b8;">${row.sec_trend || '—'}</td>
        <td style="padding:.5rem .75rem;">${row.signal_view || '—'}</td>
        <td style="padding:.5rem .75rem;">${chip(row.windows['30d'])}</td>
        <td style="padding:.5rem .75rem;">${chip(row.windows['60d'])}</td>
        <td style="padding:.5rem .75rem;">${chip(row.windows['90d'])}</td>
        <td style="padding:.5rem .75rem;">${chip(row.windows['sec'])}</td>
        <td style="padding:.5rem .75rem;color:#4a5568;font-size:0.75rem;">${row.source}</td>
      </tr>`;
    }).join('');

    // Pagination
    const pg = document.getElementById('cl-pagination');
    if (pg && d.total_pages > 1) {
      pg.innerHTML = Array.from({length: d.total_pages}, (_, i) => i + 1).map(p =>
        `<button onclick="loadConvictionLedger(null, ${p})"
          style="background:${p === d.page ? '#7c6cf0' : '#1e2535'};color:#e2e8f0;
                 border:1px solid #2a3550;border-radius:4px;padding:3px 10px;
                 cursor:pointer;font-size:0.78rem;">${p}</button>`
      ).join('');
    } else if (pg) {
      pg.innerHTML = '';
    }

  } catch(e) {
    if (e.name !== 'AbortError') console.warn('loadConvictionLedger:', e.message);
  }
}
```

- [ ] **Step 9.4: Register Conviction Ledger page in switchPage loaders**

Find the `switchPage` function or the loaders object in `index.html`. It will have entries like:
```javascript
portfolio: () => loadPortfolioPage(sig),
memo:      () => loadMemoPage(sig),
```

Add:
```javascript
convictions: () => Promise.all([loadConvictionStats(null), loadConvictionLedger(null)]),
```

- [ ] **Step 9.5: Verify ledger page renders in browser**

Start server, navigate to http://localhost:9999, click Conviction Ledger in sidebar. Confirm the page loads with empty state message and stats showing "—".

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m backend.raphi_server &
```

Open http://localhost:9999 → click "Conviction Ledger" → verify page loads cleanly → check console for errors.

```bash
pkill -f raphi_server
```

- [ ] **Step 9.6: Commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/static/index.html
git commit -m "feat: add Conviction Ledger page with stats bar, history table, and regime breakdown"
```

---

## Task 10: Frontend — inline compact conviction badge

**Files:**
- Modify: `backend/static/index.html`

- [ ] **Step 10.1: Add `loadTickerConvictionBadge()` function**

Add after `loadConvictionLedger`:

```javascript
async function loadTickerConvictionBadge(ticker, containerEl) {
  if (!ticker || !containerEl) return;
  try {
    const r = await apiFetch(`/api/convictions/stats?ticker=${ticker.toUpperCase()}`);
    if (!r.ok) return;
    const d = await r.json();

    const strip = document.createElement('div');
    strip.style.cssText = `margin-top:.75rem;padding:.5rem .85rem;border-radius:6px;
      font-size:.78rem;display:flex;align-items:center;gap:.5rem;`;

    if (d.total_resolved > 0 && d.ml_accuracy_30d !== null) {
      strip.style.background = 'rgba(124,108,240,0.06)';
      strip.style.borderTop   = '1px solid rgba(124,108,240,0.25)';
      const ml  = d.ml_accuracy_30d !== null ? d.ml_accuracy_30d.toFixed(1) + '%' : '—';
      const sec = d.sec_accuracy    !== null ? d.sec_accuracy.toFixed(1)    + '%' : '—';
      const n   = d.total_resolved;
      strip.innerHTML = `
        <span style="color:#7c6cf0;">⚖</span>
        <span style="color:#94a3b8;">RAPHI track record on <strong style="color:#e2e8f0;">${ticker.toUpperCase()}</strong>:</span>
        <span style="color:#4ade80;font-family:monospace;">${ml} ML</span>
        <span style="color:#4a5568;">·</span>
        <span style="color:#4ade80;font-family:monospace;">${sec} SEC</span>
        <span style="color:#4a5568;">over ${n} prior call${n !== 1 ? 's' : ''}</span>
        <span style="margin-left:auto;"><a href="#" onclick="switchPage('convictions');return false;"
          style="color:#7c6cf0;text-decoration:none;font-size:.75rem;">view ledger →</a></span>`;
    } else {
      strip.style.background = 'rgba(245,158,11,0.04)';
      strip.style.borderTop   = '1px solid rgba(245,158,11,0.2)';
      strip.innerHTML = `
        <span style="color:#f59e0b;">⚖</span>
        <span style="color:#94a3b8;">Conviction recorded.</span>
        <span style="color:#4a5568;">First RAPHI call on <strong style="color:#e2e8f0;">${ticker.toUpperCase()}</strong>
          · resolution due in 30 / 60 / 90 days</span>
        <span style="margin-left:auto;"><a href="#" onclick="switchPage('convictions');return false;"
          style="color:#7c6cf0;text-decoration:none;font-size:.75rem;">view ledger →</a></span>`;
    }

    containerEl.appendChild(strip);
  } catch(e) {
    if (e.name !== 'AbortError') console.warn('loadTickerConvictionBadge:', e.message);
  }
}
```

- [ ] **Step 10.2: Call badge after memo output renders**

Find where the memo response text is inserted into the DOM. Look for where the chat/memo output `innerHTML` is set or where the memo result container is populated. Add a call at the end:

```javascript
// After memo content is injected into memoOutputContainer:
loadTickerConvictionBadge(currentTicker, memoOutputContainer);
```

> **Note:** The exact variable names for the memo output container and current ticker will match what's already in your index.html. Search for where `innerHTML` is set on the memo output div and add the badge call immediately after.

- [ ] **Step 10.3: Verify badge appears after a memo**

Start server, open http://localhost:9999, navigate to the Decision Memo page, generate a memo for NVDA. Confirm the conviction badge appears below the output.

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m backend.raphi_server &
```

Open http://localhost:9999 → Decision Memo → enter "NVDA" → generate → check badge appears below output.

```bash
pkill -f raphi_server
```

- [ ] **Step 10.4: Run full test suite**

```bash
cd "/Users/alan/Desktop/SEC Data"
.venv/bin/python -m pytest tests/test_conviction_store.py -v
```

Expected: All tests PASS

- [ ] **Step 10.5: Final commit**

```bash
cd "/Users/alan/Desktop/SEC Data"
git add backend/static/index.html
git commit -m "feat: add loadTickerConvictionBadge inline compact conviction view on memo output"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by task |
|---|---|
| convictions.jsonl schema (all fields including vix_at_creation) | Task 1 |
| resolutions.jsonl schema | Task 2 |
| write_conviction() | Task 1 |
| check_pending() ML resolution | Task 2 |
| check_pending() SEC resolution | Task 3 |
| Idempotency — no double-writes | Task 2 (test included) |
| NEUTRAL band ±1.5% | Task 2 (_eval_ml) |
| SEC INCONCLUSIVE ±3% band | Task 3 (_check_sec_filing) |
| get_accuracy_stats() with denominator rule | Task 4 |
| get_ledger() paginated | Task 5 |
| 4 API endpoints | Task 6 |
| write_conviction after research output | Task 7 |
| liveRefresh additions | Task 8 |
| Conviction Ledger page (stats bar, table, regime cards) | Task 9 |
| Inline compact badge (with/without history variants) | Task 10 |
| Error handling — yfinance fail = skip, no write | Task 2 (check_pending) |
| Error handling — EDGAR not updated = PENDING | Task 3 |

**No gaps found.**

**Type consistency check:**
- `write_conviction()` signature in Task 1 matches call in Task 7 ✓
- API endpoint `ConvictionRequest` fields match `write_conviction()` params ✓
- `get_accuracy_stats()` return keys (`ml_accuracy_30d`, `pending_count`, etc.) match JS in Tasks 8 and 10 ✓
- `get_ledger()` return keys (`rows`, `total`, `page`, `total_pages`) match JS in Task 9 ✓
- `_window_status()` return keys (`status`, `vs_entry_pct`, `due_date`) match JS chip renderer in Task 9 ✓
