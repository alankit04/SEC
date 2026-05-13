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


# ── ML resolution tests ────────────────────────────────────────────────────────

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


# ── SEC resolution tests ───────────────────────────────────────────────────────

def test_check_pending_sec_confirmed(ledger_dir):
    """SEC trend 'accelerating' + next filing shows +5% revenue = CONFIRMED."""
    import conviction_store as cs
    make_conviction(ledger_dir, trend="accelerating", days_ago=35)

    with patch("conviction_store.yf.Ticker") as mock_ticker, \
         patch("conviction_store._sec_latest_revenue_after", return_value=(1_050_000, "2026-03-31")):
        mock_ticker.return_value.history.return_value = MagicMock(empty=True)
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


# ── Accuracy stats tests ───────────────────────────────────────────────────────

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
    make_conviction(ledger_dir, ticker="AAPL", direction="LONG", days_ago=5)

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
