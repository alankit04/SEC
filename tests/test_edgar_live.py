"""
Tests for backend/edgar_live.py — live EDGAR API client.

All network calls are patched so tests run offline and deterministically.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_submissions_payload(ticker="AAPL", cik="0000320193"):
    """Minimal EDGAR submissions JSON for one 10-Q and one 8-K."""
    return {
        "cik": cik,
        "tickers": [ticker],
        "filings": {
            "recent": {
                "form":            ["10-Q", "8-K"],
                "filingDate":      ["2026-04-15", "2026-03-01"],
                "accessionNumber": ["0000320193-26-000050", "0000320193-26-000030"],
                "primaryDocument": ["aapl-20260331.htm", "0000320193-26-000030-index.htm"],
                "reportDate":      ["2026-03-31", ""],
            }
        },
    }


def _patch_get(payload):
    """Patch _get() in edgar_live to return payload without hitting the network."""
    return patch("backend.edgar_live._get", return_value=payload)


def _patch_cik(cik="0000320193"):
    return patch("backend.edgar_live._cik_from_ticker", return_value=cik)


# ── tests ────────────────────────────────────────────────────────────────────

def test_get_recent_filings_returns_list():
    from backend.edgar_live import get_recent_filings, _cache
    _cache.clear()
    with _patch_cik(), _patch_get(_make_submissions_payload()):
        results = get_recent_filings("AAPL", days=365, limit=10)
    assert isinstance(results, list)
    assert len(results) >= 1
    first = results[0]
    assert first["ticker"] == "AAPL"
    assert first["form"] in {"10-Q", "8-K"}
    assert first["accession"]
    assert first["documents_url"].startswith("https://www.sec.gov/Archives/edgar/data/")
    assert first["sec_url"].startswith("https://www.sec.gov/")


def test_get_recent_filings_respects_form_filter():
    from backend.edgar_live import get_recent_filings, _cache
    _cache.clear()
    with _patch_cik(), _patch_get(_make_submissions_payload()):
        results = get_recent_filings("AAPL", forms=["10-Q"], days=365, limit=10)
    assert all(r["form"] == "10-Q" for r in results)


def test_get_recent_filings_returns_empty_when_cik_missing():
    from backend.edgar_live import get_recent_filings, _cache
    _cache.clear()
    with patch("backend.edgar_live._cik_from_ticker", return_value=None):
        results = get_recent_filings("ZZZZ")
    assert results == []


def test_get_recent_filings_returns_empty_when_api_fails():
    from backend.edgar_live import get_recent_filings, _cache
    _cache.clear()
    with _patch_cik(), _patch_get(None):
        results = get_recent_filings("AAPL", days=365)
    assert results == []


def test_get_recent_filings_uses_cache_on_second_call():
    from backend.edgar_live import get_recent_filings, _cache
    _cache.clear()
    payload = _make_submissions_payload()
    with _patch_cik(), _patch_get(payload) as mock_get:
        get_recent_filings("AAPL", days=365, limit=5)
        get_recent_filings("AAPL", days=365, limit=5)
    # _get should only be called once; second call hits in-process cache
    assert mock_get.call_count == 1


def test_get_recent_8k_filters_to_8k():
    from backend.edgar_live import get_recent_8k, _cache
    _cache.clear()
    with _patch_cik(), _patch_get(_make_submissions_payload()):
        results = get_recent_8k("AAPL", days=365)
    assert all(r["form"] == "8-K" for r in results)


def test_get_ticker_live_summary_structure():
    from backend.edgar_live import get_ticker_live_summary, _cache
    _cache.clear()
    with _patch_cik(), _patch_get(_make_submissions_payload()):
        summary = get_ticker_live_summary("AAPL", days=365)
    assert summary["ticker"] == "AAPL"
    assert "filings" in summary
    assert "material_events" in summary
    assert "insider_transactions" in summary
    assert "retrieved_at" in summary
    assert isinstance(summary["filings"], list)
    assert isinstance(summary["material_events"], list)
    assert isinstance(summary["insider_transactions"], list)


def test_get_ticker_live_summary_unknown_ticker():
    from backend.edgar_live import get_ticker_live_summary, _cache
    _cache.clear()
    with patch("backend.edgar_live._cik_from_ticker", return_value=None), \
         _patch_get(None):
        summary = get_ticker_live_summary("ZZZZ")
    assert summary["ticker"] == "ZZZZ"
    assert summary["filings"] == []
    assert summary["material_events"] == []
