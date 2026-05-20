import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sec_data import QUARTERS, SECData, filing_citation, sec_accession_url


def test_sec_data_uses_data_directory_and_all_quarters():
    root = Path(__file__).parent.parent
    sec = SECData(root)

    assert sec.base == root / "data"
    assert "2024q4" in QUARTERS
    assert len(QUARTERS) == 16


def test_company_tickers_json_drives_cik_lookup():
    root = Path(__file__).parent.parent
    sec = SECData(root)

    assert sec.cik_for_ticker("NVDA") == "1045810"
    results = sec.search_companies("NVIDIA", limit=1)
    assert results[0]["ticker"] == "NVDA"


def test_sec_universe_can_screen_beyond_watchlist():
    root = Path(__file__).parent.parent
    sec = SECData(root)

    universe = sec.company_universe(sic="36", tickered_only=True, limit=10)

    assert universe["count"] > 0
    assert universe["companies"][0]["ticker"]
    assert all(row["sic"].startswith("36") for row in universe["companies"])
    assert any(row["ticker"] not in {"NVDA", "AAPL", "MSFT", "META", "TSLA", "AMZN", "GOOGL"} for row in universe["companies"])


def test_sec_industry_summary_reports_local_universe_buckets():
    root = Path(__file__).parent.parent
    sec = SECData(root)

    summary = sec.industry_summary()

    assert summary["total_industries"] > 0
    assert any(row["industry"] == "Electronic Equipment" for row in summary["industries"])


def test_sec_filing_citations_include_archives_url_and_normalized_dates():
    citation = filing_citation(
        cik="1045810",
        adsh="0001045810-25-000230",
        form="10-Q",
        filed="20251119",
        period="20251031.0",
        quarter="2025q4",
    )

    assert citation["source"] == "SEC Financial Statement Data Sets"
    assert citation["filed"] == "2025-11-19"
    assert citation["period"] == "2025-10-31"
    assert citation["sec_url"] == sec_accession_url("1045810", "0001045810-25-000230")
    assert citation["sec_url"].endswith("/1045810/000104581025000230/")
