import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from citation_index import CitationDocument, CitationIndex


def test_citation_index_stores_and_searches_local_documents(tmp_path):
    idx = CitationIndex(database_url="", sqlite_path=tmp_path / "citations.sqlite")

    added = idx.add_document(CitationDocument(
        ticker="ASST",
        source_type="press_release",
        title="Strive Bitcoin Treasury Update",
        url="https://example.com/asst-strive",
        text="Strive announced a Bitcoin treasury strategy and merger update for ASST shareholders.",
    ))
    result = idx.search("Bitcoin treasury merger", ticker="ASST", limit=3)

    assert added["inserted_chunks"] >= 1
    assert result["provider"] == "local_citation_index"
    assert result["backend"] == "sqlite"
    assert result["count"] >= 1
    assert result["results"][0]["url"] == "https://example.com/asst-strive"
    assert "Bitcoin treasury" in result["results"][0]["snippet"]


def test_citation_index_refreshes_from_firecrawl_only_when_missing(tmp_path, monkeypatch):
    idx = CitationIndex(database_url="", sqlite_path=tmp_path / "citations.sqlite")
    calls = {"search": 0}

    monkeypatch.setattr("citation_index.firecrawl_client.is_available", lambda: True)

    def fake_search(query, limit=5, scrape_results=True, max_chars_per_result=5000):
        calls["search"] += 1
        return [{
            "success": True,
            "title": "ASST source",
            "url": "https://example.com/source",
            "description": "Source description",
            "markdown": "ASST Strive source evidence about merger and Bitcoin treasury strategy.",
        }]

    monkeypatch.setattr("citation_index.firecrawl_client.search_web", fake_search)

    first = idx.search_with_refresh("Strive Bitcoin treasury", ticker="ASST", refresh_if_missing=True, min_results=1)
    second = idx.search_with_refresh("Strive Bitcoin treasury", ticker="ASST", refresh_if_missing=True, min_results=1)

    assert calls["search"] == 1
    assert first["refresh"]["attempted"] is True
    assert second["refresh"]["attempted"] is False
    assert second["count"] >= 1


class FakeSEC:
    def ticker_filings(self, ticker, limit=8):
        return [{
            "accession": "0001234567-26-000001",
            "form": "8-K",
            "filed": "2026-05-01",
            "period": "2026-04-30",
            "quarter": "2026q2",
            "sec_url": "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/",
            "citation": {
                "accession": "0001234567-26-000001",
                "form": "8-K",
                "filed": "2026-05-01",
                "sec_url": "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000001/",
            },
        }]

    def company_financial_entries(self, ticker, limit_filings=8):
        return [{
            "accession": "0001234567-26-000002",
            "form": "10-Q",
            "filed": "2026-05-10",
            "period": "2026-03-31",
            "tag": "Revenues",
            "metric": "revenue",
            "val": 1250000,
            "uom": "USD",
            "sec_url": "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000002/",
            "citation": {
                "accession": "0001234567-26-000002",
                "form": "10-Q",
                "filed": "2026-05-10",
                "sec_url": "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000002/",
            },
        }]


def test_citation_index_ingests_sec_filings_and_xbrl(tmp_path):
    idx = CitationIndex(database_url="", sqlite_path=tmp_path / "citations.sqlite")

    ingested = idx.ingest_sec_ticker(FakeSEC(), "ASST")
    filing = idx.search("8-K accession 0001234567-26-000001", ticker="ASST")
    financial = idx.search("revenue Revenues 1250000", ticker="ASST")

    assert ingested["indexed_documents"] == 2
    assert filing["count"] >= 1
    assert financial["count"] >= 1
    assert filing["results"][0]["source_type"] == "sec_filing"
