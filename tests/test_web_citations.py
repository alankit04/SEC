import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import web_citations
from citation_index import CitationDocument, CitationIndex


def test_web_citations_searches_local_index_first(tmp_path, monkeypatch):
    idx = CitationIndex(database_url="", sqlite_path=tmp_path / "citations.sqlite")
    idx.add_document(CitationDocument(
        ticker="ASST",
        source_type="news",
        title="ASST result",
        url="https://example.com/asst",
        text="ASST Strive news source snippet about the company.",
    ))
    monkeypatch.setattr(web_citations, "_cache", {})

    result = web_citations.search_citations("Strive news", ticker="ASST", limit=1, index=idx)

    assert result["provider"] == "local_citation_index"
    assert result["count"] == 1
    assert result["results"][0]["url"] == "https://example.com/asst"
    assert result["results"][0]["provider"] == "RAPHI Citation Index"


def test_web_citations_reports_missing_local_results_without_refresh(tmp_path, monkeypatch):
    idx = CitationIndex(database_url="", sqlite_path=tmp_path / "citations.sqlite")
    monkeypatch.setattr(web_citations, "_cache", {})

    result = web_citations.search_citations("ASST news", ticker="ASST", limit=1, index=idx)

    assert result["provider"] == "local_citation_index"
    assert result["count"] == 0
    assert "No local citation results found" in result["error"]
