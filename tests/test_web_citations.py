import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import web_citations


def test_firecrawl_search_results_are_normalized(monkeypatch):
    monkeypatch.setattr(web_citations, "_cache", {})
    monkeypatch.setattr(web_citations.firecrawl_client, "is_available", lambda: True)

    def fake_search(query, limit=5, scrape_results=True, max_chars_per_result=1600):
        assert query == "ASST Strive news"
        return [{
            "success": True,
            "title": "ASST result",
            "url": "https://example.com/asst",
            "description": "ASST source snippet",
            "markdown": "# ASST source snippet",
        }]

    monkeypatch.setattr(web_citations.firecrawl_client, "search_web", fake_search)

    result = web_citations.search_citations("Strive news", ticker="ASST", limit=1)

    assert result["provider"] == "firecrawl_search"
    assert result["count"] == 1
    assert result["results"][0]["url"] == "https://example.com/asst"
    assert result["results"][0]["provider"] == "Firecrawl Search"


def test_no_provider_returns_actionable_error(monkeypatch):
    monkeypatch.setattr(web_citations, "_cache", {})
    monkeypatch.setattr(web_citations.firecrawl_client, "is_available", lambda: False)

    result = web_citations.search_citations("ASST news", ticker="ASST", limit=1)

    assert result["provider"] == "none"
    assert result["count"] == 0
    assert "FIRECRAWL_API_KEY" in result["error"]
