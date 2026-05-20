import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import web_citations


def test_google_search_results_are_normalized(monkeypatch):
    monkeypatch.setattr(web_citations, "_GOOGLE_API_KEY", "key")
    monkeypatch.setattr(web_citations, "_GOOGLE_CX", "cx")
    monkeypatch.setattr(web_citations, "_cache", {})

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "items": [{
                    "title": "ASST result",
                    "link": "https://example.com/asst",
                    "displayLink": "example.com",
                    "snippet": "ASST source snippet",
                    "pagemap": {"metatags": [{"article:published_time": "2026-05-20T00:00:00Z"}]},
                }]
            }

    class FakeClient:
        def __init__(self, timeout=30.0):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params):
            assert "customsearch" in url
            assert params["q"] == "ASST Strive news"
            return FakeResponse()

    monkeypatch.setattr(web_citations.httpx, "Client", FakeClient)

    result = web_citations.search_citations("Strive news", ticker="ASST", limit=1)

    assert result["provider"] == "google_custom_search"
    assert result["count"] == 1
    assert result["results"][0]["url"] == "https://example.com/asst"
    assert result["results"][0]["provider"] == "Google Programmable Search"


def test_no_provider_returns_actionable_error(monkeypatch):
    monkeypatch.setattr(web_citations, "_GOOGLE_API_KEY", "")
    monkeypatch.setattr(web_citations, "_GOOGLE_CX", "")
    monkeypatch.setattr(web_citations, "_cache", {})
    monkeypatch.setattr(web_citations.firecrawl_client, "is_available", lambda: False)

    result = web_citations.search_citations("ASST news", ticker="ASST", limit=1)

    assert result["provider"] == "none"
    assert result["count"] == 0
    assert "GOOGLE_SEARCH_API_KEY" in result["error"]
