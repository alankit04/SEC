"""web_citations.py — Perplexity-style web citation search via Firecrawl."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from urllib.parse import urlparse

try:
    import firecrawl_client
except ImportError:  # pragma: no cover
    from backend import firecrawl_client

_SEARCH_TTL = 900
_cache: dict[str, tuple[float, dict]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _cached(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _SEARCH_TTL:
        return entry[1]
    return None


def _store(key: str, value: dict) -> dict:
    _cache[key] = (time.time(), value)
    return value


def provider_status() -> dict:
    return {
        "firecrawl_configured": firecrawl_client.is_available(),
        "primary_provider": "firecrawl_search",
        "required_env": ["FIRECRAWL_API_KEY"],
    }


def _normalize_firecrawl_item(item: dict, idx: int) -> dict:
    url = item.get("url", "")
    markdown = item.get("markdown", "") or ""
    snippet = item.get("description") or markdown.replace("\n", " ")[:280]
    return {
        "id": idx,
        "title": item.get("title", ""),
        "url": url,
        "domain": _domain(url),
        "snippet": snippet,
        "display_link": _domain(url),
        "published": "",
        "provider": "Firecrawl Search",
        "retrieved_at": _now_iso(),
    }


def search_citations(
    query: str,
    *,
    ticker: str = "",
    limit: int = 5,
) -> dict:
    clean_query = " ".join(str(query or "").split())[:500]
    ticker = str(ticker or "").strip().upper()
    limit = min(max(int(limit or 5), 1), 10)
    if ticker and ticker not in clean_query.upper():
        clean_query = f"{ticker} {clean_query}"
    if not clean_query:
        return {"provider": "none", "query": clean_query, "results": [], "count": 0, "error": "query is required"}

    cache_key = f"{clean_query}:{ticker}:{limit}:firecrawl"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    if firecrawl_client.is_available():
        results_raw = firecrawl_client.search_web(clean_query, limit=limit, scrape_results=True, max_chars_per_result=1600)
        results = [
            _normalize_firecrawl_item(item, idx)
            for idx, item in enumerate(results_raw, start=1)
            if item.get("success") and item.get("url")
        ]
        errors = [item.get("error") for item in results_raw if not item.get("success") and item.get("error")]
        return _store(cache_key, {
            "provider": "firecrawl_search",
            "query": clean_query,
            "results": results,
            "count": len(results),
            "error": errors[0] if errors and not results else None,
            "retrieved_at": _now_iso(),
            "source_note": "Firecrawl search fallback",
        })

    return _store(cache_key, {
        "provider": "none",
        "query": clean_query,
        "results": [],
        "count": 0,
        "error": "No web citation provider configured. Set FIRECRAWL_API_KEY.",
        "retrieved_at": _now_iso(),
    })
