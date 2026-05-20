"""
web_citations.py — Perplexity-style web citation search.

Primary provider:
  - Google Programmable Search JSON API
    https://developers.google.com/custom-search/v1

Fallback provider:
  - Firecrawl search, when Google credentials are absent.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from urllib.parse import urlparse

import httpx

try:
    import firecrawl_client
except ImportError:  # pragma: no cover
    from backend import firecrawl_client

_GOOGLE_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
_GOOGLE_CX = os.environ.get("GOOGLE_SEARCH_CX", "").strip()
_GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
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


def google_configured() -> bool:
    return bool(_GOOGLE_API_KEY and _GOOGLE_CX)


def provider_status() -> dict:
    return {
        "google_configured": google_configured(),
        "firecrawl_configured": firecrawl_client.is_available(),
        "primary_provider": "google_custom_search" if google_configured() else "firecrawl_search",
        "required_google_env": ["GOOGLE_SEARCH_API_KEY", "GOOGLE_SEARCH_CX"],
    }


def _normalize_google_item(item: dict, idx: int) -> dict:
    pagemap = item.get("pagemap") or {}
    metatags = (pagemap.get("metatags") or [{}])[0] if isinstance(pagemap.get("metatags"), list) else {}
    url = item.get("link", "")
    return {
        "id": idx,
        "title": item.get("title", ""),
        "url": url,
        "domain": _domain(url),
        "snippet": item.get("snippet", ""),
        "display_link": item.get("displayLink", ""),
        "published": (
            metatags.get("article:published_time")
            or metatags.get("og:updated_time")
            or metatags.get("date")
            or ""
        ),
        "provider": "Google Programmable Search",
        "retrieved_at": _now_iso(),
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
    prefer_google: bool = True,
) -> dict:
    clean_query = " ".join(str(query or "").split())[:500]
    ticker = str(ticker or "").strip().upper()
    limit = min(max(int(limit or 5), 1), 10)
    if ticker and ticker not in clean_query.upper():
        clean_query = f"{ticker} {clean_query}"
    if not clean_query:
        return {"provider": "none", "query": clean_query, "results": [], "count": 0, "error": "query is required"}

    cache_key = f"{clean_query}:{ticker}:{limit}:{prefer_google}:{google_configured()}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    if prefer_google and google_configured():
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(
                    _GOOGLE_ENDPOINT,
                    params={
                        "key": _GOOGLE_API_KEY,
                        "cx": _GOOGLE_CX,
                        "q": clean_query,
                        "num": limit,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            results = [
                _normalize_google_item(item, idx)
                for idx, item in enumerate(data.get("items") or [], start=1)
            ]
            return _store(cache_key, {
                "provider": "google_custom_search",
                "query": clean_query,
                "results": results,
                "count": len(results),
                "retrieved_at": _now_iso(),
                "source_note": "Google Programmable Search JSON API",
            })
        except Exception as exc:
            if not firecrawl_client.is_available():
                return _store(cache_key, {
                    "provider": "google_custom_search",
                    "query": clean_query,
                    "results": [],
                    "count": 0,
                    "error": str(exc),
                    "retrieved_at": _now_iso(),
                })

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
        "error": "No web citation provider configured. Set GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_CX or FIRECRAWL_API_KEY.",
        "retrieved_at": _now_iso(),
    })
