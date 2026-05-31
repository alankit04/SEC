"""web_citations.py — local-first citation search with optional Firecrawl refresh."""

from __future__ import annotations

from datetime import datetime, timezone
import time

try:
    from citation_index import CitationIndex, get_citation_index
except ImportError:  # pragma: no cover
    from backend.citation_index import CitationIndex, get_citation_index

_SEARCH_TTL = 120   # 2 minutes — matches edgar_live search TTL for consistent freshness
_cache: dict[str, tuple[float, dict]] = {}
_index = get_citation_index()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cached(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _SEARCH_TTL:
        return entry[1]
    return None


def _store(key: str, value: dict) -> dict:
    _cache[key] = (time.time(), value)
    return value


def provider_status() -> dict:
    status = _index.status()
    return {
        **status,
        "primary_provider": "local_citation_index",
        "refresh_provider": "firecrawl_search",
        "required_refresh_env": ["FIRECRAWL_API_KEY"],
    }


def search_citations(
    query: str,
    *,
    user_scope: str = "global",
    ticker: str = "",
    limit: int = 5,
    refresh_if_missing: bool = False,
    index: CitationIndex | None = None,
) -> dict:
    clean_query = " ".join(str(query or "").split())[:500]
    ticker = str(ticker or "").strip().upper()
    limit = min(max(int(limit or 5), 1), 10)
    if ticker and ticker not in clean_query.upper():
        clean_query = f"{ticker} {clean_query}"
    if not clean_query:
        return {"provider": "none", "query": clean_query, "results": [], "count": 0, "error": "query is required"}

    scope = str(user_scope or "global").strip()[:128] or "global"
    cache_key = f"{scope}:{clean_query}:{ticker}:{limit}:{refresh_if_missing}:local_index"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    citation_index = index or _index
    result = citation_index.search_with_refresh(
        clean_query,
        user_scope=scope,
        ticker=ticker,
        limit=limit,
        refresh_if_missing=refresh_if_missing,
        min_results=2,
    )
    result.setdefault("source_note", "RAPHI local citation index")
    result.setdefault("retrieved_at", _now_iso())
    if not result.get("results"):
        result["error"] = (
            result.get("refresh", {}).get("error")
            or "No local citation results found. Run citation refresh/indexing for this source."
        )
    return _store(cache_key, {
        **result,
        "retrieved_at": _now_iso(),
    })
