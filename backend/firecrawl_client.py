"""
firecrawl_client.py — Firecrawl API client for scraping web sources.

Used to pull narrative content not available in structured SEC XBRL data:
  - Earnings call transcripts (Seeking Alpha, The Motley Fool)
  - Analyst commentary and price target updates
  - Company IR pages and press releases
  - News articles by URL

Requires FIRECRAWL_API_KEY in environment. Degrades gracefully if key missing.

API docs: https://docs.firecrawl.dev/api-reference/endpoint/scrape
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import httpx

logger = logging.getLogger("raphi.firecrawl")

_API_KEY      = os.environ.get("FIRECRAWL_API_KEY", "").strip()
_BASE_URL     = "https://api.firecrawl.dev/v1"
_SCRAPE_TTL   = 1800   # 30 minutes — web content changes slowly for our purposes
_SEARCH_TTL   = 900    # 15 minutes

_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: int):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def _store(key: str, value) -> None:
    _cache[key] = (time.time(), value)


def is_available() -> bool:
    """Return True if a Firecrawl API key is configured."""
    return bool(_API_KEY)


def scrape_url(
    url: str,
    *,
    max_chars: int = 6000,
    only_main_content: bool = True,
) -> dict:
    """
    Scrape a single URL and return clean markdown content.

    Args:
        url: URL to scrape
        max_chars: Truncate output markdown to this length
        only_main_content: Strip nav/headers/footers (recommended True)

    Returns:
        dict with keys: url, title, markdown, success, error
    """
    if not _API_KEY:
        return {"url": url, "success": False, "error": "FIRECRAWL_API_KEY not configured", "markdown": ""}

    cache_key = f"scrape:{url}:{max_chars}"
    cached = _cached(cache_key, _SCRAPE_TTL)
    if cached is not None:
        return cached  # type: ignore[return-value]

    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": only_main_content,
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                f"{_BASE_URL}/scrape",
                json=payload,
                headers={"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("data", {})
        markdown = content.get("markdown", "") or ""
        if len(markdown) > max_chars:
            cutoff = markdown.rfind("\n", 0, max_chars)
            markdown = markdown[:cutoff] if cutoff > 0 else markdown[:max_chars]

        result = {
            "url":      url,
            "title":    content.get("metadata", {}).get("title", ""),
            "markdown": markdown,
            "success":  True,
            "error":    None,
        }
        _store(cache_key, result)
        return result

    except httpx.HTTPStatusError as exc:
        err = f"HTTP {exc.response.status_code}"
        logger.warning("Firecrawl scrape failed for %s: %s", url, err)
        return {"url": url, "success": False, "error": err, "markdown": ""}
    except Exception as exc:
        logger.warning("Firecrawl scrape exception for %s: %s", url, exc)
        return {"url": url, "success": False, "error": str(exc), "markdown": ""}


def search_web(
    query: str,
    *,
    limit: int = 5,
    scrape_results: bool = True,
    max_chars_per_result: int = 3000,
) -> list[dict]:
    """
    Search the web via Firecrawl's /search endpoint and optionally scrape top results.

    Args:
        query: Search query
        limit: Number of results (max 10 recommended)
        scrape_results: If True, include scraped markdown from each result URL
        max_chars_per_result: Truncate each result's content

    Returns:
        List of dicts: url, title, description, markdown (if scraped), success
    """
    if not _API_KEY:
        return [{"success": False, "error": "FIRECRAWL_API_KEY not configured"}]

    cache_key = f"search:{query}:{limit}:{scrape_results}"
    cached = _cached(cache_key, _SEARCH_TTL)
    if cached is not None:
        return cached  # type: ignore[return-value]

    payload = {
        "query": query,
        "limit": min(limit, 10),
        "scrapeOptions": {
            "formats": ["markdown"],
            "onlyMainContent": True,
        } if scrape_results else {},
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{_BASE_URL}/search",
                json=payload,
                headers={"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in (data.get("data") or []):
            markdown = item.get("markdown", "") or ""
            if len(markdown) > max_chars_per_result:
                cutoff = markdown.rfind("\n", 0, max_chars_per_result)
                markdown = markdown[:cutoff] if cutoff > 0 else markdown[:max_chars_per_result]
            results.append({
                "url":         item.get("url", ""),
                "title":       item.get("title", ""),
                "description": item.get("description", ""),
                "markdown":    markdown,
                "success":     True,
            })

        _store(cache_key, results)
        return results

    except httpx.HTTPStatusError as exc:
        err = f"HTTP {exc.response.status_code}"
        logger.warning("Firecrawl search failed for '%s': %s", query, err)
        return [{"success": False, "error": err}]
    except Exception as exc:
        logger.warning("Firecrawl search exception for '%s': %s", query, exc)
        return [{"success": False, "error": str(exc)}]


def get_earnings_transcript(ticker: str, *, max_chars: int = 6000) -> dict:
    """
    Search for the most recent earnings call transcript for a ticker
    and return its content as clean markdown.

    Sources tried: Seeking Alpha, The Motley Fool, Fool.com
    Falls back gracefully if nothing found or API key missing.
    """
    if not _API_KEY:
        return {"ticker": ticker, "success": False, "error": "FIRECRAWL_API_KEY not configured", "markdown": ""}

    query = f"{ticker} earnings call transcript Q1 2025 OR Q4 2024 OR Q2 2025"
    results = search_web(query, limit=3, scrape_results=True, max_chars_per_result=max_chars)

    for result in results:
        if result.get("success") and result.get("markdown"):
            return {
                "ticker":   ticker.upper(),
                "source":   result.get("url", ""),
                "title":    result.get("title", ""),
                "markdown": result["markdown"],
                "success":  True,
            }

    return {
        "ticker":  ticker.upper(),
        "success": False,
        "error":   "No transcript found in search results",
        "markdown": "",
    }


def get_analyst_coverage(ticker: str, *, max_chars: int = 4000) -> dict:
    """
    Search for recent analyst price target changes or coverage initiations for a ticker.
    Returns scraped content with ratings, targets, and rationale.
    """
    if not _API_KEY:
        return {"ticker": ticker, "success": False, "error": "FIRECRAWL_API_KEY not configured", "markdown": ""}

    query = f"{ticker} analyst price target rating 2025 upgrade downgrade initiation"
    results = search_web(query, limit=3, scrape_results=True, max_chars_per_result=max_chars)

    for result in results:
        if result.get("success") and result.get("markdown"):
            return {
                "ticker":   ticker.upper(),
                "source":   result.get("url", ""),
                "title":    result.get("title", ""),
                "markdown": result["markdown"],
                "success":  True,
            }

    return {
        "ticker":  ticker.upper(),
        "success": False,
        "error":   "No analyst coverage found",
        "markdown": "",
    }
