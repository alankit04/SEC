"""
edgar_live.py — Real-time SEC EDGAR data via public APIs (no API key required).

Endpoints used:
  - data.sec.gov/submissions/CIK{cik}.json  → all filings for a company
  - efts.sec.gov/LATEST/search-index        → full-text search (8-K, Form 4, etc.)
  - www.sec.gov/Archives/edgar/data/...     → actual filing document text

All requests are rate-limited to ≤10 req/s per SEC EDGAR guidelines.
"""

from __future__ import annotations

import re
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from functools import lru_cache

import httpx

logger = logging.getLogger("raphi.edgar_live")

# SEC guidelines: max 10 requests/second, User-Agent required
_HEADERS = {
    "User-Agent": "RAPHI-Investment-Platform contact@raphi.ai",
    "Accept-Encoding": "gzip, deflate",
}
_RATE_LIMIT_SLEEP = 0.12   # 120 ms between requests → ~8 req/s (under 10 limit)

# Cache TTLs (seconds)
_SUBMISSION_TTL   = 300    # 5 minutes — always fetch near-fresh filings
_FILING_TEXT_TTL  = 600    # 10 minutes — filing prose is stable but respect server load
_SEARCH_TTL       = 120    # 2 minutes

_MAX_RETRIES = 3
_RETRY_BASE  = 1.0   # seconds; doubled each retry on 429

_cache: dict[str, tuple[float, object]] = {}
_last_request_ts = 0.0
_rate_lock = __import__("threading").Lock()


def _rate_limit() -> None:
    global _last_request_ts
    with _rate_lock:
        elapsed = time.monotonic() - _last_request_ts
        if elapsed < _RATE_LIMIT_SLEEP:
            time.sleep(_RATE_LIMIT_SLEEP - elapsed)
        _last_request_ts = time.monotonic()


def _cached(key: str, ttl: int):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def _store(key: str, value) -> None:
    _cache[key] = (time.time(), value)


def _get(url: str, params: dict | None = None, timeout: float = 30.0) -> dict | list | None:
    """GET with rate limiting, 429 back-off retry, and error handling. Returns parsed JSON or None."""
    for attempt in range(_MAX_RETRIES):
        _rate_limit()
        try:
            with httpx.Client(headers=_HEADERS, timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", _RETRY_BASE * (2 ** attempt)))
                    logger.warning("EDGAR 429 — waiting %.1fs (attempt %d/%d): %s", wait, attempt + 1, _MAX_RETRIES, url)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("EDGAR HTTP %s for %s", exc.response.status_code, url)
            return None
        except Exception as exc:
            logger.warning("EDGAR request failed: %s — %s", url, exc)
            return None
    logger.warning("EDGAR gave up after %d retries (429): %s", _MAX_RETRIES, url)
    return None


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------

def _cik_from_ticker(ticker: str) -> str | None:
    """
    Try to resolve a CIK from ticker via local company_tickers.json or EDGAR search.
    Returns zero-padded 10-digit CIK string or None.
    """
    from pathlib import Path
    try:
        from paths import COMPANY_TICKERS_FILE
    except ImportError:
        from backend.paths import COMPANY_TICKERS_FILE

    # 1. Local company_tickers.json (fast path)
    ticker_upper = ticker.strip().upper()
    try:
        with open(COMPANY_TICKERS_FILE, "r") as f:
            data = json.load(f)
        for entry in data.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
    except Exception:
        pass

    # 2. EDGAR full-text company search (fallback)
    result = _get(
        "https://efts.sec.gov/LATEST/search-index",
        params={"q": f'"{ticker_upper}"', "dateRange": "custom",
                "startdt": "2020-01-01", "forms": "10-K", "hits.hits._source": "entity_name,file_num"},
    )
    if result:
        hits = (result.get("hits") or {}).get("hits") or []
        if hits:
            src = hits[0].get("_source") or {}
            entity_id = hits[0].get("_id", "")
            # _id format: {cik}-{accession}
            cik_part = entity_id.split("-")[0] if "-" in entity_id else ""
            if cik_part.isdigit():
                return cik_part.zfill(10)

    return None


# ---------------------------------------------------------------------------
# Recent filings from company submissions
# ---------------------------------------------------------------------------

def get_recent_filings(
    ticker: str,
    *,
    cik: str | None = None,
    forms: list[str] | None = None,
    days: int = 90,
    limit: int = 20,
) -> list[dict]:
    """
    Return recent SEC filings for a ticker from EDGAR's submissions API.

    Args:
        ticker: Stock ticker (e.g. "NVDA")
        cik: Optional pre-resolved CIK (10-digit padded). Resolved from ticker if not given.
        forms: List of form types to filter (e.g. ["10-Q", "8-K", "4"]). None = all.
        days: Look back this many days from today.
        limit: Max filings to return.

    Returns:
        List of filing dicts with keys: form, filed, accession, description, cik, ticker,
        sec_url, documents_url.
    """
    cache_key = f"submissions:{ticker}:{','.join(forms or [])}:{days}"
    cached = _cached(cache_key, _SUBMISSION_TTL)
    if cached is not None:
        return cached  # type: ignore[return-value]

    resolved_cik = cik or _cik_from_ticker(ticker)
    if not resolved_cik:
        logger.warning("Could not resolve CIK for %s", ticker)
        return []

    data = _get(f"https://data.sec.gov/submissions/CIK{resolved_cik}.json")
    if not data:
        return []

    filings_data = data.get("filings", {}).get("recent", {})
    if not filings_data:
        return []

    # Build aligned lists
    form_list   = filings_data.get("form", [])
    date_list   = filings_data.get("filingDate", [])
    adsh_list   = filings_data.get("accessionNumber", [])
    desc_list   = filings_data.get("primaryDocument", [])
    report_list = filings_data.get("reportDate", [])

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    results = []

    for form, filed, adsh, primary_doc, report_date in zip(
        form_list, date_list, adsh_list, desc_list, report_list
    ):
        if filed < cutoff:
            continue
        if forms and form not in forms:
            continue

        clean_adsh = str(adsh).replace("-", "")
        cik_clean  = str(resolved_cik).lstrip("0")
        documents_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{clean_adsh}/"
        )
        sec_filing_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={resolved_cik}&type={form}&dateb=&owner=include&count=5"
        )

        results.append({
            "ticker":        ticker.upper(),
            "cik":           resolved_cik,
            "form":          form,
            "filed":         filed,
            "report_date":   str(report_date or ""),
            "accession":     str(adsh),
            "primary_doc":   str(primary_doc or ""),
            "documents_url": documents_url,
            "sec_url":       sec_filing_url,
        })

        if len(results) >= limit:
            break

    _store(cache_key, results)
    return results


# ---------------------------------------------------------------------------
# 8-K real-time events
# ---------------------------------------------------------------------------

def get_recent_8k(ticker: str, *, cik: str | None = None, days: int = 30) -> list[dict]:
    """Return the most recent 8-K filings (material events) for a ticker."""
    return get_recent_filings(ticker, cik=cik, forms=["8-K"], days=days, limit=10)


# ---------------------------------------------------------------------------
# Form 4 insider transactions
# ---------------------------------------------------------------------------

def get_form4_transactions(ticker: str, *, cik: str | None = None, days: int = 90) -> list[dict]:
    """Return recent Form 4 insider transactions for a ticker."""
    return get_recent_filings(ticker, cik=cik, forms=["4"], days=days, limit=20)


# ---------------------------------------------------------------------------
# Full-text filing retrieval (risk factors, MD&A prose)
# ---------------------------------------------------------------------------

def get_filing_text(
    accession: str,
    cik: str,
    *,
    primary_doc: str | None = None,
    max_chars: int = 8000,
) -> str | None:
    """
    Fetch the actual text of a filing document.
    Returns up to max_chars of clean text or None if unavailable.

    Note: SEC documents are HTML/XBRL; we strip tags for readable prose.
    """
    cache_key = f"text:{accession}"
    cached = _cached(cache_key, _FILING_TEXT_TTL)
    if cached is not None:
        return cached  # type: ignore[return-value]

    clean_adsh = accession.replace("-", "")
    cik_clean  = str(cik).lstrip("0")
    base_url   = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{clean_adsh}/"

    # Try the primary document first, then the index
    doc_to_fetch = primary_doc or f"{clean_adsh}.htm"
    urls_to_try = [
        base_url + doc_to_fetch,
        base_url + clean_adsh + ".htm",
        base_url + clean_adsh + ".txt",
    ]

    text = None
    for url in urls_to_try:
        for attempt in range(_MAX_RETRIES):
            _rate_limit()
            try:
                with httpx.Client(headers=_HEADERS, timeout=30.0, follow_redirects=True) as client:
                    resp = client.get(url)
                    if resp.status_code == 429:
                        wait = float(resp.headers.get("Retry-After", _RETRY_BASE * (2 ** attempt)))
                        logger.warning("EDGAR 429 — waiting %.1fs (attempt %d/%d): %s", wait, attempt + 1, _MAX_RETRIES, url)
                        time.sleep(wait)
                        continue
                    if resp.status_code == 200:
                        raw = resp.text
                        # Strip HTML tags
                        text = re.sub(r"<[^>]+>", " ", raw)
                        text = re.sub(r"&[a-z]+;", " ", text)
                        text = re.sub(r"\s{3,}", "\n\n", text).strip()
                        break
                    break  # non-200, non-429: try next URL candidate
            except Exception as exc:
                logger.debug("Filing text fetch failed for %s: %s", url, exc)
                break
        if text:
            break

    if text:
        # Truncate cleanly at sentence boundary
        if len(text) > max_chars:
            cutoff = text.rfind(". ", 0, max_chars)
            text = text[:cutoff + 1] if cutoff > 0 else text[:max_chars]
        _store(cache_key, text)

    return text


# ---------------------------------------------------------------------------
# EFTS full-text search (searches content of all EDGAR filings)
# ---------------------------------------------------------------------------

def search_filings_fulltext(
    query: str,
    *,
    ticker: str | None = None,
    forms: list[str] | None = None,
    days: int = 90,
    limit: int = 10,
) -> list[dict]:
    """
    Search the full text of SEC filings via EDGAR's EFTS search API.

    Args:
        query: Search query string (e.g. "artificial intelligence risk factors")
        ticker: Optional ticker to restrict results to one company
        forms: Optional list of form types (e.g. ["10-Q", "8-K"])
        days: Look back this many calendar days
        limit: Max results

    Returns:
        List of dicts with: entity_name, form, filed, accession, excerpt, cik, documents_url
    """
    cache_key = f"efts:{query}:{ticker}:{','.join(forms or [])}:{days}"
    cached = _cached(cache_key, _SEARCH_TTL)
    if cached is not None:
        return cached  # type: ignore[return-value]

    start_dt = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    params: dict = {
        "q": f'"{query}"' if " " in query else query,
        "dateRange": "custom",
        "startdt": start_dt,
        "hits.hits.total.value": 1,
        "hits.hits._source": "period_of_report,file_date,entity_name,file_num,period_of_report",
    }
    if ticker:
        params["q"] = f'"{ticker}" AND ({params["q"]})'
    if forms:
        params["forms"] = ",".join(forms)

    data = _get("https://efts.sec.gov/LATEST/search-index", params=params)
    if not data:
        return []

    hits = (data.get("hits") or {}).get("hits") or []
    results = []

    for hit in hits[:limit]:
        src         = hit.get("_source") or {}
        entity_id   = hit.get("_id", "")
        highlight   = hit.get("highlight", {})
        excerpt     = " ... ".join(
            (highlight.get("file_description") or highlight.get("period_of_report") or [""])[:2]
        )

        # _id format: {cik}-{accession-no-dashes}
        parts = entity_id.split("-", 1)
        cik_str = parts[0].zfill(10) if parts[0].isdigit() else ""
        accession_raw = parts[1] if len(parts) > 1 else ""

        # Reconstruct standard accession format NNNNNNNNNN-YY-NNNNNN
        accession = ""
        if len(accession_raw) >= 18:
            accession = f"{accession_raw[:10]}-{accession_raw[10:12]}-{accession_raw[12:]}"

        documents_url = ""
        if cik_str and accession_raw:
            cik_clean = cik_str.lstrip("0")
            documents_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_raw}/"

        results.append({
            "entity_name":   src.get("entity_name", ""),
            "form":          src.get("form_type", ""),
            "filed":         src.get("file_date", ""),
            "period":        src.get("period_of_report", ""),
            "accession":     accession,
            "cik":           cik_str,
            "documents_url": documents_url,
            "excerpt":       excerpt,
        })

    _store(cache_key, results)
    return results


# ---------------------------------------------------------------------------
# Convenience: get structured recent-activity summary for a ticker
# ---------------------------------------------------------------------------

def get_ticker_live_summary(
    ticker: str,
    *,
    cik: str | None = None,
    days: int = 60,
) -> dict:
    """
    One-call method: returns a structured dict with recent 8-Ks, insider Form 4s,
    and the most recent 10-Q/10-K filings for a ticker.

    Used by the chat pipeline to inject real-time SEC evidence into LLM context.
    """
    resolved_cik = cik or _cik_from_ticker(ticker)

    # All three in sequence (rate-limited inside each call)
    annual_quarterly = get_recent_filings(
        ticker, cik=resolved_cik, forms=["10-K", "10-Q"], days=days, limit=5
    )
    events_8k  = get_recent_8k(ticker, cik=resolved_cik, days=days)
    form4_list = get_form4_transactions(ticker, cik=resolved_cik, days=days)

    return {
        "ticker":            ticker.upper(),
        "cik":               resolved_cik or "unknown",
        "period_days":       days,
        "filings":           annual_quarterly,
        "material_events":   events_8k,
        "insider_transactions": form4_list,
        "retrieved_at":      datetime.now(timezone.utc).isoformat(),
    }
