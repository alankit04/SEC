"""
raphi_mcp_server.py — RAPHI MCP Server (stdio transport)

Security fixes applied:
    H1 / M3  X-Internal-Token header on every httpx call to FastAPI (set RAPHI_INTERNAL_TOKEN)
    M1        Ticker symbol validated against strict regex ^[A-Z]{1,5}(?:\.[A-Z])?$ before use in URLs

Requires the unified RAPHI server to be running on :9999.

Run standalone for testing:
    cd "/Users/alan/Desktop/SEC Data"
    RAPHI_INTERNAL_TOKEN=dev .venv/bin/python -m backend.raphi_mcp_server
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

try:
    from tool_result_cache import ToolResultCache
except ImportError:  # pragma: no cover
    from backend.tool_result_cache import ToolResultCache

logger = logging.getLogger("raphi.mcp")

BASE_URL = "http://localhost:9999"
app      = Server("raphi")

# H1/M3: Shared secret between MCP server and FastAPI backend
_INTERNAL_TOKEN = os.environ.get("RAPHI_INTERNAL_TOKEN", "")

# M1: Strict ticker allowlist regex — A–Z (1–5), optional class suffix (e.g. BRK.B)
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")

# Figma integration (MCP-backed):
# - FIGMA_ACCESS_TOKEN: Personal access token / OAuth bearer
# - FIGMA_FILE_KEY: default file key for read/write operations
_FIGMA_TOKEN = os.environ.get("FIGMA_ACCESS_TOKEN", "").strip()
_FIGMA_FILE_KEY = os.environ.get("FIGMA_FILE_KEY", "").strip()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_CACHE_DIR = PROJECT_ROOT / ".model_cache"
PORTFOLIO_FILE = PROJECT_ROOT / "portfolio.json"
SETTINGS_FILE = PROJECT_ROOT / "settings.json"
CITATION_SQLITE = DATA_DIR / "citation_index.sqlite"

_TOOL_RESULT_CACHE = ToolResultCache(
    default_ttl_s=int(os.environ.get("RAPHI_TOOL_CACHE_DEFAULT_TTL_S", "120")),
    default_stale_grace_s=int(os.environ.get("RAPHI_TOOL_CACHE_STALE_GRACE_S", "60")),
)

_TOOL_TTLS_S: dict[str, int] = {
    "market_overview": 45,
    "stock_detail": 60,
    "stock_news": 900,
    "sec_filings": 7200,
    "sec_search": 3600,
    "sec_universe": 3600,
    "sec_industries": 3600,
    "ml_signal": 3600,
    "gnn_signal": 3600,
    "gnn_status": 120,
    "portfolio_snapshot": 30,
    "portfolio_alerts": 30,
    "memory_status": 30,
    "memory_retrieve": 30,
    "edgar_live_filings": 300,
    "edgar_search_fulltext": 600,
    "firecrawl_scrape": 1200,
    "firecrawl_search": 900,
    "web_citations": 900,
}

_TOOL_SOURCES: dict[str, str] = {
    "market_overview": "Yahoo Finance via yfinance",
    "stock_detail": "Yahoo Finance via yfinance",
    "stock_news": "Yahoo Finance via yfinance",
    "sec_filings": "SEC Financial Statement Data Sets",
    "sec_search": "SEC Financial Statement Data Sets",
    "sec_universe": "SEC Financial Statement Data Sets",
    "sec_industries": "SEC Financial Statement Data Sets",
    "ml_signal": "SignalEngine (.model_cache)",
    "gnn_signal": "GNN model state (.model_cache)",
    "gnn_status": "GNN model state (.model_cache)",
    "portfolio_snapshot": "Local portfolio",
    "portfolio_alerts": "Local portfolio",
    "memory_status": "Graph memory",
    "memory_retrieve": "Graph memory",
    "edgar_live_filings": "SEC EDGAR live",
    "edgar_search_fulltext": "SEC EDGAR EFTS",
    "firecrawl_scrape": "Firecrawl",
    "firecrawl_search": "Firecrawl",
    "web_citations": "RAPHI citation index",
}

_VERSION_MEMO: dict[str, tuple[float, str]] = {}
_VERSION_MEMO_TTL_S = 20


def _cached_mtime_version(label: str, path: Path) -> str:
    key = f"{label}:{path}"
    now = time.time()
    cached = _VERSION_MEMO.get(key)
    if cached and (now - cached[0]) < _VERSION_MEMO_TTL_S:
        return cached[1]
    try:
        stamp = int(path.stat().st_mtime)
    except Exception:
        stamp = 0
    value = f"{label}-v{stamp}"
    _VERSION_MEMO[key] = (now, value)
    return value


def _tool_ttl(name: str) -> int:
    return int(_TOOL_TTLS_S.get(name, 120))


def _sanitize_scope(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.:@+-]", "_", str(value or "").strip())
    return clean[:128] if clean else ""


def _tool_scope(name: str, arguments: dict) -> str:
    if name in {"portfolio_snapshot", "portfolio_alerts", "memory_status", "memory_retrieve"}:
        argument_scope = _sanitize_scope(arguments.get("__user_scope", ""))
        if argument_scope:
            return argument_scope
        return os.environ.get("RAPHI_CACHE_USER_SCOPE", "local-user")
    return "global"


def _tool_data_version(name: str, arguments: dict) -> str:
    if name in {"sec_filings", "sec_search", "sec_universe", "sec_industries"}:
        return _cached_mtime_version("sec-data", DATA_DIR)
    if name in {"portfolio_snapshot", "portfolio_alerts"}:
        return _cached_mtime_version("portfolio", PORTFOLIO_FILE)
    if name in {"memory_status", "memory_retrieve"}:
        return _cached_mtime_version("memory-settings", SETTINGS_FILE)
    if name in {"web_citations"}:
        return _cached_mtime_version("citation-index", CITATION_SQLITE)
    if name in {"market_overview", "stock_detail", "stock_news"}:
        return os.environ.get("RAPHI_MARKET_DATA_VERSION", "yfinance-v1")
    if name in {"edgar_live_filings", "edgar_search_fulltext"}:
        return os.environ.get("RAPHI_EDGAR_DATA_VERSION", "edgar-live-v1")
    if name in {"firecrawl_scrape", "firecrawl_search"}:
        return os.environ.get("RAPHI_FIRECRAWL_DATA_VERSION", "firecrawl-v1")
    if name in {"ml_signal", "gnn_signal", "gnn_status"}:
        return _cached_mtime_version("model-cache", MODEL_CACHE_DIR)
    return "v1"


def _tool_model_version(name: str, arguments: dict) -> str:
    ticker = str(arguments.get("ticker", "")).strip().upper()
    if name == "ml_signal":
        if ticker:
            return _cached_mtime_version("ml", MODEL_CACHE_DIR / f"{ticker}.pkl")
        return _cached_mtime_version("ml", MODEL_CACHE_DIR)
    if name in {"gnn_signal", "gnn_status"}:
        return _cached_mtime_version("gnn", MODEL_CACHE_DIR)
    return ""


def _maybe_attach_cache_meta(value, meta: dict):
    if os.environ.get("RAPHI_CACHE_EXPOSE_META", "0") != "1":
        return value
    if isinstance(value, dict):
        out = dict(value)
        out["_cache"] = meta
        return out
    return {
        "value": value,
        "_cache": meta,
    }


async def _http_get_json(client: httpx.AsyncClient, path: str, params: dict | None = None):
    response = await client.get(path, params=params)
    return response.json()


async def _http_post_json(client: httpx.AsyncClient, path: str, payload: dict):
    response = await client.post(path, json=payload)
    return response.json()


async def _cached_tool_json(
    *,
    name: str,
    arguments: dict,
    producer,
):
    value, meta = await _TOOL_RESULT_CACHE.get_or_compute(
        tool_name=name,
        arguments=arguments,
        source=_TOOL_SOURCES.get(name, "RAPHI tool"),
        producer=producer,
        ttl_s=_tool_ttl(name),
        stale_grace_s=max(0, _tool_ttl(name) // 2),
        data_version=_tool_data_version(name, arguments),
        model_version=_tool_model_version(name, arguments),
        user_scope=_tool_scope(name, arguments),
    )
    return _maybe_attach_cache_meta(value, meta)


def _validate_ticker(raw: str) -> str:
    """Uppercase and validate ticker. Raises ValueError if invalid."""
    ticker = raw.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise ValueError(
            f"Invalid ticker '{ticker}'. Must be 1–5 uppercase letters with optional class suffix (e.g. NVDA, BRK.B)."
        )
    return ticker


def _get_headers() -> dict:
    """Return auth headers for internal FastAPI calls."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _INTERNAL_TOKEN:
        headers["X-Internal-Token"] = _INTERNAL_TOKEN
    return headers


def _figma_headers() -> dict:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _FIGMA_TOKEN:
        headers["X-Figma-Token"] = _FIGMA_TOKEN
    return headers


def _resolve_figma_file_key(arguments: dict) -> str:
    key = str(arguments.get("file_key") or _FIGMA_FILE_KEY).strip()
    if not key:
        raise ValueError("Figma file key missing. Set FIGMA_FILE_KEY or pass file_key.")
    return key


def _count_frames(node: dict) -> int:
    """Count FRAME nodes in a Figma subtree."""
    if not isinstance(node, dict):
        return 0
    count = 1 if node.get("type") == "FRAME" else 0
    for child in node.get("children") or []:
        count += _count_frames(child)
    return count


async def _figma_get(path: str, params: dict | None = None) -> dict:
    if not _FIGMA_TOKEN:
        raise ValueError("FIGMA_ACCESS_TOKEN is not set. Configure it in environment.")
    query = f"?{urlencode(params)}" if params else ""
    url = f"https://api.figma.com{path}{query}"
    async with httpx.AsyncClient(headers=_figma_headers(), timeout=60.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def _figma_post(path: str, payload: dict) -> dict:
    if not _FIGMA_TOKEN:
        raise ValueError("FIGMA_ACCESS_TOKEN is not set. Configure it in environment.")
    url = f"https://api.figma.com{path}"
    async with httpx.AsyncClient(headers=_figma_headers(), timeout=60.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="market_overview",
            description="Real-time market overview: S&P 500, Nasdaq, VIX, 10Y yield, Gold, DXY.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="stock_detail",
            description="Price, P/E, market cap, sector, fundamentals for a ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. NVDA, BRK.B).",
                        "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$",
                    }
                },
                "required": ["ticker"],
            },
        ),
        types.Tool(
            name="stock_news",
            description="Recent news with VADER sentiment scores for a ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. NVDA, BRK.B).",
                        "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$",
                    }
                },
                "required": ["ticker"],
            },
        ),
        types.Tool(
            name="sec_filings",
            description="SEC EDGAR 10-K/10-Q filings and XBRL financials for a ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. NVDA, BRK.B).",
                        "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$",
                    }
                },
                "required": ["ticker"],
            },
        ),
        types.Tool(
            name="sec_search",
            description="Search SEC EDGAR by company name. Returns CIK, ticker, SIC.",
            inputSchema={
                "type": "object",
                "properties": {
                    "q":     {"type": "string", "description": "Company name or keyword to search", "maxLength": 200},
                    "limit": {"type": "integer", "default": 20, "description": "Max results (1–100)", "minimum": 1, "maximum": 100},
                },
                "required": ["q"],
            },
        ),
        types.Tool(
            name="sec_universe",
            description="Screen the local SEC filing universe beyond the watchlist by name, SIC, industry, form, and ticker availability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Optional company-name keyword", "maxLength": 200},
                    "sic": {"type": "string", "description": "Optional SIC prefix such as 36 for Electronic Equipment", "maxLength": 4},
                    "industry": {"type": "string", "description": "Optional industry keyword such as Banking or Electronic", "maxLength": 80},
                    "form": {"type": "string", "description": "Optional filing form filter such as 10-K or 10-Q", "maxLength": 20},
                    "tickered_only": {"type": "boolean", "default": True},
                    "limit": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="sec_industries",
            description="Aggregate the local SEC universe by 2-digit SIC industry.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="ml_signal",
            description="XGBoost+LSTM ensemble trading signal with SHAP feature explainability. Includes GNN blend when the graph model is trained.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. NVDA, BRK.B).",
                        "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$",
                    }
                },
                "required": ["ticker"],
            },
        ),
        types.Tool(
            name="gnn_signal",
            description="GraphSAGE GNN-only signal with graph-neighbor influence scores for a ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. NVDA, BRK.B).",
                        "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$",
                    }
                },
                "required": ["ticker"],
            },
        ),
        types.Tool(
            name="gnn_status",
            description="GNN model readiness, backend, graph size, cache age, and covered tickers.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="gnn_train",
            description="Trigger real GNN graph rebuild and model training for the watchlist or supplied tickers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string", "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$"},
                        "description": "Optional ticker universe. Defaults to the RAPHI watchlist.",
                    },
                    "force": {
                        "type": "boolean",
                        "default": True,
                        "description": "Force rebuild even if the current cache is still fresh.",
                    },
                    "background": {
                        "type": "boolean",
                        "default": True,
                        "description": "Queue training in the FastAPI background task runner.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="portfolio_snapshot",
            description="Portfolio positions, P&L, VaR 95/99, Sharpe ratio, alpha vs SPY.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="portfolio_alerts",
            description="Active risk alerts: VaR breaches, stop-loss proximity, model drift.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="memory_status",
            description="Permanent graph memory status and Neo4j availability.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="memory_retrieve",
            description="Retrieve relevant permanent graph memory for a query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Query for memory retrieval", "maxLength": 1000},
                    "limit": {"type": "integer", "default": 8, "minimum": 1, "maximum": 25},
                },
                "required": ["q"],
            },
        ),
        types.Tool(
            name="figma_status",
            description="Check whether Figma MCP connection is configured via environment variables.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="figma_get_file",
            description="Fetch Figma file metadata and document tree (optional depth and IDs).",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_key": {"type": "string", "description": "Figma file key (optional if FIGMA_FILE_KEY is set)."},
                    "depth": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Optional node depth limit."},
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of node IDs to restrict payload.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="figma_design_summary",
            description="Compact design diagnostics: page names and frame counts with a deterministic design_present flag.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_key": {"type": "string", "description": "Figma file key (optional if FIGMA_FILE_KEY is set)."},
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                        "default": 2,
                        "description": "Tree depth used for summary extraction. Lower is lighter and less rate-limit prone.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="figma_get_nodes",
            description="Fetch specific nodes from a Figma file by node IDs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_key": {"type": "string", "description": "Figma file key (optional if FIGMA_FILE_KEY is set)."},
                    "ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of node IDs.",
                    },
                },
                "required": ["ids"],
            },
        ),
        types.Tool(
            name="figma_get_comments",
            description="Retrieve comments for a Figma file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_key": {"type": "string", "description": "Figma file key (optional if FIGMA_FILE_KEY is set)."}
                },
                "required": [],
            },
        ),
        types.Tool(
            name="figma_post_comment",
            description="Create a comment in a Figma file at a given x,y position.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_key": {"type": "string", "description": "Figma file key (optional if FIGMA_FILE_KEY is set)."},
                    "message": {"type": "string", "maxLength": 2000},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["message", "x", "y"],
            },
        ),        # ── Live SEC EDGAR ─────────────────────────────────────────────────────
        types.Tool(
            name="edgar_live_filings",
            description=(
                "Real-time SEC EDGAR filings for a ticker: most recent 10-K, 10-Q, 8-K, and Form 4 "
                "insider transactions directly from data.sec.gov. No API key required. "
                "Returns accession numbers, filing dates, and direct document URLs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker (supports class suffix like BRK.B).",
                        "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$",
                    },
                    "days": {
                        "type": "integer",
                        "default": 60,
                        "description": "Look-back window in calendar days (default 60, max 365).",
                        "minimum": 1,
                        "maximum": 365,
                    },
                },
                "required": ["ticker"],
            },
        ),
        types.Tool(
            name="edgar_search_fulltext",
            description=(
                "Search the full text of all SEC filings via EDGAR's EFTS engine. "
                "Finds filings that contain a specific phrase, risk factor keyword, or topic. "
                "Optionally filter by ticker and form type."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search phrase (e.g. 'artificial intelligence risk', 'share repurchase').",
                        "maxLength": 300,
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Optional ticker to restrict search to one company.",
                        "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$",
                    },
                    "forms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional form types to filter (e.g. ['10-Q', '8-K']).",
                    },
                    "days": {
                        "type": "integer",
                        "default": 90,
                        "description": "Look-back window in calendar days.",
                        "minimum": 1,
                        "maximum": 365,
                    },
                    "limit": {
                        "type": "integer",
                        "default": 8,
                        "description": "Max results to return.",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        # ── Firecrawl web scraping ─────────────────────────────────────────────
        types.Tool(
            name="firecrawl_scrape",
            description=(
                "Scrape any web URL and return clean markdown content. "
                "Use for: earnings call transcripts, IR press releases, analyst articles, "
                "news stories by direct URL. Requires FIRECRAWL_API_KEY."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to scrape (must be https://).",
                        "maxLength": 2000,
                    },
                    "max_chars": {
                        "type": "integer",
                        "default": 6000,
                        "description": "Max characters of markdown to return.",
                        "minimum": 500,
                        "maximum": 15000,
                    },
                },
                "required": ["url"],
            },
        ),
        types.Tool(
            name="firecrawl_search",
            description=(
                "Search the web and return scraped markdown from top results. "
                "Use for: earnings transcripts, analyst price targets, company news narratives. "
                "Requires FIRECRAWL_API_KEY."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                        "maxLength": 500,
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return (max 10).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="web_citations",
            description=(
                "Local-first citation search. Searches RAPHI's durable citation index first, "
                "then optionally refreshes missing evidence through Firecrawl."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query.", "maxLength": 500},
                    "ticker": {
                        "type": "string",
                        "description": "Optional stock ticker for query scoping.",
                        "pattern": "^[A-Z]{1,5}(?:\\.[A-Z])?$",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of citation results to return.",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "refresh_if_missing": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use Firecrawl to add new sources only if local index has insufficient evidence.",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    headers = _get_headers()
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=120.0) as client:
        try:
            r = None
            data = None
            if name == "market_overview":
                clean_args: dict = {}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/market/overview"),
                )

            elif name == "stock_detail":
                ticker = _validate_ticker(arguments["ticker"])
                clean_args = {"ticker": ticker}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, f"/api/stock/{ticker}"),
                )

            elif name == "stock_news":
                ticker = _validate_ticker(arguments["ticker"])
                clean_args = {"ticker": ticker}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, f"/api/stock/{ticker}/news"),
                )

            elif name == "sec_filings":
                ticker = _validate_ticker(arguments["ticker"])
                clean_args = {"ticker": ticker}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, f"/api/stock/{ticker}/filings"),
                )

            elif name == "sec_search":
                q = str(arguments.get("q", ""))[:200]   # cap search query length
                limit = min(int(arguments.get("limit", 20)), 100)
                clean_args = {"q": q, "limit": limit}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/sec/search", params=clean_args),
                )

            elif name == "sec_universe":
                clean_args = {
                    "q": str(arguments.get("q", ""))[:200],
                    "sic": str(arguments.get("sic", ""))[:4],
                    "industry": str(arguments.get("industry", ""))[:80],
                    "form": str(arguments.get("form", ""))[:20],
                    "tickered_only": bool(arguments.get("tickered_only", True)),
                    "limit": min(int(arguments.get("limit", 50)), 500),
                }
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/sec/universe", params=clean_args),
                )

            elif name == "sec_industries":
                clean_args = {}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/sec/industries"),
                )

            elif name == "ml_signal":
                ticker = _validate_ticker(arguments["ticker"])
                clean_args = {"ticker": ticker}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, f"/api/stock/{ticker}/signals"),
                )

            elif name == "gnn_signal":
                ticker = _validate_ticker(arguments["ticker"])
                clean_args = {"ticker": ticker}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, f"/api/stock/{ticker}/gnn"),
                )

            elif name == "gnn_status":
                clean_args = {}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/gnn/status"),
                )

            elif name == "gnn_train":
                raw_tickers = arguments.get("tickers") or []
                tickers = [_validate_ticker(t) for t in raw_tickers]
                payload = {
                    "tickers": tickers,
                    "force": bool(arguments.get("force", True)),
                    "background": bool(arguments.get("background", True)),
                }
                r = await client.post("/api/gnn/train", json=payload)
                data = r.json()
                await _TOOL_RESULT_CACHE.invalidate_tool("gnn_status")
                await _TOOL_RESULT_CACHE.invalidate_tool("gnn_signal")
                await _TOOL_RESULT_CACHE.invalidate_tool("ml_signal")

            elif name == "portfolio_snapshot":
                clean_args = {}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/portfolio"),
                )

            elif name == "portfolio_alerts":
                clean_args = {}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/alerts"),
                )

            elif name == "memory_status":
                clean_args = {}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/memory/status"),
                )

            elif name == "memory_retrieve":
                q = str(arguments.get("q", ""))[:1000]
                limit = min(int(arguments.get("limit", 8)), 25)
                clean_args = {"q": q, "limit": limit}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/memory/retrieve", params=clean_args),
                )

            elif name == "figma_status":
                data = {
                    "connected": bool(_FIGMA_TOKEN),
                    "file_key_configured": bool(_FIGMA_FILE_KEY),
                    "required_env": ["FIGMA_ACCESS_TOKEN", "FIGMA_FILE_KEY"],
                }
                text = json.dumps(data, default=str)
                return [types.TextContent(type="text", text=text)]

            elif name == "figma_get_file":
                file_key = _resolve_figma_file_key(arguments)
                params: dict[str, str] = {}
                if arguments.get("depth") is not None:
                    params["depth"] = str(min(max(int(arguments["depth"]), 1), 10))
                ids = arguments.get("ids") or []
                if ids:
                    params["ids"] = ",".join(str(i) for i in ids[:100])
                data = await _figma_get(f"/v1/files/{file_key}", params=params or None)
                text = json.dumps(data, default=str)
                if len(text) > 8000:
                    text = text[:8000] + "...(truncated)"
                return [types.TextContent(type="text", text=text)]

            elif name == "figma_design_summary":
                file_key = _resolve_figma_file_key(arguments)
                depth = min(max(int(arguments.get("depth", 2)), 1), 4)
                data = await _figma_get(f"/v1/files/{file_key}", params={"depth": str(depth)})

                pages = []
                frame_total = 0
                node_total = 0
                for page in (data.get("document", {}) or {}).get("children", []) or []:
                    children = page.get("children") or []
                    child_count = len(children)
                    frame_count = _count_frames(page)
                    frame_total += frame_count
                    node_total += child_count
                    pages.append({
                        "id": page.get("id"),
                        "name": page.get("name"),
                        "frame_count": frame_count,
                        "child_count": child_count,
                    })

                summary = {
                    "file_key": file_key,
                    "file_name": data.get("name"),
                    "last_modified": data.get("lastModified"),
                    "page_count": len(pages),
                    "frame_total": frame_total,
                    "design_present": bool(frame_total > 0 or node_total > 0),
                    "pages": pages,
                    "note": "Summary mode keeps payload small to reduce 429 rate-limit risk.",
                }
                return [types.TextContent(type="text", text=json.dumps(summary, default=str))]

            elif name == "figma_get_nodes":
                file_key = _resolve_figma_file_key(arguments)
                ids = [str(i).strip() for i in (arguments.get("ids") or []) if str(i).strip()]
                if not ids:
                    raise ValueError("figma_get_nodes requires at least one node id.")
                params = {"ids": ",".join(ids[:100])}
                data = await _figma_get(f"/v1/files/{file_key}/nodes", params=params)
                text = json.dumps(data, default=str)
                if len(text) > 8000:
                    text = text[:8000] + "...(truncated)"
                return [types.TextContent(type="text", text=text)]

            elif name == "figma_get_comments":
                file_key = _resolve_figma_file_key(arguments)
                data = await _figma_get(f"/v1/files/{file_key}/comments")
                text = json.dumps(data, default=str)
                if len(text) > 8000:
                    text = text[:8000] + "...(truncated)"
                return [types.TextContent(type="text", text=text)]

            elif name == "figma_post_comment":
                file_key = _resolve_figma_file_key(arguments)
                message = str(arguments.get("message", "")).strip()[:2000]
                if not message:
                    raise ValueError("figma_post_comment requires a non-empty message.")
                x = float(arguments.get("x"))
                y = float(arguments.get("y"))
                payload = {"message": message, "client_meta": {"x": x, "y": y}}
                data = await _figma_post(f"/v1/files/{file_key}/comments", payload)
                text = json.dumps(data, default=str)
                if len(text) > 8000:
                    text = text[:8000] + "...(truncated)"
                return [types.TextContent(type="text", text=text)]

            # ── Live SEC EDGAR tools ───────────────────────────────────────────
            elif name == "edgar_live_filings":
                ticker = _validate_ticker(arguments["ticker"])
                days = min(int(arguments.get("days", 60)), 365)
                clean_args = {"ticker": ticker, "days": days}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, f"/api/stock/{ticker}/live-filings", params={"days": days}),
                )

            elif name == "edgar_search_fulltext":
                query = str(arguments.get("query", ""))[:300]
                ticker_raw = str(arguments.get("ticker", "")).strip().upper()
                ticker_param = _validate_ticker(ticker_raw) if ticker_raw else ""
                forms = [str(f) for f in (arguments.get("forms") or [])][:6]
                days = min(int(arguments.get("days", 90)), 365)
                limit = min(int(arguments.get("limit", 8)), 20)
                params: dict = {"query": query, "days": days, "limit": limit}
                if ticker_param:
                    params["ticker"] = ticker_param
                if forms:
                    params["forms"] = ",".join(forms)
                clean_args = dict(params)
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_get_json(client, "/api/edgar/search", params=params),
                )

            # ── Firecrawl tools ────────────────────────────────────────────────
            elif name == "firecrawl_scrape":
                url = str(arguments.get("url", ""))[:2000]
                if not url.startswith("https://"):
                    raise ValueError("firecrawl_scrape: url must start with https://")
                max_chars = min(int(arguments.get("max_chars", 6000)), 15000)
                clean_args = {"url": url, "max_chars": max_chars}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_post_json(client, "/api/firecrawl/scrape", clean_args),
                )

            elif name == "firecrawl_search":
                query = str(arguments.get("query", ""))[:500]
                limit = min(int(arguments.get("limit", 5)), 10)
                clean_args = {"query": query, "limit": limit}
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_post_json(client, "/api/firecrawl/search", clean_args),
                )

            elif name == "web_citations":
                query = str(arguments.get("query", ""))[:500]
                ticker_raw = str(arguments.get("ticker", "")).strip().upper()
                ticker = _validate_ticker(ticker_raw) if ticker_raw else ""
                limit = min(int(arguments.get("limit", 5)), 10)
                clean_args = {
                    "query": query,
                    "ticker": ticker,
                    "limit": limit,
                    "refresh_if_missing": bool(arguments.get("refresh_if_missing", False)),
                }
                data = await _cached_tool_json(
                    name=name,
                    arguments=clean_args,
                    producer=lambda: _http_post_json(client, "/api/web/citations", clean_args),
                )

            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )]

            if data is None and r is not None:
                data = r.json()
            text = json.dumps(data, default=str)
            if len(text) > 8000:
                text = text[:8000] + "...(truncated)"
            return [types.TextContent(type="text", text=text)]

        except ValueError as e:
            # Ticker / input validation failure
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

        except httpx.ConnectError:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "error": "RAPHI backend not running. Start with: uvicorn backend.raphi_server:app --port 9999"
                }),
            )]

        except Exception as e:
            logger.error("MCP tool %s failed: %s", name, e)
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
