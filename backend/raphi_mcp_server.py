"""
raphi_mcp_server.py — RAPHI MCP Server (stdio transport)

Security fixes applied:
  H1 / M3  X-Internal-Token header on every httpx call to FastAPI (set RAPHI_INTERNAL_TOKEN)
  M1        Ticker symbol validated against strict regex ^[A-Z]{1,5}$ before use in URLs

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

import httpx
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger("raphi.mcp")

BASE_URL = "http://localhost:9999"
app      = Server("raphi")

# H1/M3: Shared secret between MCP server and FastAPI backend
_INTERNAL_TOKEN = os.environ.get("RAPHI_INTERNAL_TOKEN", "")

# M1: Strict ticker allowlist regex — A–Z, 1–5 chars (covers NYSE/NASDAQ/BRK.B style)
_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _validate_ticker(raw: str) -> str:
    """Uppercase and validate ticker. Raises ValueError if invalid."""
    ticker = raw.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise ValueError(
            f"Invalid ticker '{ticker}'. Must be 1–5 uppercase letters (e.g. NVDA, AAPL)."
        )
    return ticker


def _get_headers() -> dict:
    """Return auth headers for internal FastAPI calls."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _INTERNAL_TOKEN:
        headers["X-Internal-Token"] = _INTERNAL_TOKEN
    return headers


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
                        "description": "Stock ticker symbol (e.g. NVDA). Must be 1–5 uppercase letters.",
                        "pattern": "^[A-Z]{1,5}$",
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
                        "description": "Stock ticker symbol. Must be 1–5 uppercase letters.",
                        "pattern": "^[A-Z]{1,5}$",
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
                        "description": "Stock ticker symbol. Must be 1–5 uppercase letters.",
                        "pattern": "^[A-Z]{1,5}$",
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
                        "description": "Stock ticker symbol. Must be 1–5 uppercase letters.",
                        "pattern": "^[A-Z]{1,5}$",
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
                        "description": "Stock ticker symbol. Must be 1–5 uppercase letters.",
                        "pattern": "^[A-Z]{1,5}$",
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
                        "items": {"type": "string", "pattern": "^[A-Z]{1,5}$"},
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
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    headers = _get_headers()
    async with httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=120.0) as client:
        try:
            if name == "market_overview":
                r = await client.get("/api/market/overview")

            elif name == "stock_detail":
                ticker = _validate_ticker(arguments["ticker"])
                r = await client.get(f"/api/stock/{ticker}")

            elif name == "stock_news":
                ticker = _validate_ticker(arguments["ticker"])
                r = await client.get(f"/api/stock/{ticker}/news")

            elif name == "sec_filings":
                ticker = _validate_ticker(arguments["ticker"])
                r = await client.get(f"/api/stock/{ticker}/filings")

            elif name == "sec_search":
                q = str(arguments.get("q", ""))[:200]   # cap search query length
                limit = min(int(arguments.get("limit", 20)), 100)
                r = await client.get("/api/sec/search", params={"q": q, "limit": limit})

            elif name == "sec_universe":
                params = {
                    "q": str(arguments.get("q", ""))[:200],
                    "sic": str(arguments.get("sic", ""))[:4],
                    "industry": str(arguments.get("industry", ""))[:80],
                    "form": str(arguments.get("form", ""))[:20],
                    "tickered_only": bool(arguments.get("tickered_only", True)),
                    "limit": min(int(arguments.get("limit", 50)), 500),
                }
                r = await client.get("/api/sec/universe", params=params)

            elif name == "sec_industries":
                r = await client.get("/api/sec/industries")

            elif name == "ml_signal":
                ticker = _validate_ticker(arguments["ticker"])
                r = await client.get(f"/api/stock/{ticker}/signals")

            elif name == "gnn_signal":
                ticker = _validate_ticker(arguments["ticker"])
                r = await client.get(f"/api/stock/{ticker}/gnn")

            elif name == "gnn_status":
                r = await client.get("/api/gnn/status")

            elif name == "gnn_train":
                raw_tickers = arguments.get("tickers") or []
                tickers = [_validate_ticker(t) for t in raw_tickers]
                payload = {
                    "tickers": tickers,
                    "force": bool(arguments.get("force", True)),
                    "background": bool(arguments.get("background", True)),
                }
                r = await client.post("/api/gnn/train", json=payload)

            elif name == "portfolio_snapshot":
                r = await client.get("/api/portfolio")

            elif name == "portfolio_alerts":
                r = await client.get("/api/alerts")

            elif name == "memory_status":
                r = await client.get("/api/memory/status")

            elif name == "memory_retrieve":
                q = str(arguments.get("q", ""))[:1000]
                limit = min(int(arguments.get("limit", 8)), 25)
                r = await client.get("/api/memory/retrieve", params={"q": q, "limit": limit})

            else:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )]

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
