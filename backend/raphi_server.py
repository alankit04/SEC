"""
raphi_server.py — RAPHI Unified Server  (A2A primary · FastAPI data router · port 9999)

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  RAPHI — single FastAPI/Starlette app  (port 9999)  │
  │                                                     │
  │  A2A Protocol (primary entry point)                 │
  │    POST /                  ← A2A task submission    │
  │    GET  /.well-known/*     ← agent card             │
  │    GET  /extended-card     ← auth'd extended card   │
  │                                                     │
  │  Agent Swarm (Claude Agent SDK subagents)           │
  │    @market-analyst    → real-time prices + news     │
  │    @sec-researcher    → EDGAR XBRL financials       │
  │    @ml-signals        → XGBoost+LSTM predictions    │
  │    @portfolio-risk    → VaR, Sharpe, stop-loss      │
  │    @memo-synthesizer  → full investment memo        │
  │                                                     │
  │  Data API sub-router  /api/*  (FastAPI router)      │
  │  Dashboard            GET /   (static HTML)         │
  │  Static assets        /static/*                     │
  └─────────────────────────────────────────────────────┘

Run:
    cd "/Users/alan/Desktop/SEC Data"
    .venv/bin/python -m backend.raphi_server
"""

import json
import os
import re
import sys
from pathlib import Path

import uvicorn
import pandas as pd
from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ── SDK / A2A imports ─────────────────────────────────────────────────
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

sys.path.insert(0, str(Path(__file__).parent))

from typing import Optional

from market_data       import MarketData
from sec_data          import SECData
from ml_model          import SignalEngine
from gnn_model         import GNNSignalEngine
from portfolio_manager import PortfolioManager
from a2a_executor_v2   import RaphiAgent, RaphiAgentExecutor
from security          import TokenAuth, init_sentry, sanitize_user_input
from graph_memory      import GraphMemoryError, get_graph_memory
from llm_guardrails    import GuardrailContext, validate_and_repair_response
from conviction_store  import (
    CONVICTIONS_FILE,
    RESOLUTIONS_FILE,
    write_conviction,
    check_pending,
    get_accuracy_stats,
    get_ledger,
)
from model_optimization import (
    ReinforcementPolicy,
    optimization_status,
    optimize_from_conviction_ledger,
)
import raphi_mcp_server as mcp_bridge
import edgar_live
import firecrawl_client
import web_citations

# ── Initialise Sentry before anything else ───────────────────────────
init_sentry()

# ── Paths ─────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
SETTINGS_FILE     = BASE / "settings.json"
DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "META", "TSLA", "AMZN", "GOOGL"]
TICKER_RE         = re.compile(r"^[A-Z]{1,5}$")

TICKER_IDENTITY_OVERRIDES = {
    "ASST": {
        "current_name": "Strive, Inc.",
        "former_name": "Asset Entities Inc.",
        "identity_note": (
            "ASST is Strive, Inc. after the Asset Entities / Strive merger; "
            "legacy SEC and market metadata may still reference Asset Entities Inc."
        ),
        "strategy_note": "Public Bitcoin treasury / asset-management strategy.",
    }
}

# ── Data singletons ───────────────────────────────────────────────────
market    = MarketData()
sec       = SECData(BASE)
engine    = SignalEngine()
portfolio = PortfolioManager()
memory    = get_graph_memory()
gnn       = GNNSignalEngine.get(sec)

# ══════════════════════════════════════════════════════════════════════
# A2A AGENT CARD  — describes the RAPHI swarm to the outside world
# ══════════════════════════════════════════════════════════════════════
_agent_card = AgentCard(
    name="RAPHI",
    description=(
        "Institutional-grade AI investment platform. "
        "Swarm of 5 specialised subagents: market analyst, SEC researcher, "
        "ML signals, portfolio risk, and memo synthesizer — all orchestrated "
        "via the Agent-to-Agent (A2A) protocol and Claude Agent SDK."
    ),
    url="http://localhost:9999/",
    version="2.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="market_intel",
            name="Market Intelligence",
            description=(
                "Real-time prices, technicals, fundamentals, and news sentiment "
                "via @market-analyst subagent."
            ),
            tags=["market", "stocks", "prices", "news", "sentiment"],
            examples=["What's the price of NVDA?", "Give me a market overview"],
        ),
        AgentSkill(
            id="sec_research",
            name="SEC Filings Research",
            description=(
                "16 quarters of EDGAR XBRL data (9,457+ companies) via "
                "@sec-researcher subagent."
            ),
            tags=["sec", "filings", "edgar", "10-K", "10-Q"],
            examples=["Find Apple SEC filings", "Tesla Q3 revenue trend"],
        ),
        AgentSkill(
            id="ml_signals",
            name="ML And Graph Trading Signals",
            description=(
                "XGBoost + LSTM ensemble, SHAP explainability, and GraphSAGE "
                "neighbor influence via @ml-signals subagent."
            ),
            tags=["ml", "signals", "xgboost", "lstm", "shap", "gnn", "graphsage"],
            examples=["Generate signal for MSFT", "Show NVDA graph-neighbor influence"],
        ),
        AgentSkill(
            id="portfolio_risk",
            name="Portfolio Risk",
            description=(
                "VaR (95/99%), Sharpe, alpha, P&L attribution, stop-loss "
                "alerts via @portfolio-risk subagent."
            ),
            tags=["portfolio", "var", "sharpe", "risk", "alerts"],
            examples=["Show portfolio risk", "What are my alerts?"],
        ),
        AgentSkill(
            id="investment_memo",
            name="Investment Memo",
            description=(
                "Full buy/sell/hold memo from @memo-synthesizer, which "
                "orchestrates all four subagents in parallel."
            ),
            tags=["memo", "buy", "sell", "hold", "recommendation"],
            examples=["Write memo for GOOGL", "Should I buy TSLA?"],
        ),
    ],
)

# ══════════════════════════════════════════════════════════════════════
# A2A HANDLER + EXECUTOR  (Claude Agent SDK swarm)
# ══════════════════════════════════════════════════════════════════════
_agent    = RaphiAgent()
_executor = RaphiAgentExecutor(_agent)
_handler  = DefaultRequestHandler(
    agent_executor=_executor,
    task_store=InMemoryTaskStore(),
)

# ══════════════════════════════════════════════════════════════════════
# FASTAPI APP  (unified — A2A routes injected directly)
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="RAPHI — A2A Agent Swarm",
    version="2.0.0",
    description="Real-time Agentic Platform for Human Investment Intelligence",
)

# ── Rate limiter (slowapi) ────────────────────────────────────────────
# data endpoints: 60/min · AI endpoints: 5/min · health: unlimited
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

api_key        = os.environ.get("RAPHI_API_KEY", "")
internal_token = os.environ.get("RAPHI_INTERNAL_TOKEN", "")
if not api_key:
    import warnings
    warnings.warn("RAPHI_API_KEY not set — A2A server is UNPROTECTED", stacklevel=1)

# ── Security middleware ───────────────────────────────────────────────
# TokenAuth must be added BEFORE CORSMiddleware (inner → outer execution)
# H1/M3: internal_token lets MCP bridge authenticate via X-Internal-Token
app.add_middleware(TokenAuth, api_key=api_key, internal_token=internal_token)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:9999",
        "http://localhost:8000",
        "http://127.0.0.1:9999",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["POST", "GET", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
)

# ── Inject A2A protocol routes into FastAPI (A2AStarletteApplication
#    calls app.routes.extend(routes) — works on any Starlette subclass) ──
_a2a_app = A2AStarletteApplication(
    agent_card=_agent_card,
    http_handler=_handler,
)
_a2a_app.add_routes_to_app(app)   # POST / + GET /.well-known/* now on FastAPI

# ── Dashboard + static ────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the RAPHI A2A dashboard (GET / → HTML; POST / → A2A handled above)."""
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class MCPBridgeRequest(BaseModel):
    tool: str
    arguments: dict = Field(default_factory=dict)


@app.post("/mcp")
async def mcp_bridge_call(body: MCPBridgeRequest):
    """Bridge HTTP /mcp requests to the stdio MCP tool implementations."""
    raw_tool = str(body.tool or "").strip()
    if not raw_tool:
        raise HTTPException(422, "tool is required")

    tool_name = raw_tool
    if tool_name.startswith("mcp__raphi__"):
        tool_name = tool_name[len("mcp__raphi__"):]

    try:
        contents = await mcp_bridge.call_tool(tool_name, body.arguments or {})
    except Exception as exc:
        raise HTTPException(422, str(exc))

    text_chunks: list[str] = [
        getattr(item, "text", "")
        for item in contents
        if getattr(item, "text", "")
    ]
    text = "\n".join(text_chunks).strip()
    if not text:
        return {"tool": raw_tool, "result": None}

    try:
        parsed = json.loads(text)
        return {"tool": raw_tool, "result": parsed}
    except Exception:
        return {"tool": raw_tool, "result": {"text": text}}


# ══════════════════════════════════════════════════════════════════════
# DATA API SUB-ROUTER  (/api/*)
# ══════════════════════════════════════════════════════════════════════
api = APIRouter(prefix="/api", tags=["data"])


# ── helpers ───────────────────────────────────────────────────────────
def _sse(event: str, data: str) -> str:
    if event == "token":
        data = json.dumps(str(data))
    return f"event: {event}\ndata: {data}\n\n"


def _now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M")


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        import json
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {"watchlist": DEFAULT_WATCHLIST}


def _ticker_symbol(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not TICKER_RE.match(ticker):
        raise HTTPException(422, f"Invalid ticker '{ticker}'. Use 1-5 uppercase letters.")
    return ticker


def _watchlist() -> list[str]:
    raw_watchlist = _load_settings().get("watchlist", DEFAULT_WATCHLIST)
    tickers: list[str] = []
    seen: set[str] = set()
    for raw in raw_watchlist:
        try:
            ticker = _ticker_symbol(raw)
        except HTTPException:
            continue
        if ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
    return tickers or DEFAULT_WATCHLIST


def _ticker_identity(ticker: str) -> dict:
    ticker = _ticker_symbol(ticker)
    return TICKER_IDENTITY_OVERRIDES.get(ticker, {"current_name": ticker})


def _apply_ticker_identity(ticker: str, detail: dict | None) -> dict:
    ticker = _ticker_symbol(ticker)
    detail = dict(detail or {})
    identity = _ticker_identity(ticker)
    if ticker in TICKER_IDENTITY_OVERRIDES:
        original_name = detail.get("name") or detail.get("longName") or ""
        detail["name"] = identity["current_name"]
        detail["current_name"] = identity["current_name"]
        detail["former_name"] = identity.get("former_name")
        detail["identity_note"] = identity.get("identity_note")
        detail["strategy_note"] = identity.get("strategy_note")
        if original_name and original_name != identity["current_name"]:
            detail["provider_name"] = original_name
    return detail


def _gnn_universe(*extra_tickers: str, requested: Optional[list[str]] = None) -> list[str]:
    universe: list[str] = []
    seen: set[str] = set()
    for raw in [*extra_tickers, *(requested or []), *_watchlist()]:
        ticker = _ticker_symbol(raw)
        if ticker not in seen:
            universe.append(ticker)
            seen.add(ticker)
    if len(universe) < 2:
        raise HTTPException(422, "GNN needs at least 2 valid tickers.")
    return universe


def _register_ticker_for_agentic_analysis(ticker: str) -> dict:
    """Persist a newly requested ticker and attempt to include it in the GNN graph."""
    ticker = _ticker_symbol(ticker)
    settings = _load_settings()
    watchlist = []
    seen = set()
    for raw in settings.get("watchlist", DEFAULT_WATCHLIST):
        try:
            symbol = _ticker_symbol(raw)
        except HTTPException:
            continue
        if symbol not in seen:
            watchlist.append(symbol)
            seen.add(symbol)

    added = ticker not in seen
    if added:
        watchlist.append(ticker)
        settings["watchlist"] = watchlist
        settings.setdefault("auto_added_tickers", [])
        if ticker not in settings["auto_added_tickers"]:
            settings["auto_added_tickers"].append(ticker)
        settings.setdefault("ticker_identities", {})
        if ticker in TICKER_IDENTITY_OVERRIDES:
            settings["ticker_identities"][ticker] = TICKER_IDENTITY_OVERRIDES[ticker]
        _save_settings(settings)

    universe = _gnn_universe(ticker, requested=watchlist)
    result = {
        "ticker": ticker,
        "added_to_watchlist": added,
        "universe": universe,
        "gnn_added": False,
        "gnn_status": {},
    }
    try:
        gnn.ensure_trained(universe)
        status = gnn.status()
        result["gnn_status"] = status
        result["gnn_added"] = ticker in set(status.get("tickers", []))
    except Exception as exc:
        result["gnn_error"] = str(exc)
    return result


def _save_settings(s: dict) -> None:
    import json
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)


def _anthropic_api_key() -> str:
    """Return the Anthropic key from env first, then project settings."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    return str(_load_settings().get("anthropic_api_key", "")).strip()


def _fmt_portfolio(snap: dict) -> str:
    lines = [
        f"Total: ${snap.get('total_value', 0):,.0f} | "
        f"P&L: ${snap.get('total_pnl', 0):,.0f} ({snap.get('total_pnl_pct', 0):+.1f}%) | "
        f"VaR 95%: ${snap.get('var_95', 0):,.0f} | "
        f"Sharpe: {snap.get('sharpe', 0):.2f}"
    ]
    for p in snap.get("positions", []):
        lines.append(
            f"  {p['ticker']} {p.get('direction','LONG')} {p.get('shares',0)}sh "
            f"@ ${p.get('entry_price',0):.2f} → ${p.get('current_price',0):.2f} "
            f"({p.get('pnl_pct',0):+.1f}%)"
        )
    return "\n".join(lines)


def _memory_context(query: str, limit: int = 6) -> str:
    """Retrieve compact permanent memory context without blocking user flows."""
    try:
        memories = memory.retrieve_context(query, limit=limit)
        return memory.format_context(memories)
    except Exception:
        return ""


def _maybe_write_conviction(ticker: str, sig_cache_path: Path, response_text: str) -> None:
    """
    After a chat or memo response, write a conviction if the response contains
    a Signal View conclusion. Fires-and-forgets — never raises, never blocks.
    """
    import re, pickle
    try:
        sv_match = re.search(
            r"Signal\s*View[:\s]+([Pp]ositive|[Nn]egative|[Nn]eutral)", response_text
        )
        if not sv_match:
            return

        signal_view = sv_match.group(1).capitalize()

        if not sig_cache_path.exists():
            return
        with open(sig_cache_path, "rb") as f:
            sig = pickle.load(f)

        ml_dir  = sig.get("direction", "NEUTRAL")
        ml_prob = sig.get("confidence", 50) / 100
        ml_ver  = "xgb_v2.1"

        detail = market.stock_detail(ticker.upper())
        price  = detail.get("price")
        if not price:
            return

        fin         = sec.company_financial_entries(ticker.upper())
        latest_rev  = None
        sec_trend   = None
        sec_due     = None
        sec_qtrs    = 0
        if fin:
            rev_tags = {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                        "SalesRevenueNet"}
            rev_entries = sorted(
                [e for e in fin if e.get("tag") in rev_tags and e.get("val")],
                key=lambda x: x.get("period", ""), reverse=True
            )
            if len(rev_entries) >= 2:
                sec_qtrs   = len(rev_entries)
                latest_rev = float(rev_entries[0]["val"])
                prev_rev   = float(rev_entries[1]["val"])
                delta      = (latest_rev - prev_rev) / prev_rev * 100 if prev_rev else 0
                sec_trend  = "accelerating" if delta > 3 else (
                             "decelerating" if delta < -3 else "stable")

        conviction_tier = "HIGH" if ml_prob >= 0.70 else ("MEDIUM" if ml_prob >= 0.60 else "LOW")

        write_conviction(
            ticker=ticker.upper(),
            ml_direction=ml_dir,
            ml_probability=ml_prob,
            ml_model_version=ml_ver,
            sec_trend=sec_trend,
            sec_latest_revenue=latest_rev,
            sec_quarters_used=sec_qtrs,
            sec_next_filing_due=sec_due,
            signal_view=signal_view,
            conviction=conviction_tier,
            source="chat",
            entry_price=float(price),
        )
    except Exception as e:
        import logging
        logging.getLogger("raphi.convictions").warning("_maybe_write_conviction: %s", e)


# ── health ────────────────────────────────────────────────────────────
@api.get("/health")
def health():
    return {"status": "ok", "server": "raphi-unified", "a2a": True}


# ── market ────────────────────────────────────────────────────────────
@api.get("/market/overview")
@limiter.limit("60/minute")
def market_overview(request: Request):
    return market.market_overview()


@api.get("/stock/{ticker}")
@limiter.limit("60/minute")
def stock_detail(ticker: str, request: Request):
    ticker = _ticker_symbol(ticker)
    data = _apply_ticker_identity(ticker, market.stock_detail(ticker))
    if "error" in data:
        raise HTTPException(404, data["error"])
    return data


@api.get("/stock/{ticker}/news")
@limiter.limit("60/minute")
def stock_news(ticker: str, request: Request):
    return market.stock_news(ticker.upper())


@api.get("/stock/{ticker}/signals")
@limiter.limit("60/minute")
def stock_signals(ticker: str, request: Request, bg: BackgroundTasks = None):
    ticker = _ticker_symbol(ticker)
    detail = market.stock_detail(ticker)
    funds  = {
        "pe_ratio":       detail.get("pe_ratio"),
        "revenue_growth": detail.get("revenue_growth"),
    }
    result = engine.train_and_predict(ticker, funds)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result


class GNNTrainRequest(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    force: bool = True
    background: bool = True


@api.get("/stock/{ticker}/gnn")
@limiter.limit("30/minute")
def stock_gnn(ticker: str, request: Request):
    ticker = _ticker_symbol(ticker)
    result = gnn.predict(ticker, _gnn_universe(ticker))
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result


@api.get("/gnn/signals")
@limiter.limit("30/minute")
def gnn_signals(request: Request, tickers: Optional[str] = None):
    requested = (
        [part.strip() for part in tickers.split(",") if part.strip()]
        if tickers else None
    )
    universe = _gnn_universe(requested=requested)
    results = gnn.predict_batch(universe)
    errors = {ticker: data for ticker, data in results.items() if "error" in data}
    if errors and len(errors) == len(results):
        raise HTTPException(422, {"error": "GNN batch prediction failed", "details": errors})
    return {"signals": results, "status": gnn.status(), "errors": errors}


@api.get("/gnn/status")
@limiter.limit("60/minute")
def gnn_status(request: Request):
    return gnn.status()


@api.post("/gnn/train")
@limiter.limit("10/minute")
def gnn_train(request: Request, bg: BackgroundTasks, body: Optional[GNNTrainRequest] = None):
    train_request = body or GNNTrainRequest()
    universe = _gnn_universe(requested=train_request.tickers or None)

    def _train() -> None:
        gnn.ensure_trained(universe, force=train_request.force)

    if train_request.background:
        bg.add_task(_train)
        return {
            "status": "training_started",
            "tickers": universe,
            "force": train_request.force,
            "background": True,
        }

    try:
        _train()
    except Exception as exc:
        raise HTTPException(422, str(exc))
    return {
        "status": "trained",
        "tickers": universe,
        "force": train_request.force,
        "background": False,
        "gnn": gnn.status(),
    }


@api.get("/stock/{ticker}/filings")
@limiter.limit("60/minute")
def stock_filings(ticker: str, request: Request):
    ticker = _ticker_symbol(ticker)
    filings = sec.ticker_filings(ticker)
    financials = sec.company_financials(ticker)
    financial_citations = sec.company_financial_citations(ticker)
    return {
        "cik":        sec.cik_for_ticker(ticker),
        "filings":    filings,
        "financials": financials,
        "financial_citations": financial_citations,
        "citation_count": len(financial_citations) + len(filings),
        "source": "SEC Financial Statement Data Sets and SEC EDGAR Archives",
    }


@api.get("/stock/{ticker}/live-filings")
@limiter.limit("30/minute")
def stock_live_filings(ticker: str, request: Request, days: int = 60):
    """Real-time SEC EDGAR filings: 10-K, 10-Q, 8-K, Form 4 from data.sec.gov."""
    ticker = _ticker_symbol(ticker)
    days = min(max(int(days), 1), 365)
    try:
        summary = edgar_live.get_ticker_live_summary(ticker, days=days)
        return summary
    except Exception as exc:
        raise HTTPException(502, f"EDGAR live fetch failed: {exc}")


@api.get("/edgar/search")
@limiter.limit("20/minute")
def edgar_fulltext_search(
    request: Request,
    query: str,
    ticker: Optional[str] = None,
    forms: Optional[str] = None,
    days: int = 90,
    limit: int = 8,
):
    """Full-text search across all EDGAR filings via EFTS."""
    days = min(max(int(days), 1), 365)
    limit = min(max(int(limit), 1), 20)
    form_list = [f.strip() for f in forms.split(",")] if forms else None
    try:
        results = edgar_live.search_filings_fulltext(
            query[:300],
            ticker=ticker.strip().upper() if ticker else None,
            forms=form_list,
            days=days,
            limit=limit,
        )
        return {"results": results, "count": len(results)}
    except Exception as exc:
        raise HTTPException(502, f"EDGAR search failed: {exc}")


class FirecrawlScrapeRequest(BaseModel):
    url: str
    max_chars: int = 6000


class FirecrawlSearchRequest(BaseModel):
    query: str
    limit: int = 5


class WebCitationRequest(BaseModel):
    query: str
    ticker: str = ""
    limit: int = 5


@api.post("/firecrawl/scrape")
@limiter.limit("10/minute")
def firecrawl_scrape_route(body: FirecrawlScrapeRequest, request: Request):
    """Scrape a URL via Firecrawl and return clean markdown."""
    if not body.url.startswith("https://"):
        raise HTTPException(422, "url must start with https://")
    max_chars = min(max(int(body.max_chars), 500), 15000)
    result = firecrawl_client.scrape_url(body.url, max_chars=max_chars)
    if not result.get("success"):
        raise HTTPException(502, result.get("error", "scrape failed"))
    return result


@api.post("/firecrawl/search")
@limiter.limit("10/minute")
def firecrawl_search_route(body: FirecrawlSearchRequest, request: Request):
    """Search the web via Firecrawl and return scraped markdown from top results."""
    limit = min(max(int(body.limit), 1), 10)
    results = firecrawl_client.search_web(body.query[:500], limit=limit, scrape_results=True)
    errors = [r for r in results if not r.get("success")]
    if errors and len(errors) == len(results):
        raise HTTPException(502, errors[0].get("error", "search failed"))
    return {"results": [r for r in results if r.get("success")], "count": len(results)}


@api.post("/web/citations")
@limiter.limit("20/minute")
def web_citations_route(body: WebCitationRequest, request: Request):
    """Firecrawl-backed web citation search for Perplexity-like sourced answers."""
    ticker = _ticker_symbol(body.ticker) if body.ticker else ""
    limit = min(max(int(body.limit), 1), 10)
    result = web_citations.search_citations(
        body.query,
        ticker=ticker,
        limit=limit,
    )
    if result.get("error") and not result.get("results"):
        raise HTTPException(502, result["error"])
    return result


def _clean_price_rows(df, fields: list[str], limit: int = 8) -> list[dict]:
    rows: list[dict] = []
    if df is None or getattr(df, "empty", True):
        return rows
    clean = df.replace({float("inf"): None, float("-inf"): None}).where(df.notnull(), None)
    for row in clean.head(limit).to_dict(orient="records"):
        rows.append({field: row.get(field) for field in fields})
    return rows


@api.get("/stock/{ticker}/technicals")
@limiter.limit("60/minute")
def stock_technicals(ticker: str, request: Request):
    ticker = _ticker_symbol(ticker)
    hist = market.ohlcv(ticker, period="1y")
    if hist is None or hist.empty or "Close" not in hist or len(hist) < 30:
        raise HTTPException(422, f"Not enough price history for {ticker}")

    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna() if "Volume" in hist else None
    if len(close) < 30:
        raise HTTPException(422, f"Not enough close history for {ticker}")

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd - macd_signal).iloc[-1])

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    stdev20 = float(close.rolling(20).std().iloc[-1])
    last = float(close.iloc[-1])
    upper = sma20 + (2 * stdev20)
    lower = sma20 - (2 * stdev20)
    bollinger_pct = ((last - lower) / (upper - lower) * 100) if upper != lower else 50

    ret_5d = ((last / float(close.iloc[-6])) - 1) * 100 if len(close) >= 6 else None
    ret_20d = ((last / float(close.iloc[-21])) - 1) * 100 if len(close) >= 21 else None
    ret_50d = ((last / float(close.iloc[-51])) - 1) * 100 if len(close) >= 51 else None
    realized_vol = float(close.pct_change().dropna().tail(20).std() * (252 ** 0.5) * 100)
    volume_ratio = None
    if volume is not None and len(volume) >= 20:
        avg_volume = float(volume.tail(20).mean())
        volume_ratio = float(volume.iloc[-1] / avg_volume) if avg_volume else None

    bullish_votes = 0
    bearish_votes = 0
    bullish_votes += int(last > sma20) + int(sma50 is not None and last > sma50)
    bearish_votes += int(last < sma20) + int(sma50 is not None and last < sma50)
    bullish_votes += int(macd_hist > 0) + int((ret_20d or 0) > 0)
    bearish_votes += int(macd_hist < 0) + int((ret_20d or 0) < 0)
    trend = "bullish" if bullish_votes > bearish_votes else "bearish" if bearish_votes > bullish_votes else "neutral"

    return {
        "ticker": ticker,
        "price": round(last, 2),
        "summary": {
            "trend": trend,
            "bullish_votes": bullish_votes,
            "bearish_votes": bearish_votes,
            "realized_vol_20d": round(realized_vol, 2),
        },
        "indicators": [
            {"name": "RSI 14", "value": round(rsi, 2), "signal": "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"},
            {"name": "MACD Histogram", "value": round(macd_hist, 3), "signal": "bullish" if macd_hist > 0 else "bearish" if macd_hist < 0 else "neutral"},
            {"name": "SMA 20", "value": round(sma20, 2), "signal": "above" if last > sma20 else "below"},
            {"name": "SMA 50", "value": round(sma50, 2) if sma50 else None, "signal": "above" if sma50 and last > sma50 else "below" if sma50 else "unavailable"},
            {"name": "SMA 200", "value": round(sma200, 2) if sma200 else None, "signal": "above" if sma200 and last > sma200 else "below" if sma200 else "unavailable"},
            {"name": "Bollinger Position", "value": round(bollinger_pct, 1), "signal": "upper band" if bollinger_pct >= 80 else "lower band" if bollinger_pct <= 20 else "mid band"},
            {"name": "5D Return", "value": round(ret_5d, 2) if ret_5d is not None else None, "signal": "positive" if (ret_5d or 0) > 0 else "negative" if ret_5d is not None else "unavailable"},
            {"name": "20D Return", "value": round(ret_20d, 2) if ret_20d is not None else None, "signal": "positive" if (ret_20d or 0) > 0 else "negative" if ret_20d is not None else "unavailable"},
            {"name": "50D Return", "value": round(ret_50d, 2) if ret_50d is not None else None, "signal": "positive" if (ret_50d or 0) > 0 else "negative" if ret_50d is not None else "unavailable"},
            {"name": "Volume Ratio", "value": round(volume_ratio, 2) if volume_ratio is not None else None, "signal": "elevated" if volume_ratio and volume_ratio >= 1.3 else "normal" if volume_ratio else "unavailable"},
        ],
    }


@api.get("/stock/{ticker}/options")
@limiter.limit("30/minute")
def stock_options(ticker: str, request: Request, expiration: Optional[str] = None):
    import yfinance as yf

    ticker = _ticker_symbol(ticker)
    try:
        contract = yf.Ticker(ticker)
        expirations = list(contract.options or [])
        if not expirations:
            return {"ticker": ticker, "available": False, "reason": "No listed option expirations returned by provider"}
        selected = expiration if expiration in expirations else expirations[0]
        chain = contract.option_chain(selected)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        call_volume = int(calls.get("volume", []).fillna(0).sum()) if "volume" in calls else 0
        put_volume = int(puts.get("volume", []).fillna(0).sum()) if "volume" in puts else 0
        ratio = round(put_volume / call_volume, 2) if call_volume else None
        fields = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
        sort_cols = [col for col in ["volume", "openInterest"] if col in calls]
        calls_top = calls.sort_values(sort_cols, ascending=False) if sort_cols else calls
        puts_top = puts.sort_values(sort_cols, ascending=False) if sort_cols else puts
        return {
            "ticker": ticker,
            "available": True,
            "expiration": selected,
            "expirations": expirations[:12],
            "put_call_volume_ratio": ratio,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "calls": _clean_price_rows(calls_top, fields),
            "puts": _clean_price_rows(puts_top, fields),
        }
    except Exception as exc:
        return {"ticker": ticker, "available": False, "reason": str(exc)}


# ── portfolio ─────────────────────────────────────────────────────────
@api.get("/portfolio")
@limiter.limit("60/minute")
def get_portfolio(request: Request):
    return portfolio.snapshot()


class Positions(BaseModel):
    positions: list


@api.put("/portfolio")
@limiter.limit("60/minute")
def update_portfolio(body: Positions, request: Request):
    portfolio.update_positions(body.positions)
    return {"ok": True}


class PortfolioPositionRequest(BaseModel):
    ticker: str
    shares: float = 1
    entry_price: Optional[float] = None
    direction: str = "LONG"
    stop_loss: Optional[float] = None


@api.post("/portfolio/positions")
@limiter.limit("30/minute")
def add_portfolio_position(body: PortfolioPositionRequest, request: Request):
    ticker = _ticker_symbol(body.ticker)
    direction = body.direction.upper()
    if direction not in {"LONG", "SHORT"}:
        raise HTTPException(422, "direction must be LONG or SHORT")
    if body.shares <= 0:
        raise HTTPException(422, "shares must be greater than zero")

    entry = body.entry_price
    if entry is None or entry <= 0:
        detail = market.stock_detail(ticker)
        entry = detail.get("price") or market.ticker_price(ticker).get("price")
    if entry is None or entry <= 0:
        raise HTTPException(422, f"Could not resolve a current price for {ticker}")

    existing = []
    for pos in portfolio.get_positions():
        existing_ticker = str(pos.get("ticker", "")).strip().upper()
        if TICKER_RE.match(existing_ticker):
            existing.append({**pos, "ticker": existing_ticker})

    updated = False
    for pos in existing:
        if pos["ticker"] == ticker and str(pos.get("direction", "LONG")).upper() == direction:
            pos["shares"] = float(pos.get("shares", 0)) + float(body.shares)
            pos["entry_price"] = float(entry)
            if body.stop_loss is not None:
                pos["stop_loss"] = body.stop_loss
            updated = True
            break

    if not updated:
        new_pos = {
            "ticker": ticker,
            "shares": float(body.shares),
            "entry_price": float(entry),
            "direction": direction,
        }
        if body.stop_loss is not None:
            new_pos["stop_loss"] = body.stop_loss
        existing.append(new_pos)

    portfolio.update_positions(existing)
    return portfolio.snapshot()


# ── signals (all watchlist) ───────────────────────────────────────────
@api.get("/signals")
@limiter.limit("60/minute")
def all_signals(request: Request):
    settings  = _load_settings()
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)
    fund_map  = {
        t: {
            "pe_ratio":       (d := market.stock_detail(t)).get("pe_ratio"),
            "revenue_growth": d.get("revenue_growth"),
        }
        for t in watchlist
    }
    return engine.multi_signals(watchlist, fund_map)


ASSET_SIGNAL_UNIVERSES = {
    "macro": [
        ("SPY", "S&P 500 ETF", "Equity beta"),
        ("QQQ", "NASDAQ 100 ETF", "Growth beta"),
        ("GLD", "Gold", "Safe-haven hedge"),
        ("DX-Y.NYB", "US Dollar Index", "Dollar liquidity"),
        ("^VIX", "VIX", "Equity volatility"),
    ],
    "crypto": [
        ("BTC-USD", "Bitcoin", "Crypto beta"),
        ("ETH-USD", "Ethereum", "Smart-contract beta"),
        ("SOL-USD", "Solana", "High-beta crypto"),
    ],
    "fixed_income": [
        ("TLT", "20Y Treasury ETF", "Duration"),
        ("IEF", "7-10Y Treasury ETF", "Intermediate duration"),
        ("HYG", "High Yield Credit ETF", "Credit risk"),
        ("LQD", "Investment Grade Credit ETF", "IG credit"),
        ("BIL", "T-Bill ETF", "Cash proxy"),
    ],
}


@api.get("/cross-asset/signals")
@limiter.limit("60/minute")
def cross_asset_signals(request: Request, asset_class: str = "macro"):
    key = asset_class.lower().replace(" ", "_")
    if key not in ASSET_SIGNAL_UNIVERSES:
        raise HTTPException(422, f"Unsupported asset_class: {asset_class}")

    signals = []
    for ticker, name, thesis in ASSET_SIGNAL_UNIVERSES[key]:
        quote = market.ticker_price(ticker)
        pct = quote.get("pct")
        price = quote.get("price")
        if pct is None:
            direction = "HOLD"
            confidence = 50
        elif ticker == "^VIX":
            direction = "HEDGE" if pct > 0 else "RISK-ON"
            confidence = min(95, 55 + abs(float(pct)) * 4)
        elif pct > 0.25:
            direction = "LONG"
            confidence = min(95, 55 + abs(float(pct)) * 6)
        elif pct < -0.25:
            direction = "SHORT"
            confidence = min(95, 55 + abs(float(pct)) * 6)
        else:
            direction = "HOLD"
            confidence = 52
        signals.append({
            "ticker": ticker,
            "name": name,
            "asset_class": key,
            "direction": direction,
            "confidence": round(confidence, 1),
            "current_price": price,
            "change_pct": pct,
            "thesis": thesis,
        })
    return {"asset_class": key, "signals": signals}


# ── alerts ────────────────────────────────────────────────────────────
@api.get("/alerts")
@limiter.limit("60/minute")
def get_alerts(request: Request):
    import pickle
    snap     = portfolio.snapshot()
    settings = _load_settings()
    alerts: list = []

    var95 = snap.get("var_95", 0)
    total = snap.get("total_value", 0)
    if total and var95 / total > 0.02:
        alerts.append({
            "type": "RISK", "severity": "CRITICAL", "icon": "🔴",
            "title": f"Portfolio VaR breach — 95% VaR is ${var95:,.0f} ({var95/total*100:.1f}%)",
            "sub": "Portfolio · Risk Engine", "time": _now_str(),
        })

    for p in snap.get("positions", []):
        sl, cur = p.get("stop_loss"), p.get("current_price", 0)
        if sl and cur:
            gap = abs(cur - sl) / cur * 100
            if gap < 3:
                alerts.append({
                    "type": "POSITION", "severity": "WARNING", "icon": "⚠️",
                    "title": f"{p['ticker']} within {gap:.1f}% of stop-loss (${sl})",
                    "sub": f"Position Monitor · {p['ticker']}", "time": _now_str(),
                })
        if p.get("pnl_pct", 0) < -5:
            alerts.append({
                "type": "POSITION", "severity": "WARNING", "icon": "📉",
                "title": f"{p['ticker']} down {p['pnl_pct']:.1f}% — review thesis",
                "sub": f"Position Monitor · {p['ticker']}", "time": _now_str(),
            })

    for t in settings.get("watchlist", DEFAULT_WATCHLIST)[:4]:
        f = BASE / ".model_cache" / f"{t}.pkl"
        if f.exists():
            try:
                with open(f, "rb") as fh:
                    r = pickle.load(fh)
                if r.get("ensemble_accuracy", 100) < 80:
                    alerts.append({
                        "type": "MODEL", "severity": "INFO", "icon": "🤖",
                        "title": f"Model accuracy for {t} is {r['ensemble_accuracy']:.1f}% (< 80%)",
                        "sub": "Model Monitor · retraining queued", "time": _now_str(),
                    })
            except Exception:
                pass

    return {"alerts": alerts, "count": len(alerts)}


# ── model performance ─────────────────────────────────────────────────
@api.get("/models/performance")
@limiter.limit("60/minute")
def model_performance(request: Request):
    import pickle
    settings  = _load_settings()
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)
    models, xgb_accs, lstm_accs = [], [], []

    for t in watchlist:
        f = BASE / ".model_cache" / f"{t}.pkl"
        if not f.exists():
            continue
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            xgb_accs.append(r.get("xgb_accuracy", 0))
            lstm_accs.append(r.get("lstm_accuracy", 0))
            models.append({
                "ticker": t, "xgb_acc": r.get("xgb_accuracy"),
                "lstm_acc": r.get("lstm_accuracy"), "ens_acc": r.get("ensemble_accuracy"),
                "n_train": r.get("n_train"), "trained_at": r.get("trained_at"),
            })
        except Exception:
            pass

    avg_xgb  = round(sum(xgb_accs)  / len(xgb_accs),  1) if xgb_accs  else None
    avg_lstm = round(sum(lstm_accs) / len(lstm_accs), 1) if lstm_accs else None
    return {
        "models": models,
        "avg_xgb_acc":  avg_xgb,
        "avg_lstm_acc": avg_lstm,
        "avg_ens_acc":  round((avg_xgb + avg_lstm) / 2, 1) if avg_xgb and avg_lstm else None,
    }


# ── local model optimization: RL, distillation, quantization ──────────
@api.get("/models/optimization")
@limiter.limit("60/minute")
def models_optimization(request: Request):
    return optimization_status()


@api.post("/models/rl/update")
@limiter.limit("20/minute")
def models_rl_update(request: Request):
    return optimize_from_conviction_ledger(CONVICTIONS_FILE, RESOLUTIONS_FILE)


@api.get("/stock/{ticker}/optimization")
@limiter.limit("60/minute")
def stock_model_optimization(ticker: str, request: Request):
    ticker = _ticker_symbol(ticker)
    signal = _load_signal_payload(ticker)
    policy = ReinforcementPolicy()
    return {
        "ticker": ticker,
        "rl_policy": {
            "available": True,
            "q_values": policy.q_values(ticker),
            "updates": int(policy.state.get("updates", 0)),
            "source": "conviction_ledger_resolutions",
            "latest_signal_adjustment": signal.get("rl_policy", {}),
        },
        "distilled_student": signal.get(
            "distilled_student",
            {"available": False, "reason": "no cached signal artifact yet"},
        ),
        "quantized_student": signal.get(
            "quantized_student",
            {"available": False, "reason": "no cached signal artifact yet"},
        ),
    }


# ── conviction ledger ─────────────────────────────────────────────────
import re as _re


class ConvictionRequest(BaseModel):
    ticker:              str
    ml_direction:        str
    ml_probability:      float
    ml_model_version:    str
    sec_trend:           Optional[str]   = None
    sec_latest_revenue:  Optional[float] = None
    sec_quarters_used:   Optional[int]   = None
    sec_next_filing_due: Optional[str]   = None
    signal_view:         str
    conviction:          str
    source:              str             = "memo"
    entry_price:         float


@api.post("/convictions")
@limiter.limit("60/minute")
def post_conviction(body: ConvictionRequest, request: Request):
    if not _re.match(r"^[A-Z]{1,5}$", body.ticker.upper()):
        raise HTTPException(422, "Invalid ticker")
    conviction_id = write_conviction(
        ticker=body.ticker,
        ml_direction=body.ml_direction,
        ml_probability=body.ml_probability,
        ml_model_version=body.ml_model_version,
        sec_trend=body.sec_trend,
        sec_latest_revenue=body.sec_latest_revenue,
        sec_quarters_used=body.sec_quarters_used,
        sec_next_filing_due=body.sec_next_filing_due,
        signal_view=body.signal_view,
        conviction=body.conviction,
        source=body.source,
        entry_price=body.entry_price,
    )
    return {"conviction_id": conviction_id}


@api.get("/convictions/check")
@limiter.limit("60/minute")
def convictions_check(request: Request):
    return check_pending()


@api.get("/convictions/stats")
@limiter.limit("60/minute")
def convictions_stats(request: Request, ticker: Optional[str] = None):
    return get_accuracy_stats(ticker=ticker)


@api.get("/convictions/ledger")
@limiter.limit("60/minute")
def convictions_ledger(request: Request, page: int = 1, ticker: Optional[str] = None):
    return get_ledger(page=page, ticker=ticker)


# ── permanent graph memory ────────────────────────────────────────────
class MemoryRememberRequest(BaseModel):
    user_text: str
    assistant_text: str = ""
    source: str = "manual"
    metadata: Optional[dict] = None
    importance: float = 0.55


@api.get("/memory/status")
@limiter.limit("60/minute")
def memory_status(request: Request):
    return memory.status()


@api.post("/memory/remember")
@limiter.limit("30/minute")
def memory_remember(body: MemoryRememberRequest, request: Request):
    try:
        return memory.remember_interaction(
            user_text=body.user_text,
            assistant_text=body.assistant_text,
            source=body.source,
            metadata=body.metadata,
            importance=body.importance,
        )
    except GraphMemoryError as e:
        raise HTTPException(503, str(e))


@api.get("/memory/retrieve")
@limiter.limit("60/minute")
def memory_retrieve(q: str, request: Request, limit: int = 8):
    try:
        memories = memory.retrieve_context(q, limit=limit)
        return {"memories": memories, "context": memory.format_context(memories)}
    except GraphMemoryError as e:
        raise HTTPException(503, str(e))


# ── SEC ───────────────────────────────────────────────────────────────
@api.get("/sec/search")
@limiter.limit("60/minute")
def sec_search(q: str, request: Request, limit: int = 20):
    return sec.search_companies(q, limit)


@api.get("/sec/universe")
@limiter.limit("30/minute")
def sec_universe(
    request: Request,
    q: str = "",
    sic: str = "",
    industry: str = "",
    form: str = "",
    tickered_only: bool = True,
    limit: int = 100,
):
    return sec.company_universe(
        q=q,
        sic=sic,
        industry=industry,
        form=form,
        tickered_only=tickered_only,
        limit=limit,
    )


@api.get("/sec/industries")
@limiter.limit("30/minute")
def sec_industries(request: Request):
    return sec.industry_summary()


@api.get("/sec/stats")
@limiter.limit("60/minute")
def sec_stats(request: Request):
    return sec.summary_stats()


# ── news ──────────────────────────────────────────────────────────────
@api.get("/news")
@limiter.limit("60/minute")
def multi_news(request: Request):
    snap      = portfolio.snapshot()
    settings  = _load_settings()
    tickers   = [p["ticker"] for p in snap.get("positions", [])]
    all_tks   = list(dict.fromkeys(tickers + settings.get("watchlist", DEFAULT_WATCHLIST)))[:6]

    articles, ticker_sentiments = [], {}
    for t in all_tks:
        news = market.stock_news(t, limit=5)
        for n in news:
            n["primary_ticker"] = t
            articles.append(n)
        if news:
            ticker_sentiments[t] = round(sum(a["score"] for a in news) / len(news), 3)

    seen, unique = set(), []
    for a in articles:
        if a["title"] not in seen:
            seen.add(a["title"]); unique.append(a)
    unique.sort(key=lambda x: abs(x["score"]), reverse=True)

    avg = sum(ticker_sentiments.values()) / len(ticker_sentiments) if ticker_sentiments else 0
    return {
        "articles": unique[:15],
        "ticker_sentiments": ticker_sentiments,
        "market_sentiment": int(50 + avg * 50),
    }


# ── AI chat (streaming SSE) ───────────────────────────────────────────
def _allowed_ticker_context(ticker: str, snap: dict) -> set[str]:
    settings = _load_settings()
    tickers = {ticker.upper()}
    tickers.update(str(t).upper() for t in settings.get("watchlist", DEFAULT_WATCHLIST))
    tickers.update(str(p.get("ticker", "")).upper() for p in snap.get("positions", []))
    return {t for t in tickers if TICKER_RE.match(t)}


def _requires_memo_schema(message: str) -> bool:
    return bool(_re.search(r"\b(memo|investment thesis|recommendation|buy|sell|hold|trade plan)\b", message, _re.I))


def _guardrail_context(ticker: str, snap: dict, source_summary: str, require_memo_schema: bool = False) -> GuardrailContext:
    return GuardrailContext(
        ticker=ticker.upper(),
        allowed_tickers=_allowed_ticker_context(ticker, snap),
        source_summary=source_summary,
        require_memo_schema=require_memo_schema,
    )


def _chunk_text(text: str, size: int = 700):
    for idx in range(0, len(text), size):
        yield text[idx:idx + size]


def _select_chat_model(message: str, mode: str = "balanced") -> str:
    default_model = os.environ.get("RAPHI_CHAT_MODEL", "claude-opus-4-5")
    fast_model = os.environ.get("RAPHI_FAST_CHAT_MODEL", default_model)
    if mode == "fast":
        return fast_model
    simple = bool(_re.search(r"\b(price|quote|news|status|summary|what is)\b", message, _re.I))
    complex_ask = bool(_re.search(r"\b(memo|thesis|portfolio|risk|sec|filing|gnn|explain|recommend)\b", message, _re.I))
    return fast_model if simple and not complex_ask else default_model


CHAT_NON_TICKER_TERMS = {
    # System / infra terms
    "RAPHI", "SEC", "API", "MCP", "A2A", "GNN", "ML", "AI", "XBRL", "EDGAR",
    "JSON", "CSV", "PDF", "URL", "LLM", "NYSE", "NASDAQ", "HTTP", "SSE",
    # Finance / market abbreviations (must NOT be treated as tickers)
    "EPS", "FCF", "P/E", "PE", "AUM", "NAV", "VIX", "SPX", "YTD", "TTM",
    "RSI", "EV", "EBIT", "EBITDA", "GAAP", "OPEX", "CAPEX", "ROIC", "WACC",
    "IPO", "ETF", "ESG", "HFT", "OEM", "PMI", "CPI", "GDP", "DCF",
    "VAR", "VaR", "FOMC", "FED", "FX", "FY", "PCE",
    # Tech / product terms frequently in AI/chip analysis
    "CUDA", "GPU", "TAM", "FSD",
    # Company peers often mentioned by full abbreviation (not tickers in context)
    "TSMC", "OEM",
    # SEC form names
    "DEF", "HR", "FORM",
    # Trade signals
    "BUY", "SELL", "HOLD", "LONG", "SHORT",
    # Common English words that regex would otherwise match as 2-5 char uppercase
    "YES", "NO", "ME", "MY", "WE", "US", "IF", "OR", "IN", "IS", "IT", "AT",
    "WHAT", "WHO", "HOW", "WHY", "WHEN", "WHERE", "WHICH", "WRITE", "GIVE",
    "TELL", "KNOW", "ABOUT", "STOCK", "STOCKS", "MEMO", "RISK", "PRICE",
    "BEST", "NEXT", "DAY", "CAN", "YOU", "ARE", "THE", "THIS", "THAT",
    "WITH", "FROM", "FOR", "AND", "USING", "PERFORMANCE",
    "PULL", "PULLS", "PULLED", "RECENT", "MOST", "FULL", "DEEP", "DIVE",
    "DATA", "SHOW", "SHOWS", "LIKE", "WOULD", "COULD", "SHOULD",
    "FIND", "LARGE", "CAP", "TYPE", "LAST", "PAST", "OVER", "SAME", "REAL",
    "INTO", "JUST", "ALSO", "MORE", "LESS", "VERY", "HAVE", "BEEN", "THAN",
    "EACH", "BOTH", "SUCH", "DOES", "WANT", "NEED", "WILL", "HELD", "PLAN",
}


def _extract_ticker_from_text(text: str) -> str | None:
    for token in _re.findall(r"\b[A-Z]{2,5}\b", str(text or "").upper()):
        if token not in CHAT_NON_TICKER_TERMS and TICKER_RE.match(token):
            return token
    return None


def _resolve_chat_ticker(req: "ChatRequest") -> str:
    explicit = _extract_ticker_from_text(req.message)
    if explicit:
        return explicit
    user_history = [item for item in req.history[-10:] if item.get("role") == "user"]
    for item in reversed(user_history):
        found = _extract_ticker_from_text(item.get("content", ""))
        if found:
            return found
    for item in reversed(req.history[-6:]):
        if item.get("role") == "assistant":
            continue
        found = _extract_ticker_from_text(item.get("content", ""))
        if found:
            return found
    return _ticker_symbol(req.ticker)


def _is_identity_or_capability_query(message: str) -> bool:
    text = str(message or "").strip().lower()
    return bool(_re.search(
        r"\b(who are you|what are you|what can you do|what you can do|capabilit(?:y|ies)|how many stocks|total stocks|stock universe|coverage universe)\b",
        text,
    ))


def _chat_identity_response(message: str) -> str:
    text = str(message or "").lower()
    watchlist = _watchlist()
    if _re.search(r"\b(how many stocks|total stocks|stock universe|coverage universe|give me a number)\b", text):
        try:
            stats = sec.summary_stats()
            companies = int(stats.get("total_companies", 0) or 0)
            filings = int(stats.get("total_filings", 0) or 0)
            quarters = int(stats.get("total_quarters", 0) or 0)
        except Exception:
            companies = filings = quarters = 0
        return (
            "## RAPHI Coverage\n\n"
            f"- Local SEC company universe: **{companies:,} companies**\n"
            f"- Local SEC filing rows: **{filings:,} filings** across **{quarters} quarters**\n"
            f"- Default watchlist: **{len(watchlist)} tickers** ({', '.join(watchlist)})\n"
            "- Live stock lookup: available on demand for valid US ticker symbols"
        )

    return (
        "## RAPHI\n\n"
        "I am RAPHI, a local-first AI investment intelligence system.\n\n"
        "### What I Can Do\n"
        "- Search local SEC filings and XBRL financials\n"
        "- Cite SEC accession numbers, filing dates, and archive URLs\n"
        "- Pull live or cached market data, fundamentals, and news\n"
        "- Generate ML and GNN-backed signal context\n"
        "- Analyze portfolio P&L, VaR, Sharpe, alpha, and alerts\n"
        "- Produce exportable investment memos with provenance\n"
        "- Remember prior research context when memory is enabled\n\n"
        "Ask me for a ticker, SEC filing, model signal, portfolio risk view, or memo export."
    )


def _cached_system_blocks(stable_prompt: str, dynamic_prompt: str):
    if os.environ.get("RAPHI_PROMPT_CACHE", "1") == "0":
        return f"{stable_prompt}\n\n{dynamic_prompt}"
    return [
        {
            "type": "text",
            "text": stable_prompt,
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": dynamic_prompt},
    ]


def _load_signal_payload(ticker: str) -> dict:
    sig_cache = BASE / ".model_cache" / f"{ticker}.pkl"
    if not sig_cache.exists():
        return {"available": False}
    try:
        import pickle
        with open(sig_cache, "rb") as f:
            payload = pickle.load(f)
        return {"available": True, **payload}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _agentic_plan(message: str, ticker: str) -> list[dict]:
    text = str(message or "").lower()
    steps = [
        {"id": "understand", "label": f"Identify the user goal and primary ticker ({ticker})"},
        {"id": "market", "label": "Load price performance, fundamentals, provider source, and news URLs"},
    ]
    if _re.search(r"\b(sec|filing|10-k|10-q|fundamental|financial|citation|source|evidence|memo|investable|analyze|analysis)\b", text):
        steps.append({"id": "sec", "label": "Retrieve SEC filing metadata, XBRL metrics, accession numbers, and SEC links"})
    if _re.search(r"\b(ml|model|signal|gnn|graph|peer|neighbor|risk|investable|recommend|analyze|analysis)\b", text):
        steps.append({"id": "models", "label": "Check cached ML signal and GNN relationship layer"})
    steps.extend([
        {"id": "web", "label": "Fetch web citations for current narrative/source claims"},
        {"id": "portfolio", "label": "Compute portfolio exposure, P&L, VaR, and Sharpe context"},
        {"id": "memory", "label": "Retrieve episodic memory for prior ticker/thread context"},
        {"id": "synthesis", "label": "Synthesize answer with evidence, uncertainty, and trade/risk framing"},
        {"id": "reflection", "label": "Reflect on missing sources, hallucination risk, and guardrail repairs before finalizing"},
    ])
    return steps


def _source_checks(text: str, local_context: dict | None = None) -> dict:
    local_context = local_context or {}
    sec_ctx = local_context.get("sec", {}) or {}
    market_detail = (local_context.get("market", {}) or {}).get("detail", {}) or {}
    news = (local_context.get("market", {}) or {}).get("news", []) or []
    gnn_ctx = local_context.get("gnn", {}) or {}
    signal = local_context.get("ml_signal", {}) or {}
    body = str(text or "")
    web_ctx = local_context.get("web_citations", {}) or {}
    return {
        "sec_citation_available": bool(sec_ctx.get("recent_filings") or sec_ctx.get("financial_citations")),
        "sec_citation_used": bool(_re.search(r"https://www\.sec\.gov/Archives|accession\s+[0-9-]{10,}", body, _re.I)),
        "market_source_available": bool(market_detail.get("quote_url") or market_detail.get("source")),
        "market_source_used": "finance.yahoo.com/quote" in body or "Yahoo Finance" in body,
        "news_source_available": any(item.get("url") and item.get("url") != "#" for item in news),
        "news_source_used": bool(_re.search(r"https?://", body)) if news else True,
        "web_citations_available": bool(web_ctx.get("results")),
        "web_citations_used": any((item.get("url") or "") in body for item in web_ctx.get("results", [])[:5]),
        "ml_checked": bool(signal),
        "gnn_checked": bool(gnn_ctx.get("status") or gnn_ctx.get("signal")),
        "risk_framing_used": bool(_re.search(r"\b(risk|uncertain|uncertainty|downside|stop|invalidation|may|could)\b", body, _re.I)),
    }


def _reflection_label(checks: dict) -> str:
    missing = []
    if checks.get("sec_citation_available") and not checks.get("sec_citation_used"):
        missing.append("SEC citation link")
    if checks.get("market_source_available") and not checks.get("market_source_used"):
        missing.append("market source")
    if checks.get("news_source_available") and not checks.get("news_source_used"):
        missing.append("news source")
    if checks.get("web_citations_available") and not checks.get("web_citations_used"):
        missing.append("web citation link")
    if not checks.get("risk_framing_used"):
        missing.append("risk framing")
    if missing:
        return "Reflection found gaps: " + ", ".join(missing)
    return "Reflection passed: sources, memory, tools, and risk framing checked"


def _collect_local_agent_context(
    *,
    message: str,
    ticker: str,
    snap: dict,
    detail: dict,
    news: list,
    registration: dict | None = None,
) -> dict:
    """Collect real specialist-agent evidence for the browser fallback path."""
    lower = message.lower()
    want_sec_financials = bool(_re.search(
        r"\b(sec|filing|10-k|10-q|fundamental|financial|memo|thesis|recommend)\b",
        lower,
        _re.I,
    ))
    want_gnn = bool(_re.search(
        r"\b(gnn|graph|peer|neighbor|signal|memo|thesis|recommend|risk|explain|investable|investment|analyze|analysis|performance)\b",
        lower,
        _re.I,
    ))

    ctx: dict = {
        "ticker": ticker,
        "identity": _ticker_identity(ticker),
        "gnn_registration": registration or {},
        "market": {
            "detail": _apply_ticker_identity(ticker, detail),
            "news": news[:5],
        },
        "portfolio": snap,
        "ml_signal": _load_signal_payload(ticker),
        "sec": {
            "recent_filings": [],
            "financials": {},
            "financial_entries": [],
            "financial_citations": {},
        },
        "gnn": {
            "status": {},
            "signal": {},
        },
        "web_citations": {"provider": "none", "results": [], "count": 0},
    }

    try:
        ctx["sec"]["recent_filings"] = sec.ticker_filings(ticker, limit=8)
    except Exception as exc:
        ctx["sec"]["error"] = str(exc)

    if want_sec_financials:
        try:
            ctx["sec"]["financials"] = sec.company_financials(ticker)
        except Exception as exc:
            ctx["sec"]["financials_error"] = str(exc)
        try:
            ctx["sec"]["financial_entries"] = sec.company_financial_entries(ticker, limit_filings=4)[:16]
        except Exception as exc:
            ctx["sec"]["financial_entries_error"] = str(exc)
        try:
            ctx["sec"]["financial_citations"] = sec.company_financial_citations(ticker)
        except Exception as exc:
            ctx["sec"]["financial_citations_error"] = str(exc)

    try:
        ctx["gnn"]["status"] = gnn.status()
    except Exception as exc:
        ctx["gnn"]["status"] = {"error": str(exc)}

    if want_gnn:
        try:
            ctx["gnn"]["signal"] = gnn.predict(ticker, _gnn_universe(ticker))
        except Exception as exc:
            ctx["gnn"]["signal"] = {"error": str(exc), "ticker": ticker}

    # ── Live SEC EDGAR (real-time filings, 8-K events, insider Form 4s) ──────
    want_live_sec = bool(_re.search(
        r"\b(recent|latest|last\s+\d+\s+days?|this\s+(week|month|quarter)|just\s+filed"
        r"|8-?k|form\s*4|insider|ownership|material\s+event|risk\s+factor"
        r"|10-?[kq]|annual|quarterly|filing|changed|updated)\b",
        lower, _re.I,
    ))
    ctx["edgar_live"] = {"available": False, "filings": [], "events_8k": [], "insider_form4": []}
    try:
        live_summary = edgar_live.get_ticker_live_summary(ticker, days=60)
        ctx["edgar_live"] = {
            "available":        True,
            "cik":              live_summary.get("cik", ""),
            "filings":          live_summary.get("filings", []),
            "events_8k":        live_summary.get("material_events", []),
            "insider_form4":    live_summary.get("insider_transactions", []),
            "retrieved_at":     live_summary.get("retrieved_at", ""),
        }
    except Exception as exc:
        ctx["edgar_live"]["error"] = str(exc)

    # ── Firecrawl web narrative (earnings transcript, analyst coverage) ───────
    want_narrative = bool(_re.search(
        r"\b(transcript|earnings\s+call|analyst|price\s+target|coverage|recommend"
        r"|upgrade|downgrade|initiat|explain|narrative|outlook|guidance)\b",
        lower, _re.I,
    ))
    ctx["firecrawl"] = {"available": firecrawl_client.is_available()}
    if want_narrative and firecrawl_client.is_available():
        try:
            ctx["firecrawl"]["transcript"] = firecrawl_client.get_earnings_transcript(ticker)
        except Exception as exc:
            ctx["firecrawl"]["transcript_error"] = str(exc)
        try:
            ctx["firecrawl"]["analyst"] = firecrawl_client.get_analyst_coverage(ticker)
        except Exception as exc:
            ctx["firecrawl"]["analyst_error"] = str(exc)

    want_web_citations = bool(_re.search(
        r"\b(source|sources|citation|citations|cite|perplexity|web|news|latest|recent|analyst|transcript|article|why|how|evidence)\b",
        lower,
        _re.I,
    ))
    if want_web_citations:
        query = f"{ticker} {message} investment analysis source"
        try:
            ctx["web_citations"] = web_citations.search_citations(query, ticker=ticker, limit=5)
        except Exception as exc:
            ctx["web_citations"] = {"provider": "web_citations", "results": [], "count": 0, "error": str(exc)}

    return ctx


def _fmt_large_number(value) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    sign = "-" if num < 0 else ""
    num = abs(num)
    for suffix, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if num >= scale:
            return f"{sign}{num / scale:.2f}{suffix}"
    return f"{sign}{num:,.2f}"


def _format_local_agent_context(ctx: dict) -> str:
    ticker = ctx.get("ticker", "")
    identity = ctx.get("identity", {}) or {}
    registration = ctx.get("gnn_registration", {}) or {}
    detail = ctx.get("market", {}).get("detail", {}) or {}
    news = ctx.get("market", {}).get("news", []) or []
    sec_ctx = ctx.get("sec", {}) or {}
    gnn_ctx = ctx.get("gnn", {}) or {}
    signal = ctx.get("ml_signal", {}) or {}
    snap = ctx.get("portfolio", {}) or {}

    lines = [
        "Local specialist-agent evidence:",
        "",
        "@company-identity",
        f"- Current identity: {identity.get('current_name') or detail.get('name') or ticker}",
    ]
    if identity.get("former_name"):
        lines.append(f"- Former / legacy identity: {identity.get('former_name')}")
    if identity.get("identity_note"):
        lines.append(f"- Identity note: {identity.get('identity_note')}")
    if identity.get("strategy_note"):
        lines.append(f"- Strategy note: {identity.get('strategy_note')}")
    if detail.get("provider_name"):
        lines.append(f"- Provider returned legacy name: {detail.get('provider_name')}")
    lines.extend([
        "",
        "@gnn-registration",
        f"- Added to tracked universe this request: {registration.get('added_to_watchlist', False)}",
        f"- GNN universe includes: {', '.join(registration.get('universe', [])[:18]) or 'unavailable'}",
    ])
    if registration.get("gnn_error"):
        lines.append(f"- GNN training/register note: {registration.get('gnn_error')}")
    elif registration:
        reg_status = registration.get("gnn_status") or {}
        lines.append(
            f"- Registered in trained graph: {registration.get('gnn_added', False)} "
            f"({reg_status.get('graph_nodes', 'n/a')} nodes / {reg_status.get('graph_edges', 'n/a')} edges)"
        )
    lines.extend([
        "",
        "@market-analyst",
        f"- Price: ${detail.get('price', 'unavailable')} ({detail.get('pct', 'unavailable')}%)",
        f"- Market data source: {detail.get('source', 'Yahoo Finance via yfinance')} | quote URL {detail.get('quote_url', f'https://finance.yahoo.com/quote/{ticker}')}",
        f"- Valuation: P/E {detail.get('pe_ratio', 'unavailable')} | forward P/E {detail.get('forward_pe', 'unavailable')}",
        f"- Scale: market cap {_fmt_large_number(detail.get('market_cap'))} | revenue {_fmt_large_number(detail.get('revenue'))}",
        f"- Business: {(detail.get('short_summary') or 'unavailable')[:320]}",
        "",
        "@sec-researcher",
    ])

    filings = sec_ctx.get("recent_filings") or []
    if filings:
        for filing in filings[:5]:
            lines.append(
                f"- {filing.get('form', '?')} filed {filing.get('filed', '?')} "
                f"for period {filing.get('period', '?')} ({filing.get('quarter', '?')}); "
                f"accession {filing.get('accession', 'unavailable')}; "
                f"SEC URL {filing.get('sec_url', 'unavailable')}"
            )
    else:
        lines.append("- No local SEC filing metadata found for this ticker.")

    financials = sec_ctx.get("financials") or {}
    if financials:
        compact = [
            f"{metric}: {_fmt_large_number(value)}"
            for metric, value in list(financials.items())[:8]
        ]
        lines.append(f"- Latest XBRL metrics: {'; '.join(compact)}")
    citations = sec_ctx.get("financial_citations") or {}
    if citations:
        lines.append("- XBRL metric citations:")
        for metric, citation in list(citations.items())[:8]:
            lines.append(
                f"  - {metric}: {citation.get('form', '?')} "
                f"{citation.get('accession', 'unavailable')} filed {citation.get('filed', '?')}; "
                f"tag {citation.get('tag', 'n/a')}; value {_fmt_large_number(citation.get('value'))} "
                f"{citation.get('unit', '')}; SEC URL {citation.get('sec_url', 'unavailable')}"
            )
    if sec_ctx.get("financials_error") or sec_ctx.get("financial_entries_error"):
        lines.append(
            f"- SEC XBRL detail note: {sec_ctx.get('financials_error') or sec_ctx.get('financial_entries_error')}"
        )

    lines.extend([
        "",
        "@ml-signals",
    ])
    if signal.get("available"):
        lines.append(
            f"- Cached signal: {signal.get('direction', 'unknown')} "
            f"with {signal.get('confidence', 'unknown')} confidence"
        )
        if signal.get("features"):
            lines.append(f"- Feature context: {str(signal.get('features'))[:300]}")
    else:
        lines.append(f"- Cached signal unavailable: {signal.get('error', 'not computed')}")

    lines.extend([
        "",
        "@gnn-influence",
    ])
    status = gnn_ctx.get("status") or {}
    if status.get("trained"):
        lines.append(
            f"- Graph trained: {status.get('graph_nodes')} nodes, "
            f"{status.get('graph_edges')} edges, backend {status.get('backend')}"
        )
    else:
        lines.append(f"- Graph status: not trained ({status.get('error', 'stale or unavailable')})")
    gnn_signal = gnn_ctx.get("signal") or {}
    if gnn_signal and not gnn_signal.get("error"):
        lines.append(
            f"- GNN signal: {gnn_signal.get('direction')} "
            f"at {gnn_signal.get('confidence')}% confidence"
        )
        neighbors = gnn_signal.get("neighbors") or []
        if neighbors:
            joined = ", ".join(
                f"{n.get('ticker')} ({n.get('influence'):+.3f})"
                for n in neighbors[:6]
                if isinstance(n.get("influence"), (int, float))
            )
            lines.append(f"- Top graph neighbors: {joined or 'unavailable'}")
    elif gnn_signal.get("error"):
        lines.append(f"- GNN signal unavailable: {gnn_signal.get('error')}")

    lines.extend([
        "",
        "@portfolio-risk",
        f"- {_fmt_portfolio(snap)}",
        "",
        "@news-sentiment",
    ])
    if news:
        for item in news[:4]:
            lines.append(
                f"- {item.get('title', 'Untitled')} "
                f"({item.get('sentiment', 'neutral')}, score {item.get('score', 'n/a')}); "
                f"publisher {item.get('publisher', 'unknown')}; "
                f"source URL {item.get('url', 'unavailable')}"
            )
    else:
        lines.append("- No live news returned by the market data provider.")

    # ── Live SEC EDGAR (real-time) ────────────────────────────────────────────
    edgar_ctx = ctx.get("edgar_live") or {}
    lines.extend(["", "@edgar-live-filings"])
    if not edgar_ctx.get("available"):
        err = edgar_ctx.get("error", "CIK resolution failed or unavailable")
        lines.append(f"- EDGAR live data unavailable: {err}")
    else:
        lines.append(f"- CIK: {edgar_ctx.get('cik', 'unknown')} | retrieved: {edgar_ctx.get('retrieved_at', 'n/a')}")
        live_filings = edgar_ctx.get("filings") or []
        if live_filings:
            lines.append("- Recent 10-K/10-Q (live from EDGAR):")
            for f in live_filings[:4]:
                lines.append(
                    f"  - {f.get('form')} filed {f.get('filed')} | accession {f.get('accession')} | "
                    f"documents: {f.get('documents_url', 'unavailable')}"
                )
        else:
            lines.append("- No recent 10-K/10-Q found in last 60 days.")
        events = edgar_ctx.get("events_8k") or []
        if events:
            lines.append("- Material events (8-K, live):")
            for e in events[:4]:
                lines.append(
                    f"  - 8-K filed {e.get('filed')} | accession {e.get('accession')} | "
                    f"documents: {e.get('documents_url', 'unavailable')}"
                )
        else:
            lines.append("- No 8-K material events in last 60 days.")
        insiders = edgar_ctx.get("insider_form4") or []
        if insiders:
            lines.append("- Insider Form 4 transactions (live):")
            for i in insiders[:6]:
                lines.append(
                    f"  - Form 4 filed {i.get('filed')} | accession {i.get('accession')} | "
                    f"documents: {i.get('documents_url', 'unavailable')}"
                )
        else:
            lines.append("- No Form 4 insider transactions in last 60 days.")

    # ── Firecrawl narrative (earnings transcript, analyst coverage) ───────────
    fc_ctx = ctx.get("firecrawl") or {}
    if fc_ctx.get("available"):
        lines.extend(["", "@firecrawl-narrative"])
        transcript = fc_ctx.get("transcript") or {}
        if transcript.get("success") and transcript.get("markdown"):
            lines.append(f"- Earnings transcript source: {transcript.get('source', 'unavailable')}")
            lines.append(f"- Transcript excerpt: {transcript['markdown'][:1200]}")
        analyst = fc_ctx.get("analyst") or {}
        if analyst.get("success") and analyst.get("markdown"):
            lines.append(f"- Analyst coverage source: {analyst.get('source', 'unavailable')}")
            lines.append(f"- Analyst excerpt: {analyst['markdown'][:1000]}")

    web_ctx = ctx.get("web_citations") or {}
    lines.extend(["", "@web-citation-search"])
    if web_ctx.get("results"):
        lines.append(
            f"- Citation provider: {web_ctx.get('source_note') or web_ctx.get('provider')} | "
            f"query: {web_ctx.get('query')}"
        )
        for item in web_ctx.get("results", [])[:5]:
            lines.append(
                f"  - [{item.get('id')}] {item.get('title', 'Untitled')} | "
                f"{item.get('domain', item.get('display_link', ''))} | "
                f"{item.get('url', 'unavailable')} | "
                f"snippet: {item.get('snippet', '')[:260]}"
            )
    else:
        lines.append(f"- Web citation search unavailable: {web_ctx.get('error', 'not requested or no results')}")

    return "\n".join(lines)


async def _stream_direct_anthropic_chat(
    *,
    req: "ChatRequest",
    system: str,
    messages: list,
    api_key_anthropic: str,
    context: GuardrailContext,
):
    import anthropic

    stable_prompt = """You are RAPHI, an institutional AI investment intelligence platform.

Guardrails:
- Never fabricate numbers; cite tool/data context or say unavailable.
- Every SEC claim must include accession number, filing date, and SEC URL when available.
- Every news claim must include publisher/source URL when available.
- Every market quote or fundamental should name its provider/source URL when available.
- For web/current-events claims, cite numbered web citation results with title and URL, like [1].
- Investment views must include risk, uncertainty, sizing, and invalidation framing.
- Do not present investment outcomes as guaranteed.
- Use clean Markdown with concise bullets and readable sections."""

    client = anthropic.Anthropic(api_key=api_key_anthropic)
    collected: list[str] = []
    model = _select_chat_model(req.message, mode=req.response_mode)
    yield _sse("step", json.dumps({
        "id": "direct_llm",
        "label": f"Direct Anthropic fallback using {model}",
        "prompt_cache": os.environ.get("RAPHI_PROMPT_CACHE", "1") != "0",
    }))

    try:
        stream_kwargs = dict(
            model=model,
            max_tokens=1024,
            system=_cached_system_blocks(stable_prompt, system),
            messages=messages,
        )
        try:
            stream_ctx = client.messages.stream(**stream_kwargs)
        except TypeError:
            # Older SDKs may not accept cached system blocks; keep behavior working.
            stream_kwargs["system"] = f"{stable_prompt}\n\n{system}"
            stream_ctx = client.messages.stream(**stream_kwargs)

        with stream_ctx as stream:
            for text in stream.text_stream:
                collected.append(text)
    except Exception as e:
        yield _sse("error", str(e))
        return

    repaired, report = validate_and_repair_response("".join(collected), context)
    if report.repairs or report.warnings or report.missing_sections or report.unknown_tickers:
        yield _sse("step", json.dumps({
            "id": "guardrails",
            "label": "LLM guardrails validated and repaired the response",
            "repairs": report.repairs,
            "warnings": report.warnings,
            "missing_sections": report.missing_sections,
            "unknown_tickers": report.unknown_tickers,
        }))
    for chunk in _chunk_text(repaired):
        yield _sse("token", chunk)


class ChatRequest(BaseModel):
    message: str
    history: list = []
    ticker:  str  = "NVDA"
    thread_id: str = "default"
    response_mode: str = "balanced"
    agentic: bool = True


@api.post("/chat")
@limiter.limit("5/minute")
async def chat(req: ChatRequest, request: Request):
    import json, asyncio

    try:
        req.message = sanitize_user_input(req.message)
    except ValueError as exc:
        reject_message = str(exc)
        async def rejected():
            yield _sse("error", f"Request rejected: {reject_message}")
            yield _sse("done", "")
        return StreamingResponse(rejected(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if _is_identity_or_capability_query(req.message):
        async def identity_response():
            yield _sse("step", json.dumps({"id": "direct", "label": "Answered from RAPHI capability registry"}))
            yield _sse("token", _chat_identity_response(req.message))
            yield _sse("done", "")
        return StreamingResponse(identity_response(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    api_key_anthropic = _anthropic_api_key()
    snap   = portfolio.snapshot()
    ticker = _resolve_chat_ticker(req)
    registration = await asyncio.to_thread(_register_ticker_for_agentic_analysis, ticker)
    detail = _apply_ticker_identity(ticker, market.stock_detail(ticker))
    news   = market.stock_news(ticker, limit=3)
    identity = _ticker_identity(ticker)
    gnn_status = registration.get("gnn_status") or {}

    sig_text = ""
    sig_cache = BASE / ".model_cache" / f"{ticker}.pkl"
    if sig_cache.exists():
        import pickle
        try:
            with open(sig_cache, "rb") as f:
                sig = pickle.load(f)
            sig_text = f"RAPHI signal: {sig['direction']} ({sig['confidence']:.1f}% confidence)"
        except Exception:
            pass

    permanent_memory = _memory_context(f"{req.message} {ticker}", limit=6)
    execution_plan = _agentic_plan(req.message, ticker)
    system = f"""You are RAPHI, an AI investment intelligence platform.
Agentic execution plan:
{chr(10).join(f"- {step['label']}" for step in execution_plan)}
Portfolio: {_fmt_portfolio(snap)}
Company identity ({ticker}): {identity.get('current_name', ticker)}
Former identity: {identity.get('former_name', 'none')}
Identity note: {identity.get('identity_note', 'No special identity override.')}
Stock ({ticker}): ${detail.get('price','?')} P/E {detail.get('pe_ratio','?')}
Signal: {sig_text or 'not computed'}
GNN registration: added_to_watchlist={registration.get('added_to_watchlist')} registered_in_graph={registration.get('gnn_added')} nodes={gnn_status.get('graph_nodes','?')} edges={gnn_status.get('graph_edges','?')}
News: {chr(10).join(f"- {n['title']}" for n in news[:3])}
News sources: {chr(10).join(f"- {n.get('publisher','unknown')}: {n.get('url','unavailable')}" for n in news[:3])}
Permanent graph memory:
{permanent_memory or 'No relevant permanent memory found.'}
Use institutional language and quote specific numbers.

Presentation rules for the RAPHI web console:
- Write in clean Markdown with short headings and concise bullets.
- For investment memos, use exactly these sections: Recommendation, Key Evidence, GNN / Peer Influence, Risks, Trade Plan.
- In Key Evidence, include a Sources / Citations subsection with direct links.
- For non-SEC narrative claims, use the @web-citation-search results and cite them as [1], [2], etc.
- Do not say "SEC checked" unless you include at least one SEC accession/date/URL or explicitly say unavailable.
- Put the recommendation, confidence, and target/stop in the first 2 lines.
- Do not use ASCII diagrams, pipe-delimited relationship chains, raw graph art, or dense one-paragraph blocks.
- Keep each bullet under 24 words and use tables only for compact comparisons."""

    messages = [{"role": h["role"], "content": h["content"]} for h in req.history[-6:]]
    messages.append({"role": "user", "content": req.message})
    context = _guardrail_context(
        ticker,
        snap,
        source_summary=(
            f"market data, news, portfolio, SEC filings/XBRL, ML cache, "
            f"GNN graph influence, permanent memory, A2A/MCP tools for {ticker}"
            + (f"; known as {identity['current_name']}" if identity.get("current_name") else "")
        ),
        require_memo_schema=_requires_memo_schema(req.message),
    )

    if not api_key_anthropic:
        async def missing_key_response():
            yield _sse("step", json.dumps({
                "id": "memory",
                "label": "Permanent memory checked; Claude key is not configured",
            }))
            fallback = (
                "I can store and retrieve permanent memory, but full AI responses need "
                "ANTHROPIC_API_KEY in .env. Add that key and restart RAPHI to enable chat."
            )
            try:
                memory.remember_interaction(
                    user_text=req.message,
                    assistant_text=fallback,
                    source="chat",
                    metadata={"ticker": ticker, "anthropic_configured": False},
                    importance=0.64,
                )
            except Exception:
                pass
            repaired, report = validate_and_repair_response(fallback, context)
            if report.repairs or report.warnings:
                yield _sse("step", json.dumps({"id": "guardrails", "label": "Guardrails checked fallback response"}))
            yield _sse("token", repaired)
            yield _sse("done", "")

        return StreamingResponse(missing_key_response(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def generate():
        collected: list[str] = []
        used_agentic = False
        agentic_error = ""
        local_context: dict | None = None

        yield _sse("step", json.dumps({
            "id": "plan",
            "label": "Reasoning plan prepared: goal -> tools -> memory -> synthesis -> reflection",
            "plan": execution_plan,
        }))
        for step in execution_plan:
            await asyncio.sleep(0.01)
            yield _sse("step", json.dumps({
                "id": f"plan_{step['id']}",
                "label": f"Plan: {step['label']}",
            }))

        if req.agentic:
            yield _sse("step", json.dumps({"id": "orchestrator", "label": "Routing browser chat through A2A agent swarm"}))
            agent_prompt = (
                f"User request: {req.message}\n\n"
                f"Primary ticker: {ticker}\n"
                f"Company identity: {identity.get('current_name', ticker)}\n"
                f"Former identity: {identity.get('former_name', 'none')}\n"
                f"Identity note: {identity.get('identity_note', 'No special identity override.')}\n"
                f"Current portfolio context:\n{_fmt_portfolio(snap)}\n"
                f"Current stock context: price ${detail.get('price','?')}, P/E {detail.get('pe_ratio','?')}\n"
                f"Market source: {detail.get('source', 'Yahoo Finance via yfinance')} | {detail.get('quote_url', f'https://finance.yahoo.com/quote/{ticker}')}\n"
                f"Current signal context: {sig_text or 'not computed'}\n"
                f"GNN registration status: added_to_watchlist={registration.get('added_to_watchlist')}; "
                f"registered_in_graph={registration.get('gnn_added')}; "
                f"graph={gnn_status.get('graph_nodes','?')} nodes/{gnn_status.get('graph_edges','?')} edges\n"
                f"Recent news headlines and URLs:\n{chr(10).join(f'- {n['title']} | {n.get('publisher','unknown')} | {n.get('url','unavailable')}' for n in news[:3])}\n"
                f"Use MCP tools when more precise market, SEC, ML/GNN, portfolio, or memory data is needed. "
                f"Reply in clean Markdown with concise bullets, exact citation links, and explicit risk framing."
            )
            async for event in _agent.stream(agent_prompt, task_id=f"web:{req.thread_id}:{ticker}"):
                if event["event"] == "error":
                    agentic_error = event["data"]
                    break
                if event["event"] == "step":
                    yield _sse("step", event["data"])
                elif event["event"] == "token":
                    collected.append(event["data"])
                    yield _sse("token", event["data"])
            used_agentic = bool(collected)

        if not used_agentic:
            if agentic_error:
                yield _sse("step", json.dumps({
                    "id": "agentic_fallback",
                    "label": f"A2A SDK returned no text; running local multi-agent fallback ({agentic_error[:120]})",
                }))
            yield _sse("step", json.dumps({"id": "local_swarm", "label": "Local specialist agents collecting real evidence"}))
            local_context = await asyncio.to_thread(
                _collect_local_agent_context,
                message=req.message,
                ticker=ticker,
                snap=snap,
                detail=detail,
                news=news,
                registration=registration,
            )
            local_steps = [
                ("market",       f"@market-analyst loaded live price, fundamentals, and news for {ticker}"),
                ("sec",          "@sec-researcher loaded local SEC filing history and XBRL when requested"),
                ("signals",      "@ml-signals checked cached XGBoost/LSTM signal output"),
                ("gnn",          "@gnn-influence checked graph status and neighbor influence"),
                ("web",          "@web-citation-search fetched web citation results when requested"),
                ("portfolio",    "@portfolio-risk computed exposure, P&L, VaR, and Sharpe"),
                ("synthesize",   "@memo-synthesizer combining specialist outputs with guardrails"),
            ]
            for step_id, label in local_steps:
                yield _sse("step", json.dumps({"id": step_id, "label": label}))
                await asyncio.sleep(0.03)
            fallback_system = f"{system}\n\n{_format_local_agent_context(local_context)}"
            async for chunk in _stream_direct_anthropic_chat(
                req=req,
                system=fallback_system,
                messages=messages,
                api_key_anthropic=api_key_anthropic,
                context=context,
            ):
                if chunk.startswith("event: token"):
                    data = chunk.split("data: ", 1)[1].rstrip("\n")
                    try:
                        data = json.loads(data)
                    except Exception:
                        pass
                    collected.append(data)
                yield chunk

        checks = _source_checks("".join(collected), local_context)
        yield _sse("step", json.dumps({
            "id": "reflection",
            "label": _reflection_label(checks),
            "checks": checks,
        }))
        if _reflection_label(checks).startswith("Reflection found gaps"):
            yield _sse("token", (
                "\n\n### Reflection Check\n"
                "- Some requested evidence was not fully cited in the generated text above.\n"
                "- Use the Sources / Citations section or rerun with a narrower filing/news request for stricter provenance."
            ))

        _maybe_write_conviction(
            ticker=ticker,
            sig_cache_path=BASE / ".model_cache" / f"{ticker}.pkl",
            response_text="".join(collected),
        )
        try:
            memory.remember_interaction(
                user_text=req.message,
                assistant_text="".join(collected),
                source="chat",
                metadata={"ticker": ticker, "agentic": used_agentic, "thread_id": req.thread_id},
                importance=0.64,
            )
        except Exception:
            pass
        yield _sse("done", "")

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Investment memo (streaming) ───────────────────────────────────────
@api.post("/memo/{ticker}")
@limiter.limit("5/minute")
async def generate_memo(ticker: str, request: Request):
    import json, anthropic as _anth
    ticker = _ticker_symbol(ticker)
    api_key_anthropic = _anthropic_api_key()
    if not api_key_anthropic:
        raise HTTPException(422, "ANTHROPIC_API_KEY not set")

    detail = market.stock_detail(ticker)
    news   = market.stock_news(ticker, limit=5)
    snap   = portfolio.snapshot()
    sig    = {}
    sig_cache = BASE / ".model_cache" / f"{ticker}.pkl"
    if sig_cache.exists():
        import pickle
        try:
            with open(sig_cache, "rb") as f:
                sig = pickle.load(f)
        except Exception:
            pass

    permanent_memory = _memory_context(f"investment memo {ticker}", limit=6)
    stable_memo_prompt = """Write an institutional investment memo.

Required sections:
1. Recommendation — BUY/SELL/HOLD, confidence, price target, stop-loss, 90-day return
2. Key Evidence — SEC trend, market data, ML signal, news
3. GNN / Peer Influence — graph neighbors and what they imply
4. Risks — top 3 risks and invalidation triggers
5. Trade Plan — entry, target, stop-loss, sizing, horizon

Formatting rules:
- Clean Markdown only.
- No ASCII diagrams, pipe-delimited chains, raw graph art, or wall-of-text paragraphs.
- Use concise bullets; keep bullets under 24 words.
- Use one compact Markdown table only if it improves readability.
- Include one exact line near the top: Signal View: Positive, Signal View: Negative, or Signal View: Neutral.
- Lead with the recommendation and confidence in the first 2 lines.
- Never fabricate numbers; state unavailable when source data is missing."""
    dynamic_memo_context = f"""Ticker: {ticker}
Market data: {json.dumps(detail, indent=2)[:1500]}
ML signal: {json.dumps(sig, indent=2)[:800]}
News: {chr(10).join(f'- {n["title"]} ({n["sentiment"]})' for n in news[:5])}
Portfolio: {_fmt_portfolio(snap)}
Permanent graph memory:
{permanent_memory or 'No relevant permanent memory found.'}"""
    memo_context = _guardrail_context(
        ticker,
        snap,
        source_summary=f"market data, news, portfolio, ML cache, permanent memory for {ticker}",
        require_memo_schema=True,
    )

    async def generate():
        collected: list = []
        try:
            client = _anth.Anthropic(api_key=api_key_anthropic)
            kwargs = dict(
                model=_select_chat_model(f"investment memo {ticker}"),
                max_tokens=1500,
                system=_cached_system_blocks(stable_memo_prompt, dynamic_memo_context),
                messages=[{"role": "user", "content": f"Generate the memo for {ticker} now."}],
            )
            try:
                stream_ctx = client.messages.stream(**kwargs)
            except TypeError:
                kwargs["system"] = f"{stable_memo_prompt}\n\n{dynamic_memo_context}"
                stream_ctx = client.messages.stream(**kwargs)
            with stream_ctx as stream:
                for text in stream.text_stream:
                    collected.append(text)
        except Exception as e:
            yield _sse("error", str(e))
        final_text, report = validate_and_repair_response("".join(collected), memo_context)
        if report.repairs or report.warnings or report.missing_sections or report.unknown_tickers:
            yield _sse("step", json.dumps({
                "id": "guardrails",
                "label": "Memo schema and investment guardrails validated",
                "repairs": report.repairs,
                "warnings": report.warnings,
                "missing_sections": report.missing_sections,
            }))
        for chunk in _chunk_text(final_text):
            yield _sse("token", chunk)
        _maybe_write_conviction(
            ticker=ticker,
            sig_cache_path=BASE / ".model_cache" / f"{ticker}.pkl",
            response_text=final_text,
        )
        try:
            memory.remember_interaction(
                user_text=f"Generate investment memo for {ticker}",
                assistant_text=final_text,
                source="memo",
                metadata={"ticker": ticker},
                importance=0.66,
            )
        except Exception:
            pass
        yield _sse("done", "")

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _build_memo_export(ticker: str) -> dict:
    detail = market.stock_detail(ticker)
    news = market.stock_news(ticker, limit=5)
    snap = portfolio.snapshot()
    filings = sec.ticker_filings(ticker, limit=8)
    financials = sec.company_financials(ticker)
    financial_citations = sec.company_financial_citations(ticker)
    signal = _load_signal_payload(ticker)
    gnn_signal = {}
    try:
        gnn_signal = gnn.predict(ticker, _gnn_universe(ticker))
    except Exception as exc:
        gnn_signal = {"error": str(exc), "ticker": ticker}

    recommendation = signal.get("direction") if signal.get("available") else gnn_signal.get("direction", "HOLD")
    confidence = signal.get("confidence") if signal.get("available") else gnn_signal.get("confidence")
    return {
        "ticker": ticker,
        "exported_at": pd.Timestamp.now("UTC").isoformat(),
        "recommendation": recommendation or "HOLD",
        "confidence": confidence,
        "market": {
            "price": detail.get("price"),
            "change_pct": detail.get("pct"),
            "pe_ratio": detail.get("pe_ratio"),
            "market_cap": detail.get("market_cap"),
            "sector": detail.get("sector"),
            "industry": detail.get("industry"),
        },
        "sec": {
            "cik": sec.cik_for_ticker(ticker),
            "filings": filings,
            "financials": financials,
            "financial_citations": financial_citations,
        },
        "ml_signal": signal,
        "gnn": gnn_signal,
        "portfolio": snap,
        "news": news,
        "provenance": {
            "market": "yfinance live/cache wrapper",
            "sec": "SEC Financial Statement Data Sets and SEC EDGAR Archives",
            "ml": ".model_cache signal artifacts",
            "gnn": ".model_cache/gnn_state.pkl when trained",
            "portfolio": "portfolio.json plus live/cache prices",
        },
    }


def _memo_export_markdown(payload: dict) -> str:
    ticker = payload["ticker"]
    market_payload = payload.get("market", {})
    sec_payload = payload.get("sec", {})
    gnn_payload = payload.get("gnn", {})
    portfolio_payload = payload.get("portfolio", {})
    citations = sec_payload.get("financial_citations", {})
    citation_lines = []
    for metric, citation in list(citations.items())[:8]:
        citation_lines.append(
            f"- {metric}: {citation.get('form')} {citation.get('accession')} "
            f"filed {citation.get('filed')} ({citation.get('sec_url')})"
        )
    if not citation_lines:
        citation_lines.append("- No metric-level SEC citations available for this ticker.")

    filings = sec_payload.get("filings", [])
    filing_lines = [
        f"- {f.get('form')} {f.get('accession')} filed {f.get('filed')} ({f.get('sec_url')})"
        for f in filings[:5]
    ] or ["- No recent SEC filings found in the local dataset."]

    return "\n".join([
        f"# RAPHI Memo Export: {ticker}",
        "",
        f"Exported: {payload.get('exported_at')}",
        f"Recommendation: {payload.get('recommendation')} ({payload.get('confidence', 'n/a')} confidence)",
        "",
        "## Market",
        f"- Price: ${market_payload.get('price', 'n/a')}",
        f"- P/E: {market_payload.get('pe_ratio', 'n/a')}",
        f"- Sector / Industry: {market_payload.get('sector', 'n/a')} / {market_payload.get('industry', 'n/a')}",
        "",
        "## SEC Evidence",
        *filing_lines,
        "",
        "## Financial Citations",
        *citation_lines,
        "",
        "## GNN / Peer Influence",
        f"- Direction: {gnn_payload.get('direction', 'unavailable')}",
        f"- Confidence: {gnn_payload.get('confidence', 'unavailable')}",
        f"- Graph: {gnn_payload.get('graph_nodes', 'n/a')} nodes / {gnn_payload.get('graph_edges', 'n/a')} edges",
        "",
        "## Portfolio Risk",
        f"- Portfolio value: ${portfolio_payload.get('total_value', 0):,.0f}",
        f"- VaR 95%: ${portfolio_payload.get('var_95', 0):,.0f}",
        f"- Sharpe: {portfolio_payload.get('sharpe', 0):.2f}",
        "",
        "## Provenance",
        *(f"- {key}: {value}" for key, value in payload.get("provenance", {}).items()),
        "",
    ])


@api.get("/memo/{ticker}/export")
@limiter.limit("30/minute")
def export_memo(ticker: str, request: Request, format: str = "markdown"):
    ticker = _ticker_symbol(ticker)
    payload = _build_memo_export(ticker)
    if format.lower() in {"json", "data"}:
        return payload
    markdown = _memo_export_markdown(payload)
    filename = f"raphi-memo-{ticker.lower()}-{pd.Timestamp.now('UTC').strftime('%Y-%m-%d')}.md"
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── settings ──────────────────────────────────────────────────────────
@api.get("/settings")
@limiter.limit("60/minute")
def get_settings(request: Request):
    s = _load_settings()
    s.pop("anthropic_api_key", None)
    s["anthropic_api_key_set"] = bool(_anthropic_api_key())
    return s


class SettingsBody(BaseModel):
    watchlist:         list = []
    anthropic_api_key: str  = ""


@api.put("/settings")
@limiter.limit("60/minute")
def update_settings(body: SettingsBody, request: Request):
    s = _load_settings()
    if body.watchlist:
        s["watchlist"] = [t.upper() for t in body.watchlist]
    _save_settings(s)
    return {"ok": True}


# ── register data router ──────────────────────────────────────────────
app.include_router(api)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("RAPHI Unified Server  —  A2A primary + FastAPI sub-router")
    print("=" * 60)
    print(f"  A2A endpoint : http://127.0.0.1:9999/")
    print(f"  Agent card   : http://127.0.0.1:9999/.well-known/agent-card.json")
    print(f"  Dashboard    : http://127.0.0.1:9999/")
    print(f"  Data API     : http://127.0.0.1:9999/api/*")
    print(f"  Auth         : {'✓ RAPHI_API_KEY set' if api_key else '⚠ UNPROTECTED'}")
    print(f"  Sentry       : {'✓ enabled' if os.environ.get('SENTRY_DSN') else '○ not configured'}")
    print(f"  Agent swarm  : market-analyst · sec-researcher · ml-signals")
    print(f"                 portfolio-risk · memo-synthesizer")
    print("=" * 60)

    uvicorn.run(
        "backend.raphi_server:app",
        host="127.0.0.1",
        port=9999,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )
