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
  │    @ml-signals        → XGBoost+GB ensemble predictions    │
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
import math
import os
import re
import sys
import threading
import time
import uuid
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
from citation_index import CitationDocument, get_citation_index
import raphi_mcp_server as mcp_bridge
import edgar_live
import firecrawl_client
import web_citations
from eval_harness import EvalCase, evaluate_case, extract_citations
from eval_logger import build_run_record, log_eval_run, new_run_id
from release_gates import evaluate_release, load_run_records
from governance import assess_output, enqueue_review, list_reviews, decide_review
from provider_controls import CircuitBreaker, ProviderHealthRegistry
from user_data_store import (
    settings_path as user_settings_path,
    portfolio_path as user_portfolio_path,
    compliance_path as user_compliance_path,
    load_json as load_user_json,
    save_json as save_user_json,
)
from paths import COMPANY_TICKERS_FILE

init_sentry()  # no-op

# ── Paths ─────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
SETTINGS_FILE     = BASE / "settings.json"
DEFAULT_WATCHLIST: list[str] = []
TICKER_RE         = re.compile(r"^[A-Z]{1,5}$")
TICKER_ALLOWLIST_EXTRAS = {
    # Common ETFs/funds used in this app and not always present in SEC ticker maps.
    "SPY", "QQQ", "GLD", "TLT", "IEF", "HYG", "LQD", "BIL",
}
_TICKER_KNOWN_CACHE: dict[str, bool] = {}
_COMPANY_NAME_LOOKUP: dict[str, str] | None = None
_COMPANY_NAME_LOOKUP_LOCK = threading.Lock()

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
citations = get_citation_index()

from autonomy_controller import AutonomyController
autonomy = AutonomyController()
_monitor_jobs: dict[str, dict] = {}

# ── Lightweight TTL cache for market data (avoids repeated yfinance round-trips) ──
import threading as _threading

class _TTLCache:
    """Simple thread-safe TTL cache for expensive IO-bound calls."""
    def __init__(self, ttl_s: int = 60) -> None:
        self._lock = _threading.Lock()
        self._store: dict = {}
        self._ttl = ttl_s

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry and (time.time() - entry["ts"]) < self._ttl:
                return entry["value"]
            return None

    def set(self, key: str, value) -> None:
        with self._lock:
            self._store[key] = {"value": value, "ts": time.time()}

_market_cache = _TTLCache(ttl_s=45)

anthropic_breaker = CircuitBreaker(
    failure_threshold=int(os.environ.get("RAPHI_CB_ANTHROPIC_FAILURES", "3")),
    reset_timeout_s=int(os.environ.get("RAPHI_CB_ANTHROPIC_RESET_S", "90")),
)
firecrawl_breaker = CircuitBreaker(
    failure_threshold=int(os.environ.get("RAPHI_CB_FIRECRAWL_FAILURES", "3")),
    reset_timeout_s=int(os.environ.get("RAPHI_CB_FIRECRAWL_RESET_S", "120")),
)
edgar_breaker = CircuitBreaker(
    failure_threshold=int(os.environ.get("RAPHI_CB_EDGAR_FAILURES", "4")),
    reset_timeout_s=int(os.environ.get("RAPHI_CB_EDGAR_RESET_S", "60")),
)

provider_registry = ProviderHealthRegistry()
provider_registry.set_provider(
    "anthropic",
    configured=bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
    breaker=anthropic_breaker,
    meta={"model": "claude", "path": "anthropic"},
)
provider_registry.set_provider(
    "firecrawl",
    configured=bool(os.environ.get("FIRECRAWL_API_KEY", "").strip()),
    breaker=firecrawl_breaker,
    meta={"path": "firecrawl"},
)
provider_registry.set_provider(
    "edgar",
    configured=True,
    breaker=edgar_breaker,
    meta={"path": "sec-edgar-public"},
)

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
    url=os.environ.get("RAPHI_PUBLIC_URL", "http://localhost:9999/"),
    version="2.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=True),
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
                "XGBoost + GB ensemble ensemble, SHAP explainability, and GraphSAGE "
                "neighbor influence via @ml-signals subagent."
            ),
            tags=["ml", "signals", "xgboost", "gradient-boosting", "shap", "gnn", "graphsage"],
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
async def mcp_bridge_call(body: MCPBridgeRequest, request: Request):
    """Bridge HTTP /mcp requests to the stdio MCP tool implementations."""
    raw_tool = str(body.tool or "").strip()
    if not raw_tool:
        raise HTTPException(422, "tool is required")

    tool_name = raw_tool
    if tool_name.startswith("mcp__raphi__"):
        tool_name = tool_name[len("mcp__raphi__"):]

    arguments = dict(body.arguments or {})
    # Route-specific cache scope for user-sensitive tools.
    try:
        _, _, user_scope = _request_identity_scope(request)
        if "__user_scope" not in arguments:
            arguments["__user_scope"] = user_scope
    except HTTPException:
        # Keep legacy compatibility for internal calls that do not pass user headers.
        pass

    try:
        contents = await mcp_bridge.call_tool(tool_name, arguments)
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
    return _load_settings_for_scope("global")


def _load_settings_for_scope(user_scope: str) -> dict:
    path = user_settings_path(user_scope) if user_scope != "global" else SETTINGS_FILE
    if not path.exists():
        return {"watchlist": DEFAULT_WATCHLIST}

    import json
    with open(path) as f:
        settings = json.load(f)

    # Keep persisted state clean so plain English words never remain as tickers.
    original_watchlist = list(settings.get("watchlist", []))
    original_auto_added = list(settings.get("auto_added_tickers", []))
    settings["watchlist"] = _sanitize_ticker_list(original_watchlist)
    if "auto_added_tickers" in settings:
        settings["auto_added_tickers"] = _sanitize_ticker_list(original_auto_added)

    if (
        settings.get("watchlist") != original_watchlist
        or settings.get("auto_added_tickers", []) != original_auto_added
    ):
        _save_settings_for_scope(settings, user_scope)
    return settings


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def _ticker_symbol(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not TICKER_RE.match(ticker):
        raise HTTPException(422, f"Invalid ticker '{ticker}'. Use 1-5 uppercase letters.")
    if not _is_known_ticker(ticker):
        raise HTTPException(422, f"Invalid ticker '{ticker}'. Not found in supported ticker universe.")
    return ticker


def _is_known_ticker(ticker: str) -> bool:
    normalized = str(ticker or "").strip().upper()
    if not normalized:
        return False
    if normalized in _TICKER_KNOWN_CACHE:
        return _TICKER_KNOWN_CACHE[normalized]

    known = (
        normalized in TICKER_ALLOWLIST_EXTRAS
        or normalized in TICKER_IDENTITY_OVERRIDES
        or bool(sec.cik_for_ticker(normalized))
    )
    _TICKER_KNOWN_CACHE[normalized] = known
    return known


_COMPANY_SUFFIX_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies",
    "ltd", "limited", "llc", "plc", "ag", "sa", "nv", "spa", "holdings", "holding",
    "group", "the",
}


def _normalize_company_text(text: str) -> str:
    normalized = _re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    return _re.sub(r"\s+", " ", normalized)


def _simplify_company_name(name: str) -> str:
    tokens = _normalize_company_text(name).split()
    while tokens and tokens[-1] in _COMPANY_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def _build_company_name_lookup() -> dict[str, str]:
    global _COMPANY_NAME_LOOKUP

    if _COMPANY_NAME_LOOKUP is not None:
        return _COMPANY_NAME_LOOKUP

    with _COMPANY_NAME_LOOKUP_LOCK:
        if _COMPANY_NAME_LOOKUP is not None:
            return _COMPANY_NAME_LOOKUP

        by_name: dict[str, set[str]] = {}
        try:
            source_file = COMPANY_TICKERS_FILE if COMPANY_TICKERS_FILE.exists() else (BASE / "company_tickers.json")
            if source_file.exists():
                with open(source_file, encoding="utf-8") as f:
                    raw = json.load(f)
                rows = raw.values() if isinstance(raw, dict) else raw
                for row in rows:
                    ticker = str(row.get("ticker", "")).strip().upper()
                    title = str(row.get("title", "")).strip()
                    if not ticker or not title or not _is_known_ticker(ticker):
                        continue

                    candidates = {
                        _normalize_company_text(title),
                        _simplify_company_name(title),
                    }
                    for candidate in candidates:
                        if len(candidate) < 3:
                            continue
                        by_name.setdefault(candidate, set()).add(ticker)
        except Exception:
            by_name = {}

        for ticker, identity in TICKER_IDENTITY_OVERRIDES.items():
            for raw_name in (identity.get("current_name", ""), identity.get("former_name", "")):
                for candidate in (_normalize_company_text(raw_name), _simplify_company_name(raw_name)):
                    if len(candidate) >= 3:
                        by_name.setdefault(candidate, set()).add(ticker)

        resolved: dict[str, str] = {}
        for name, tickers in by_name.items():
            if len(tickers) == 1:
                resolved[name] = next(iter(tickers))

        _COMPANY_NAME_LOOKUP = resolved
        return _COMPANY_NAME_LOOKUP


def _extract_ticker_from_company_text(text: str) -> str | None:
    lookup = _build_company_name_lookup()
    normalized_text = _normalize_company_text(text)
    if not normalized_text:
        return None

    words = normalized_text.split()
    max_ngram = min(8, len(words))
    for width in range(max_ngram, 0, -1):
        for start in range(0, len(words) - width + 1):
            phrase = " ".join(words[start : start + width])
            ticker = lookup.get(phrase)
            if ticker and _is_known_ticker(ticker):
                return ticker
    return None


def _sanitize_ticker_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in values:
        try:
            ticker = _ticker_symbol(raw)
        except HTTPException:
            continue
        if ticker not in seen:
            cleaned.append(ticker)
            seen.add(ticker)
    return cleaned


def _watchlist(user_scope: str = "global") -> list[str]:
    raw_watchlist = _load_settings_for_scope(user_scope).get("watchlist", DEFAULT_WATCHLIST)
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
    return tickers



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


def _gnn_universe(*extra_tickers: str, requested: Optional[list[str]] = None, user_scope: str = "global") -> list[str]:
    universe: list[str] = []
    seen: set[str] = set()
    for raw in [*extra_tickers, *(requested or []), *_watchlist(user_scope)]:
        try:
            ticker = _ticker_symbol(raw)
        except HTTPException:
            continue
        if ticker not in seen:
            universe.append(ticker)
            seen.add(ticker)
    if len(universe) < 2:
        raise HTTPException(422, "GNN needs at least 2 valid tickers.")
    return universe


def _register_ticker_for_agentic_analysis(ticker: str, user_scope: str = "global") -> dict:
    """Persist a newly requested ticker and attempt to include it in the GNN graph."""
    ticker = _ticker_symbol(ticker)
    settings = _load_settings() if user_scope == "global" else _load_settings_for_scope(user_scope)
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
        if user_scope == "global":
            _save_settings(settings)
        else:
            _save_settings_for_scope(settings, user_scope)

    universe = _gnn_universe(ticker, requested=watchlist, user_scope=user_scope)
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


def _register_ticker_for_scope(ticker: str, user_scope: str) -> dict:
    try:
        return _register_ticker_for_agentic_analysis(ticker, user_scope)
    except TypeError:
        return _register_ticker_for_agentic_analysis(ticker)


def _save_settings(s: dict) -> None:
    _save_settings_for_scope(s, "global")


def _save_settings_for_scope(s: dict, user_scope: str) -> None:
    import json
    payload = dict(s)
    payload["watchlist"] = _sanitize_ticker_list(payload.get("watchlist", []))
    if "auto_added_tickers" in payload:
        payload["auto_added_tickers"] = _sanitize_ticker_list(payload.get("auto_added_tickers", []))
    path = user_settings_path(user_scope) if user_scope != "global" else SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _anthropic_api_key(user_scope: str = "global") -> str:
    """Return the Anthropic key from env first, then project settings."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    return str(_load_settings_for_scope(user_scope).get("anthropic_api_key", "")).strip()


def _anthropic_api_key_for_scope(user_scope: str) -> str:
    try:
        return _anthropic_api_key(user_scope)
    except TypeError:
        return _anthropic_api_key()


def _request_role(request: Request) -> str:
    auth_ctx = request.scope.get("raphi_auth") or {}
    role = str(auth_ctx.get("role") or request.headers.get("X-RAPHI-Role", "analyst")).strip().lower()
    return role if role in {"viewer", "analyst", "admin"} else "analyst"


_IDENTITY_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:@+-]{1,127}$")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _request_tenant_id(request: Request) -> str:
    auth_ctx = request.scope.get("raphi_auth") or {}
    tenant_id = str(auth_ctx.get("tenant") or request.headers.get("X-Tenant-Id", "local")).strip().lower()
    if not tenant_id:
        tenant_id = "local"
    if not _IDENTITY_RE.match(tenant_id):
        raise HTTPException(422, "Invalid X-Tenant-Id")
    return tenant_id[:128]


def _request_user_id(request: Request) -> str:
    require_identity = _env_bool("RAPHI_REQUIRE_IDENTITY", True)
    auth_ctx = request.scope.get("raphi_auth") or {}
    user_id = str(auth_ctx.get("sub") or request.headers.get("X-User-Id", "")).strip()
    if require_identity and not user_id:
        raise HTTPException(401, "X-User-Id header is required")
    if not user_id:
        user_id = "anonymous"
    if not _IDENTITY_RE.match(user_id):
        raise HTTPException(422, "Invalid X-User-Id")
    return user_id[:128]


def _request_identity_scope(request: Request) -> tuple[str, str, str]:
    tenant_id = _request_tenant_id(request)
    user_id = _request_user_id(request)
    return tenant_id, user_id, f"{tenant_id}:{user_id}"


def _portfolio_file_for_scope(user_scope: str) -> Path:
    return user_portfolio_path(user_scope)
def _portfolio_snapshot_for_scope(user_scope: str) -> dict:
    try:
        return portfolio.snapshot(portfolio_file=_portfolio_file_for_scope(user_scope))
    except TypeError:
        return portfolio.snapshot()


def _portfolio_get_positions_for_scope(user_scope: str) -> list:
    try:
        return portfolio.get_positions(portfolio_file=_portfolio_file_for_scope(user_scope))
    except TypeError:
        return portfolio.get_positions()


def _portfolio_update_positions_for_scope(positions: list, user_scope: str) -> None:
    try:
        portfolio.update_positions(positions, portfolio_file=_portfolio_file_for_scope(user_scope))
    except TypeError:
        portfolio.update_positions(positions)


def _require_human_review(request: Request) -> bool:
    header = str(request.headers.get("X-Human-Review", "")).strip().lower()
    if header in {"required", "true", "1", "yes"}:
        return True
    return os.environ.get("RAPHI_REVIEW_REQUIRED", "1") in {"1", "true", "yes"}


def _evidence_fail_closed_enabled() -> bool:
    return _env_bool("RAPHI_EVIDENCE_FAIL_CLOSED", False)


def _governance_block_mode_enabled() -> bool:
    return _env_bool("RAPHI_GOVERNANCE_BLOCK_MODE", False)


def _require_side_effect_approval_enabled() -> bool:
    return _env_bool("RAPHI_REQUIRE_SIDE_EFFECT_APPROVAL", True)


def _has_side_effect_approval(request: Request) -> bool:
    approved = str(request.headers.get("X-Action-Approval", "")).strip().lower()
    return approved in {"approved", "true", "1", "yes"}


def _enforce_side_effect_approval(request: Request, action: str) -> None:
    if not _require_side_effect_approval_enabled():
        return
    if _has_side_effect_approval(request):
        return
    raise HTTPException(
        409,
        (
            f"Action '{action}' requires explicit approval. "
            "Resubmit with header X-Action-Approval: approved"
        ),
    )


def _should_buffer_output() -> bool:
    # Buffer output when fail-closed controls are active so content is only emitted after policy checks.
    return _evidence_fail_closed_enabled() or _governance_block_mode_enabled()


def _load_compliance_for_scope(user_scope: str) -> dict:
    path = user_compliance_path(user_scope)
    defaults = {
        "regulated_advice_mode": False,
        "attested": False,
        "allow_recommendations": False,
        "client_profile": {
            "risk_tolerance": "moderate",
            "restricted_tickers": [],
        },
    }
    payload = load_user_json(path, defaults)
    profile = payload.get("client_profile") if isinstance(payload.get("client_profile"), dict) else {}
    payload["client_profile"] = {
        "risk_tolerance": str(profile.get("risk_tolerance", "moderate")).strip().lower() or "moderate",
        "restricted_tickers": _sanitize_ticker_list(profile.get("restricted_tickers", [])),
    }
    payload["regulated_advice_mode"] = bool(payload.get("regulated_advice_mode", False))
    payload["attested"] = bool(payload.get("attested", False))
    payload["allow_recommendations"] = bool(payload.get("allow_recommendations", False))
    return payload


def _save_compliance_for_scope(payload: dict, user_scope: str) -> None:
    clean = dict(payload or {})
    profile = clean.get("client_profile") if isinstance(clean.get("client_profile"), dict) else {}
    clean["client_profile"] = {
        "risk_tolerance": str(profile.get("risk_tolerance", "moderate")).strip().lower() or "moderate",
        "restricted_tickers": _sanitize_ticker_list(profile.get("restricted_tickers", [])),
    }
    clean["regulated_advice_mode"] = bool(clean.get("regulated_advice_mode", False))
    clean["attested"] = bool(clean.get("attested", False))
    clean["allow_recommendations"] = bool(clean.get("allow_recommendations", False))
    save_user_json(user_compliance_path(user_scope), clean)


def _contains_recommendation_intent(text: str) -> bool:
    return bool(re.search(r"\b(buy|sell|hold|recommendation|target\s*price|trade\s*plan|position\s*sizing)\b", str(text or ""), re.I))


def _apply_regulated_controls(*, user_scope: str, ticker: str, request_text: str, candidate_text: str | None = None) -> tuple[bool, dict]:
    policy = _load_compliance_for_scope(user_scope)
    findings: list[str] = []
    blocked = False

    if not policy.get("regulated_advice_mode"):
        return True, {"status": "not_regulated", "blocked": False, "findings": []}

    wants_advice = _contains_recommendation_intent(request_text) or _contains_recommendation_intent(candidate_text or "")
    if wants_advice and not policy.get("attested"):
        findings.append("Client attestation missing")
    if wants_advice and not policy.get("allow_recommendations"):
        findings.append("Recommendations are disabled for this client profile")

    restricted = set(policy.get("client_profile", {}).get("restricted_tickers", []))
    if ticker.upper() in restricted:
        findings.append(f"Ticker {ticker.upper()} is restricted by compliance profile")

    if findings:
        blocked = True
    return (not blocked), {
        "status": "pass" if not blocked else "blocked",
        "blocked": blocked,
        "regulated": True,
        "findings": findings,
    }


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


def _memory_context(query: str, user_scope: str, limit: int = 6) -> str:
    """Retrieve compact permanent memory context without blocking user flows."""
    try:
        memories = memory.retrieve_context(query, limit=limit, user_id=user_scope)
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
    provider_registry.set_provider(
        "anthropic",
        configured=bool(_anthropic_api_key()),
        breaker=anthropic_breaker,
        meta={"model": "claude", "path": "anthropic"},
    )
    return {
        "status": "ok",
        "server": "raphi-unified",
        "a2a": True,
        "providers": provider_registry.status(),
    }


@api.get("/providers/health")
@limiter.limit("60/minute")
def providers_health(request: Request):
    provider_registry.set_provider(
        "anthropic",
        configured=bool(_anthropic_api_key()),
        breaker=anthropic_breaker,
        meta={"model": "claude", "path": "anthropic"},
    )
    return provider_registry.status()


@api.get("/review/queue")
@limiter.limit("60/minute")
def review_queue(request: Request, status: str = ""):
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to access review queue")
    items = list_reviews(status=status or None)
    return {"items": items, "count": len(items)}


class ReviewDecisionBody(BaseModel):
    reviewer: str = ""
    note: str = ""


@api.post("/review/{run_id}/approve")
@limiter.limit("30/minute")
def approve_review(run_id: str, body: ReviewDecisionBody, request: Request):
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to approve reviews")
    item = decide_review(run_id, decision="approved", reviewer=body.reviewer or _request_user_id(request), note=body.note)
    if not item:
        raise HTTPException(404, "Review item not found")
    return item


@api.post("/review/{run_id}/reject")
@limiter.limit("30/minute")
def reject_review(run_id: str, body: ReviewDecisionBody, request: Request):
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to reject reviews")
    item = decide_review(run_id, decision="rejected", reviewer=body.reviewer or _request_user_id(request), note=body.note)
    if not item:
        raise HTTPException(404, "Review item not found")
    return item


@api.get("/release/gates")
@limiter.limit("20/minute")
def release_gates_status(request: Request, limit: int = 200):
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to view release gates")

    records = load_run_records(BASE / "eval_runs.jsonl")
    if limit > 0:
        records = records[-min(limit, 2000):]
    return evaluate_release(records)


def _load_run_record(run_id: str) -> dict:
    run_file = BASE / "data" / "eval_runs" / f"{run_id}.json"
    if not run_file.exists():
        raise HTTPException(404, "Run record not found")
    try:
        return json.loads(run_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"Failed to read run record: {exc}")


def _summarize_run_record(record: dict) -> dict:
    eval_result = record.get("eval_result") or {}
    review = record.get("review") if isinstance(record.get("review"), dict) else {}
    strict_gate = review.get("strict_quality_gate") if isinstance(review.get("strict_quality_gate"), dict) else {}
    return {
        "run_id": record.get("run_id"),
        "timestamp": record.get("timestamp"),
        "thread_id": record.get("thread_id"),
        "ticker": record.get("ticker"),
        "user_id": record.get("user_id"),
        "latency_ms": record.get("latency_ms"),
        "attempts": review.get("attempts"),
        "eval": {
            "overall_score": eval_result.get("overall_score"),
            "passed": eval_result.get("passed"),
        },
        "review": {
            "status": review.get("status"),
            "compliance_status": (review.get("compliance") or {}).get("status") if isinstance(review.get("compliance"), dict) else None,
            "strict_quality_gate": strict_gate,
        },
        "quality": {
            "citation_count": len(record.get("citations") or []),
            "tool_count": len(record.get("observed_tools") or []),
        },
    }


def _summarize_tool_trace(trace: list[dict]) -> list[dict]:
    summarized: list[dict] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        summarized.append(
            {
                "phase": item.get("phase"),
                "id": item.get("id"),
                "label": item.get("label"),
                "tool": item.get("tool"),
            }
        )
    return summarized


@api.get("/runs/{run_id}")
@limiter.limit("60/minute")
def run_record_detail(run_id: str, request: Request):
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to view run records")
    tenant_id, raw_user_id, _ = _request_identity_scope(request)
    record = _load_run_record(run_id)
    record_user = str(record.get("user_id") or "")
    caller = f"{tenant_id}:{raw_user_id}"
    if role != "admin" and record_user and record_user != caller:
        raise HTTPException(403, "Run record does not belong to caller")
    include_raw = str(request.query_params.get("include_raw", "")).strip().lower() in {"1", "true", "yes"}
    if include_raw:
        if role != "admin":
            raise HTTPException(403, "Only admin can access raw run records")
        return record
    return _summarize_run_record(record)


@api.get("/runs/{run_id}/tool-trace")
@limiter.limit("60/minute")
def run_record_tool_trace(run_id: str, request: Request):
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to view run tool trace")
    tenant_id, raw_user_id, _ = _request_identity_scope(request)
    caller = f"{tenant_id}:{raw_user_id}"
    record = _load_run_record(run_id)
    record_user = str(record.get("user_id") or "")
    if role != "admin" and record_user and record_user != caller:
        raise HTTPException(403, "Run record does not belong to caller")
    include_raw = str(request.query_params.get("include_raw", "")).strip().lower() in {"1", "true", "yes"}
    if include_raw and role != "admin":
        raise HTTPException(403, "Only admin can access raw tool trace")
    trace = record.get("tool_trace") or []
    payload_trace = trace if include_raw else _summarize_tool_trace(trace)
    return {
        "run_id": run_id,
        "count": len(payload_trace),
        "tool_trace": payload_trace,
    }


@api.get("/runs")
@limiter.limit("30/minute")
def run_records_list(request: Request, limit: int = 20):
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to view run records")
    tenant_id, raw_user_id, _ = _request_identity_scope(request)
    caller = f"{tenant_id}:{raw_user_id}"
    records = load_run_records(BASE / "eval_runs.jsonl")
    records = records[-min(max(int(limit), 1), 200):]
    if role != "admin":
        records = [r for r in records if str(r.get("user_id") or "") == caller]
    summaries = [
        {
            "run_id": r.get("run_id"),
            "timestamp": r.get("timestamp"),
            "ticker": r.get("ticker"),
            "thread_id": r.get("thread_id"),
            "user_id": r.get("user_id"),
            "latency_ms": r.get("latency_ms"),
            "eval": (r.get("eval_result") or {}).get("overall_score"),
            "passed": (r.get("eval_result") or {}).get("passed"),
            "review": (r.get("review") or {}).get("status"),
            "tool_count": len(r.get("observed_tools") or []),
            "citation_count": len(r.get("citations") or []),
        }
        for r in records
    ]
    return {"count": len(summaries), "records": summaries}


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
    return _json_safe(data)


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
    _enforce_side_effect_approval(request, "gnn_train")
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
    if not edgar_breaker.allow():
        raise HTTPException(503, "EDGAR provider temporarily unavailable (circuit open)")
    try:
        summary = edgar_live.get_ticker_live_summary(ticker, days=days)
        edgar_breaker.record_success()
        return summary
    except Exception as exc:
        edgar_breaker.record_failure(str(exc))
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
    if not edgar_breaker.allow():
        raise HTTPException(503, "EDGAR provider temporarily unavailable (circuit open)")
    try:
        results = edgar_live.search_filings_fulltext(
            query[:300],
            ticker=ticker.strip().upper() if ticker else None,
            forms=form_list,
            days=days,
            limit=limit,
        )
        edgar_breaker.record_success()
        return {"results": results, "count": len(results)}
    except Exception as exc:
        edgar_breaker.record_failure(str(exc))
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
    refresh_if_missing: bool = False


class CitationIndexRequest(BaseModel):
    ticker: str = ""
    source_type: str = "web"
    title: str = ""
    url: str = ""
    text: str = ""
    published_at: str = ""
    refresh_url: bool = False


@api.post("/firecrawl/scrape")
@limiter.limit("10/minute")
def firecrawl_scrape_route(body: FirecrawlScrapeRequest, request: Request):
    """Scrape a URL via Firecrawl and return clean markdown."""
    if not firecrawl_breaker.allow():
        raise HTTPException(503, "Firecrawl temporarily unavailable (circuit open)")
    if not body.url.startswith("https://"):
        raise HTTPException(422, "url must start with https://")
    max_chars = min(max(int(body.max_chars), 500), 15000)
    result = firecrawl_client.scrape_url(body.url, max_chars=max_chars)
    if not result.get("success"):
        firecrawl_breaker.record_failure(result.get("error", "scrape failed"))
        raise HTTPException(502, result.get("error", "scrape failed"))
    firecrawl_breaker.record_success()
    return result


@api.post("/firecrawl/search")
@limiter.limit("10/minute")
def firecrawl_search_route(body: FirecrawlSearchRequest, request: Request):
    """Search the web via Firecrawl and return scraped markdown from top results."""
    if not firecrawl_breaker.allow():
        raise HTTPException(503, "Firecrawl temporarily unavailable (circuit open)")
    limit = min(max(int(body.limit), 1), 10)
    results = firecrawl_client.search_web(body.query[:500], limit=limit, scrape_results=True)
    errors = [r for r in results if not r.get("success")]
    if errors and len(errors) == len(results):
        firecrawl_breaker.record_failure(errors[0].get("error", "search failed"))
        raise HTTPException(502, errors[0].get("error", "search failed"))
    firecrawl_breaker.record_success()
    return {"results": [r for r in results if r.get("success")], "count": len(results)}


@api.get("/citations/status")
@limiter.limit("60/minute")
def citations_status(request: Request):
    return citations.status()


@api.post("/citations/index")
@limiter.limit("20/minute")
def citations_index_route(body: CitationIndexRequest, request: Request):
    _enforce_side_effect_approval(request, "citations_index")
    ticker = _ticker_symbol(body.ticker) if body.ticker else ""
    text = body.text
    title = body.title
    if body.refresh_url:
        if not body.url.startswith("https://"):
            raise HTTPException(422, "url must start with https:// when refresh_url is true")
        scraped = firecrawl_client.scrape_url(body.url, max_chars=12000)
        if not scraped.get("success"):
            raise HTTPException(502, scraped.get("error", "scrape failed"))
        text = scraped.get("markdown", "")
        title = title or scraped.get("title", "")
    added = citations.add_document(CitationDocument(
        ticker=ticker,
        source_type=body.source_type,
        title=title,
        url=body.url,
        text=text,
        published_at=body.published_at,
    ))
    return {"ticker": ticker, **added}


@api.post("/citations/sec/{ticker}/index")
@limiter.limit("20/minute")
def citations_index_sec_route(ticker: str, request: Request, limit_filings: int = 8):
    _enforce_side_effect_approval(request, "citations_sec_index")
    ticker = _ticker_symbol(ticker)
    return citations.ingest_sec_ticker(sec, ticker, limit_filings=min(max(int(limit_filings), 1), 20))


@api.get("/citations/search")
@limiter.limit("60/minute")
def citations_search_route(
    request: Request,
    q: str,
    ticker: Optional[str] = None,
    limit: int = 5,
    refresh: bool = False,
):
    _, _, user_scope = _request_identity_scope(request)
    scoped_ticker = _ticker_symbol(ticker) if ticker else ""
    return citations.search_with_refresh(
        q,
        user_scope=user_scope,
        ticker=scoped_ticker,
        limit=min(max(int(limit), 1), 10),
        refresh_if_missing=bool(refresh),
    )


@api.post("/web/citations")
@limiter.limit("20/minute")
def web_citations_route(body: WebCitationRequest, request: Request):
    """Local-first citation search, with optional Firecrawl refresh."""
    _, _, user_scope = _request_identity_scope(request)
    ticker = _ticker_symbol(body.ticker) if body.ticker else ""
    limit = min(max(int(body.limit), 1), 10)
    result = web_citations.search_citations(
        body.query,
        user_scope=user_scope,
        ticker=ticker,
        limit=limit,
        refresh_if_missing=body.refresh_if_missing,
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
    _, _, user_scope = _request_identity_scope(request)
    return _portfolio_snapshot_for_scope(user_scope)


class Positions(BaseModel):
    positions: list


@api.put("/portfolio")
@limiter.limit("60/minute")
def update_portfolio(body: Positions, request: Request):
    _enforce_side_effect_approval(request, "portfolio_update")
    _, _, user_scope = _request_identity_scope(request)
    _portfolio_update_positions_for_scope(body.positions, user_scope)
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
    _enforce_side_effect_approval(request, "portfolio_add_position")
    _, _, user_scope = _request_identity_scope(request)
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
    for pos in _portfolio_get_positions_for_scope(user_scope):
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

    _portfolio_update_positions_for_scope(existing, user_scope)
    return _portfolio_snapshot_for_scope(user_scope)


# ── signals (all watchlist) ───────────────────────────────────────────
@api.get("/signals")
@limiter.limit("60/minute")
def all_signals(request: Request):
    _, _, user_scope = _request_identity_scope(request)
    settings  = _load_settings_for_scope(user_scope)
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
    _, _, user_scope = _request_identity_scope(request)
    snap     = _portfolio_snapshot_for_scope(user_scope)
    settings = _load_settings_for_scope(user_scope)
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
    _, _, user_scope = _request_identity_scope(request)
    settings  = _load_settings_for_scope(user_scope)
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)
    models, xgb_accs, gb2_accs = [], [], []

    for t in watchlist:
        f = BASE / ".model_cache" / f"{t}.pkl"
        if not f.exists():
            continue
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            xgb_accs.append(r.get("xgb_accuracy", 0))
            gb2_accs.append(r.get("gb2_accuracy", 0))
            models.append({
                "ticker": t, "xgb_acc": r.get("xgb_accuracy"),
                "gb2_acc": r.get("gb2_accuracy"), "ens_acc": r.get("ensemble_accuracy"),
                "n_train": r.get("n_train"), "trained_at": r.get("trained_at"),
            })
        except Exception:
            pass

    avg_xgb = round(sum(xgb_accs) / len(xgb_accs), 1) if xgb_accs else None
    avg_gb2 = round(sum(gb2_accs) / len(gb2_accs), 1) if gb2_accs else None
    return {
        "models":      models,
        "avg_xgb_acc": avg_xgb,
        "avg_gb2_acc": avg_gb2,
        "avg_ens_acc": round((avg_xgb + avg_gb2) / 2, 1) if avg_xgb and avg_gb2 else None,
    }


# ── local model optimization: RL, distillation, quantization ──────────
@api.get("/models/optimization")
@limiter.limit("60/minute")
def models_optimization(request: Request):
    return optimization_status()


@api.post("/models/rl/update")
@limiter.limit("20/minute")
def models_rl_update(request: Request):
    _enforce_side_effect_approval(request, "models_rl_update")
    return optimize_from_conviction_ledger(CONVICTIONS_FILE, RESOLUTIONS_FILE)


# ── autonomy controller ────────────────────────────────────────────────
def _append_retraining_record(record: dict) -> None:
    path = BASE / ".raphi_audit" / "retraining_log.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list = []
    if path.exists():
        try:
            records = json.loads(path.read_text())
        except Exception:
            pass
    records.append(record)
    path.write_text(json.dumps(records[-500:], indent=2))


@api.get("/autonomy/status")
@limiter.limit("60/minute")
def autonomy_status(request: Request):
    return autonomy.status()


class _BehaviorEvent(BaseModel):
    event_type: str
    metadata: dict = {}


@api.post("/autonomy/behavior")
@limiter.limit("60/minute")
def autonomy_behavior(body: _BehaviorEvent, request: Request):
    _, _, user_scope = _request_identity_scope(request)
    meta = {k: v for k, v in body.metadata.items() if k != "raw_text"}
    autonomy.learn_from_behavior(user_scope=user_scope, event_type=body.event_type, metadata=meta)
    return autonomy.behavior_profile(user_scope)


class _RetrainRequest(BaseModel):
    source: str = "watchlist"
    background: bool = False
    include_gnn: bool = True
    max_tickers: int = 10


@api.post("/models/retrain")
@limiter.limit("10/minute")
def models_retrain(body: _RetrainRequest, request: Request):
    _enforce_side_effect_approval(request, "models_retrain")
    _, _, user_scope = _request_identity_scope(request)

    if body.source == "behavior":
        profile = autonomy.behavior_profile(user_scope)
        tickers = profile.get("preferred_tickers", [])[:body.max_tickers]
    else:
        settings = _load_settings_for_scope(user_scope)
        tickers = [_ticker_symbol(t) for t in settings.get("watchlist", [])[:body.max_tickers]]

    if not tickers:
        return {"status": "no_tickers", "universe_source": body.source, "tickers": []}

    results = []
    for ticker in tickers:
        try:
            detail = market.stock_detail(ticker)
            funds = {
                "pe_ratio":       detail.get("pe_ratio")       if isinstance(detail, dict) else None,
                "revenue_growth": detail.get("revenue_growth") if isinstance(detail, dict) else None,
            }
            r = engine.force_retrain(ticker, funds)
            results.append(r)
        except Exception as exc:
            results.append({"ticker": ticker, "error": str(exc)})

    if body.include_gnn:
        try:
            gnn.ensure_trained(tickers, force=True)
        except Exception:
            pass

    record = {
        "triggered_at": _now_str(),
        "source": body.source,
        "tickers": tickers,
        "results": results,
        "rl_update": optimize_from_conviction_ledger(CONVICTIONS_FILE, RESOLUTIONS_FILE),
    }
    _append_retraining_record(record)
    return {"status": "trained", "universe_source": body.source, "tickers": tickers, "results": results}


class _MonitorStartRequest(BaseModel):
    ticker: str
    duration_s: int = 300
    poll_interval_s: int = 30
    intent: str = "monitor"


@api.post("/autonomy/monitor/start")
@limiter.limit("20/minute")
def autonomy_monitor_start(body: _MonitorStartRequest, request: Request):
    _enforce_side_effect_approval(request, "autonomy_monitor_start")
    job_id = str(uuid.uuid4())
    _monitor_jobs[job_id] = {
        "job_id":          job_id,
        "ticker":          _ticker_symbol(body.ticker),
        "duration_s":      body.duration_s,
        "poll_interval_s": body.poll_interval_s,
        "intent":          body.intent,
        "status":          "running",
        "started_at":      _now_str(),
        "stopped_at":      None,
    }
    return _monitor_jobs[job_id]


@api.get("/autonomy/monitor/jobs/{job_id}")
@limiter.limit("60/minute")
def autonomy_monitor_job(job_id: str, request: Request):
    job = _monitor_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Monitor job {job_id!r} not found.")
    return job


@api.post("/autonomy/monitor/jobs/{job_id}/stop")
@limiter.limit("20/minute")
def autonomy_monitor_stop(job_id: str, request: Request):
    _enforce_side_effect_approval(request, "autonomy_monitor_stop")
    job = _monitor_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Monitor job {job_id!r} not found.")
    job["status"] = "stopped"
    job["stopped_at"] = _now_str()
    return job


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
    _enforce_side_effect_approval(request, "post_conviction")
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
    _enforce_side_effect_approval(request, "memory_remember")
    _, _, user_scope = _request_identity_scope(request)
    try:
        return memory.remember_interaction(
            user_text=body.user_text,
            assistant_text=body.assistant_text,
            source=body.source,
            metadata=body.metadata,
            user_id=user_scope,
            importance=body.importance,
        )
    except GraphMemoryError as e:
        raise HTTPException(503, str(e))


@api.get("/memory/retrieve")
@limiter.limit("60/minute")
def memory_retrieve(q: str, request: Request, limit: int = 8):
    _, _, user_scope = _request_identity_scope(request)
    try:
        memories = memory.retrieve_context(q, limit=limit, user_id=user_scope)
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
    _, _, user_scope = _request_identity_scope(request)
    snap      = _portfolio_snapshot_for_scope(user_scope)
    settings  = _load_settings_for_scope(user_scope)
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
def _allowed_ticker_context(ticker: str, snap: dict, user_scope: str = "global") -> set[str]:
    settings = _load_settings_for_scope(user_scope)
    tickers = {ticker.upper()}
    tickers.update(str(t).upper() for t in settings.get("watchlist", DEFAULT_WATCHLIST))
    tickers.update(str(p.get("ticker", "")).upper() for p in snap.get("positions", []))
    return {t for t in tickers if TICKER_RE.match(t)}


def _requires_memo_schema(message: str) -> bool:
    return bool(_re.search(r"\b(memo|investment thesis|recommendation|buy|sell|hold|trade plan)\b", message, _re.I))


def _guardrail_context(ticker: str, snap: dict, source_summary: str, require_memo_schema: bool = False, user_scope: str = "global") -> GuardrailContext:
    return GuardrailContext(
        ticker=ticker.upper(),
        allowed_tickers=_allowed_ticker_context(ticker, snap, user_scope=user_scope),
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
    raw_text = str(text or "")
    for match in _re.finditer(r"\$?[A-Za-z]{2,5}", raw_text):
        raw_token = match.group(0)
        token = raw_token[1:] if raw_token.startswith("$") else raw_token

        if not raw_token.startswith("$") and token != token.upper():
            continue

        token = token.upper()
        if token in CHAT_NON_TICKER_TERMS or not TICKER_RE.match(token):
            continue
        if _is_known_ticker(token):
            return token
    return None


def _resolve_chat_ticker(req: "ChatRequest") -> str:
    explicit = _extract_ticker_from_text(req.message)
    if explicit:
        return explicit
    by_company = _extract_ticker_from_company_text(req.message)
    if by_company:
        return by_company
    user_history = [item for item in req.history[-10:] if item.get("role") == "user"]
    for item in reversed(user_history):
        content = item.get("content", "")
        found = _extract_ticker_from_text(content) or _extract_ticker_from_company_text(content)
        if found:
            return found
    for item in reversed(req.history[-6:]):
        if item.get("role") == "assistant":
            continue
        content = item.get("content", "")
        found = _extract_ticker_from_text(content) or _extract_ticker_from_company_text(content)
        if found:
            return found
    try:
        return _ticker_symbol(req.ticker)
    except HTTPException:
        return "NVDA"


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
    domains = [
        (m.group(1) or "").lower().removeprefix("www.")
        for m in _re.finditer(r"https?://([^/\s)\]]+)", body, _re.I)
    ]
    unique_domains = sorted({d for d in domains if d})
    non_core_domains = [d for d in unique_domains if d not in {"sec.gov", "finance.yahoo.com"}]
    source_diversity_required = bool(local_context.get("source_diversity_required", False))
    return {
        "sec_citation_available": bool(sec_ctx.get("recent_filings") or sec_ctx.get("financial_citations")),
        "sec_citation_used": bool(_re.search(r"https://www\.sec\.gov/Archives|accession\s+[0-9-]{10,}", body, _re.I)),
        "market_source_available": bool(market_detail.get("quote_url") or market_detail.get("source")),
        "market_source_used": "finance.yahoo.com/quote" in body or "Yahoo Finance" in body,
        "news_source_available": any(item.get("url") and item.get("url") != "#" for item in news),
        "news_source_used": bool(_re.search(r"https?://", body)) if news else True,
        "web_citations_available": bool(web_ctx.get("results")),
        "web_citations_used": any((item.get("url") or "") in body for item in web_ctx.get("results", [])[:5]),
        "source_diversity_required": source_diversity_required,
        "source_diversity_met": bool(non_core_domains),
        "link_domains": unique_domains,
        "non_core_domains": non_core_domains,
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
    if checks.get("source_diversity_required") and not checks.get("source_diversity_met"):
        missing.append("independent non-SEC/non-Yahoo source")
    if not checks.get("risk_framing_used"):
        missing.append("risk framing")
    if missing:
        return "Reflection found gaps: " + ", ".join(missing)
    return "Reflection passed: sources, memory, tools, and risk framing checked"


def _requires_source_diversity(message: str) -> bool:
    return bool(_re.search(
        r"\b(valuation|historical|peer|compare|cross[- ]?check|audit|verification|verify|risk|hedg|iv|implied volatility|z-?score|stretched|recommendation)\b",
        str(message or ""),
        _re.I,
    ))


_STEP_TOOL_MAP = {
    "market": "market",
    "sec": "sec",
    "signals": "ml",
    "models": "ml",
    "gnn": "gnn",
    "web": "citation",
    "portfolio": "portfolio",
    "memory": "memory",
}


def _extract_tool_name_from_step_payload(payload: dict) -> str | None:
    text = " ".join(
        str(payload.get(key, ""))
        for key in ("id", "label", "tool", "tool_name", "name")
    ).lower()
    if "mcp__raphi__" in text:
        start = text.index("mcp__raphi__")
        tail = text[start:].split()[0]
        return tail
    for step_id, family in _STEP_TOOL_MAP.items():
        if step_id in text:
            return family
    return None


def _apply_evidence_enforcement(text: str, checks: dict, required: bool) -> tuple[str, dict]:
    violations: list[str] = []
    if required:
        if checks.get("sec_citation_available") and not checks.get("sec_citation_used"):
            violations.append("SEC citations were available but not used")
        if checks.get("web_citations_available") and not checks.get("web_citations_used"):
            violations.append("Web citations were available but not used")
        if checks.get("market_source_available") and not checks.get("market_source_used"):
            violations.append("Market source attribution missing")
        if checks.get("source_diversity_required") and not checks.get("source_diversity_met"):
            violations.append("Independent non-SEC/non-Yahoo citation missing")

    if not violations:
        return text, {"required": required, "status": "pass", "violations": []}

    if _evidence_fail_closed_enabled():
        blocked_text = (
            "Policy block: evidence requirements were not satisfied for this response. "
            "Re-run with explicit source/citation retrieval enabled."
        )
        return blocked_text, {
            "required": required,
            "status": "blocked",
            "mode": "fail_closed",
            "violations": violations,
        }

    note = (
        "\n\n### Evidence Enforcement\n"
        "- Response downgraded due to missing required evidence links.\n"
        + "\n".join(f"- {item}" for item in violations)
    )
    return text.rstrip() + note, {
        "required": required,
        "status": "downgraded",
        "violations": violations,
    }


def _strict_quality_gate(
    *,
    message: str,
    candidate_response: str,
    checks: dict,
    eval_result: dict,
    require_evidence: bool,
) -> dict:
    metrics = eval_result.get("metrics") or {}
    citation_metric = metrics.get("citation_precision") or {}
    unsupported_metric = metrics.get("unsupported_claim_rate") or {}

    violations: list[str] = []
    citation_required = bool(require_evidence or checks.get("web_citations_available") or checks.get("sec_citation_available"))
    if citation_required and (not bool(citation_metric.get("passed", True))):
        violations.append("Citation precision gate failed")
    if citation_required and (not bool(unsupported_metric.get("passed", True))):
        violations.append("Unsupported claim gate failed")

    response = str(candidate_response or "")
    lower_response = response.lower()
    lower_message = str(message or "").lower()
    mentions_ml = bool(_re.search(r"\b(ml|model signal|signal model|xgboost|lstm)\b", lower_response + "\n" + lower_message))
    mentions_gnn = bool(_re.search(r"\b(gnn|graph signal|graph model|peer influence|neighbor)\b", lower_response + "\n" + lower_message))
    mentions_directional_view = bool(_re.search(r"\b(bullish|bearish|buy|sell|overweight|underweight|long|short)\b", lower_response))

    conflict_required = mentions_ml and mentions_gnn and mentions_directional_view
    if conflict_required:
        has_conflict_reasoning = bool(
            _re.search(
                r"\b(conflict|disagree|contradict|mixed signal|offset|lower(?:ed)? conviction|"
                r"position sizing|size (?:down|smaller)|uncertain|uncertainty|risk)\b",
                lower_response,
            )
        )
        if not has_conflict_reasoning:
            violations.append("ML/GNN conflict reasoning gate failed")

    return {
        "status": "pass" if not violations else "blocked",
        "passed": not violations,
        "citation_required": citation_required,
        "conflict_required": conflict_required,
        "violations": violations,
    }


def _collect_local_agent_context(
    *,
    message: str,
    ticker: str,
    snap: dict,
    detail: dict,
    news: list,
    registration: dict | None = None,
    force_tool_families: set[str] | None = None,
    force_refresh_citations: bool = False,
) -> dict:
    """Collect real specialist-agent evidence for the browser fallback path."""
    force_tool_families = set(force_tool_families or set())
    lower = message.lower()
    want_sec_financials = bool(_re.search(
        r"\b(sec|filing|10-k|10-q|fundamental|financial|memo|thesis|recommend)\b",
        lower,
        _re.I,
    )) or "sec" in force_tool_families
    want_gnn = bool(_re.search(
        r"\b(gnn|graph|peer|neighbor|signal|memo|thesis|recommend|risk|explain|investable|investment|analyze|analysis|performance)\b",
        lower,
        _re.I,
    )) or "gnn" in force_tool_families or "ml" in force_tool_families
    want_agentic = want_sec_financials or want_gnn

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

    if want_sec_financials or want_agentic:
        try:
            ctx["citation_index"] = citations.ingest_sec_ticker(sec, ticker, limit_filings=8)
        except Exception as exc:
            ctx["citation_index"] = {"error": str(exc)}

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
    )) or "citation" in force_tool_families
    if want_web_citations:
        query = f"{ticker} {message} investment analysis source"
        try:
            ctx["web_citations"] = web_citations.search_citations(
                query,
                ticker=ticker,
                limit=5,
                refresh_if_missing=True,
            )
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
        anthropic_breaker.record_success()
    except Exception as e:
        anthropic_breaker.record_failure(str(e))
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
    response_mode: str = "compact"
    agentic: bool = True


def _is_compact_mode(mode: str) -> bool:
    value = (mode or "").strip().lower()
    return value in {"compact", "fast", "lite"}


def _compact_text(text: str, limit: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 18)].rstrip() + " ...[compacted]"


def _compact_history(history: list, *, max_items: int, per_message_chars: int) -> list[dict]:
    compacted: list[dict] = []
    for item in history[-max_items:]:
        role = str(item.get("role", "user")) if isinstance(item, dict) else "user"
        content = _compact_text(item.get("content", "") if isinstance(item, dict) else str(item), per_message_chars)
        compacted.append({"role": role, "content": content})
    return compacted


@api.post("/chat")
@limiter.limit("10/minute")
async def chat(req: ChatRequest, request: Request):
    import json, asyncio

    user_role = _request_role(request)
    tenant_id, raw_user_id, user_scope = _request_identity_scope(request)

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

    api_key_anthropic = _anthropic_api_key_for_scope(user_scope)
    snap   = _portfolio_snapshot_for_scope(user_scope)
    ticker = _resolve_chat_ticker(req)
    compact_mode = _is_compact_mode(req.response_mode)
    memory_limit = 3 if compact_mode else 6

    def _load_sig_text(t: str) -> str:
        _sig_cache = BASE / ".model_cache" / f"{t}.pkl"
        if _sig_cache.exists():
            import pickle
            try:
                with open(_sig_cache, "rb") as f:
                    _sig = pickle.load(f)
                return f"RAPHI signal: {_sig['direction']} ({_sig['confidence']:.1f}% confidence)"
            except Exception:
                pass
        return ""

    # Parallel gather — runs I/O-bound pre-chat work concurrently instead of sequentially
    def _cached_detail(t: str) -> dict:
        cached = _market_cache.get(f"detail:{t}")
        if cached is not None:
            return cached
        result = market.stock_detail(t)
        _market_cache.set(f"detail:{t}", result)
        return result

    def _cached_news(t: str) -> list:
        cached = _market_cache.get(f"news:{t}")
        if cached is not None:
            return cached
        result = market.stock_news(t, limit=3)
        _market_cache.set(f"news:{t}", result)
        return result

    registration, raw_detail, news, permanent_memory, sig_text = await asyncio.gather(
        asyncio.to_thread(_register_ticker_for_scope, ticker, user_scope),
        asyncio.to_thread(_cached_detail, ticker),
        asyncio.to_thread(_cached_news, ticker),
        asyncio.to_thread(_memory_context, f"{req.message} {ticker}", user_scope, memory_limit),
        asyncio.to_thread(_load_sig_text, ticker),
    )
    permanent_memory = _compact_text(permanent_memory, 900 if compact_mode else 2400)
    detail = _apply_ticker_identity(ticker, raw_detail)
    identity = _ticker_identity(ticker)
    gnn_status = registration.get("gnn_status") or {}
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
- For valuation/peer/risk claims, include at least one independent citation beyond SEC and Yahoo when available.
- Do not say "SEC checked" unless you include at least one SEC accession/date/URL or explicitly say unavailable.
- Put the recommendation, confidence, and target/stop in the first 2 lines.
- Do not use ASCII diagrams, pipe-delimited relationship chains, raw graph art, or dense one-paragraph blocks.
- Keep each bullet under 24 words and use tables only for compact comparisons."""

    history_slice = _compact_history(
        req.history,
        max_items=4 if compact_mode else 6,
        per_message_chars=320 if compact_mode else 900,
    )
    messages = [{"role": h["role"], "content": h["content"]} for h in history_slice]
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
        user_scope=user_scope,
    )
    run_id = new_run_id()
    run_started = time.perf_counter()
    observed_tools: list[str] = []
    tool_trace: list[dict] = []

    if not api_key_anthropic or not anthropic_breaker.allow():
        async def missing_key_response():
            yield _sse("step", json.dumps({
                "run_id": run_id,
                "id": "memory",
                "label": "Permanent memory checked; full synthesis is temporarily unavailable",
            }))
            fallback = (
                "RAPHI can still store and retrieve durable memory, but the full research "
                "synthesis engine is temporarily unavailable in this runtime. Please try "
                "again after the AI service is reconnected."
            )
            try:
                memory.remember_interaction(
                    user_text=req.message,
                    assistant_text=fallback,
                    source="chat",
                    metadata={"ticker": ticker, "anthropic_configured": False},
                    user_id=user_scope,
                    importance=0.64,
                )
            except Exception:
                pass
            repaired, report = validate_and_repair_response(fallback, context)
            if report.repairs or report.warnings:
                yield _sse("step", json.dumps({"id": "guardrails", "label": "Guardrails checked fallback response"}))
            yield _sse("token", repaired)
            latency_ms = int((time.perf_counter() - run_started) * 1000)
            run_record = build_run_record(
                run_id=run_id,
                prompt=req.message,
                final_response=repaired,
                expected_tools=[],
                observed_tools=[],
                tool_trace=[],
                citations=[c.__dict__ for c in extract_citations(repaired)],
                ticker=ticker,
                allowed_tickers=list(_allowed_ticker_context(ticker, snap, user_scope=user_scope)),
                memo_schema=_requires_memo_schema(req.message),
                guardrail_repairs=report.repairs,
                guardrail_warnings=report.warnings,
                guardrail_missing_sections=report.missing_sections,
                guardrail_unknown_tickers=report.unknown_tickers,
                evidence_enforcement={"required": False, "status": "not_applicable", "violations": []},
                thread_id=req.thread_id,
                session_id=getattr(request, "session_id", None),
                user_id=f"{tenant_id}:{raw_user_id}",
                user_role=user_role,
                model_path="fallback/unavailable",
                provider_path="none",
                latency_ms=latency_ms,
                eval_result=None,
                governance={"status": "fallback", "findings": []},
                review={"status": "not_required"},
                quality={
                    "sec_citation_used": False,
                    "citation_count": 0,
                    "tool_trace_steps": 0,
                    "agentic_path_used": False,
                    "eval_passed": None,
                    "eval_score": None,
                },
            )
            log_eval_run(run_record)
            yield _sse("step", json.dumps({
                "id": "run_summary",
                "run_id": run_id,
                "label": "Run captured (fallback path)",
                "latency_ms": latency_ms,
            }))
            yield _sse("done", "")

        return StreamingResponse(missing_key_response(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def generate():
        max_retries = min(max(int(os.environ.get("RAPHI_CHAT_MAX_RETRIES", "0")), 0), 4)
        max_attempts = 1 + max_retries
        stream_tokens_immediately = (not _should_buffer_output()) and max_attempts == 1
        require_evidence = bool(_re.search(r"\b(cite|citation|source|evidence|sec|filing|news|prove|support)\b", req.message, _re.I))
        require_source_diversity = _requires_source_diversity(req.message)

        yield _sse("step", json.dumps({
            "id": "run_start",
            "run_id": run_id,
            "label": "Run started",
            "thread_id": req.thread_id,
            "ticker": ticker,
            "max_attempts": max_attempts,
        }))

        all_tool_trace: list[dict] = []
        all_observed_tools: list[str] = []
        attempt_failures: list[dict] = []
        retry_missing_tools: set[str] = set()
        retry_require_citations = require_source_diversity

        final_response = ""
        final_checks: dict = {}
        final_evidence_enforcement: dict = {"required": require_evidence, "status": "not_applicable", "violations": []}
        final_post_report = None
        final_eval_result: dict = {"passed": False, "overall_score": 0.0, "metrics": {}}
        final_citation_objects = []
        final_used_agentic = False
        final_strict_gate: dict = {"status": "pass", "passed": True, "violations": []}
        selected_attempt = 1

        expected_tool_ids = [step["id"] for step in execution_plan if step["id"] in _STEP_TOOL_MAP]

        for step in execution_plan:
            all_tool_trace.append({"phase": "plan", "id": step["id"], "label": step["label"]})
        yield _sse("step", json.dumps({
            "id": "plan",
            "label": "Reasoning plan prepared: goal -> tools -> memory -> synthesis -> reflection",
            "plan": execution_plan,
        }))
        for step in execution_plan:
            yield _sse("step", json.dumps({"id": f"plan_{step['id']}", "label": f"Plan: {step['label']}"}))

        for attempt in range(1, max_attempts + 1):
            selected_attempt = attempt
            attempt_collected: list[str] = []
            attempt_observed_tools: list[str] = []
            attempt_tool_trace: list[dict] = []
            local_context: dict | None = None
            agentic_error = ""
            used_agentic_attempt = False

            yield _sse("step", json.dumps({
                "id": "attempt_start",
                "label": f"Attempt {attempt}/{max_attempts} started",
                "attempt": attempt,
                "max_attempts": max_attempts,
            }))

            if attempt > 1:
                yield _sse("step", json.dumps({
                    "id": "replan",
                    "label": "Replan triggered after failed quality checks",
                    "attempt": attempt,
                    "missing_tools": sorted(retry_missing_tools),
                    "force_citations": retry_require_citations,
                    "previous_failures": attempt_failures[-1] if attempt_failures else {},
                }))

            use_agentic_attempt = req.agentic and attempt == 1
            if use_agentic_attempt:
                yield _sse("step", json.dumps({
                    "id": "orchestrator",
                    "label": "Routing browser chat through A2A agent swarm",
                    "attempt": attempt,
                }))
                _news_lines = "\n".join(
                    f"- {n['title']} ({n.get('url','unavailable')})"
                    for n in news[: (1 if compact_mode else 2)]
                )
                agent_prompt = (
                    f"{req.message}\n\n"
                    f"Ticker: {ticker} ({identity.get('current_name', ticker)})\n"
                    f"Price: ${detail.get('price','?')} | P/E: {detail.get('pe_ratio','?')} | Signal: {sig_text or 'not computed'}\n"
                    f"Portfolio: {_compact_text(_fmt_portfolio(snap), 180 if compact_mode else 520)}\n"
                    f"Headlines:\n{_news_lines}\n"
                    f"Use MCP tools for precise data. Reply in clean Markdown with concise bullets, citation links, and explicit risk framing."
                )
                try:
                    agent_stream = _agent.stream(
                        agent_prompt,
                        task_id=f"web:{req.thread_id}:{ticker}",
                        user_role=user_role,
                        compact=compact_mode,
                    )
                except TypeError:
                    agent_stream = _agent.stream(agent_prompt, task_id=f"web:{req.thread_id}:{ticker}")

                async for event in agent_stream:
                    if event["event"] == "error":
                        agentic_error = event["data"]
                        break
                    if event["event"] == "step":
                        payload = event["data"]
                        payload_obj = {}
                        if isinstance(payload, str):
                            try:
                                payload_obj = json.loads(payload)
                            except Exception:
                                payload_obj = {"label": payload}
                        elif isinstance(payload, dict):
                            payload_obj = payload
                        tool_name = _extract_tool_name_from_step_payload(payload_obj)
                        if tool_name:
                            attempt_observed_tools.append(tool_name)
                        attempt_tool_trace.append({
                            "phase": f"attempt_{attempt}_agentic",
                            "id": payload_obj.get("id", "step"),
                            "label": payload_obj.get("label", str(payload)[:200]),
                            "tool": tool_name,
                            "raw": _compact_text(json.dumps(payload_obj, sort_keys=True), 1200),
                        })
                        yield _sse("step", event["data"])
                    elif event["event"] == "token":
                        attempt_collected.append(event["data"])
                        if stream_tokens_immediately:
                            yield _sse("token", event["data"])
                used_agentic_attempt = bool(attempt_collected)

            if not used_agentic_attempt:
                if agentic_error:
                    yield _sse("step", json.dumps({
                        "id": "agentic_fallback",
                        "label": f"A2A SDK returned no text; running local multi-agent fallback ({agentic_error[:120]})",
                        "attempt": attempt,
                    }))
                force_tools = set(retry_missing_tools)
                if retry_require_citations:
                    force_tools.add("citation")
                yield _sse("step", json.dumps({
                    "id": "local_swarm",
                    "label": "Local specialist agents collecting real evidence",
                    "attempt": attempt,
                    "forced_tools": sorted(force_tools),
                }))
                local_context = await asyncio.to_thread(
                    _collect_local_agent_context,
                    message=req.message,
                    ticker=ticker,
                    snap=snap,
                    detail=detail,
                    news=news,
                    registration=registration,
                    force_tool_families=force_tools,
                    force_refresh_citations=retry_require_citations,
                )
                if local_context is None:
                    local_context = {}
                local_context["source_diversity_required"] = require_source_diversity
                local_steps = [
                    ("market", f"@market-analyst loaded live price, fundamentals, and news for {ticker}"),
                    ("sec", "@sec-researcher loaded local SEC filing history and XBRL when requested"),
                    ("signals", "@ml-signals checked cached XGBoost/GB ensemble signal output"),
                    ("gnn", "@gnn-influence checked graph status and neighbor influence"),
                    ("web", "@web-citation-search fetched web citation results when requested"),
                    ("portfolio", "@portfolio-risk computed exposure, P&L, VaR, and Sharpe"),
                    ("synthesize", "@memo-synthesizer combining specialist outputs with guardrails"),
                ]
                for step_id, label in local_steps:
                    mapped = _STEP_TOOL_MAP.get(step_id)
                    if mapped:
                        attempt_observed_tools.append(mapped)
                    attempt_tool_trace.append({
                        "phase": f"attempt_{attempt}_local_swarm",
                        "id": step_id,
                        "label": label,
                        "tool": mapped,
                    })
                    yield _sse("step", json.dumps({"id": step_id, "label": label, "attempt": attempt}))

                retry_note = ""
                if force_tools:
                    retry_note = "\n\nRetry policy: prioritize missing tool families -> " + ", ".join(sorted(force_tools))
                fallback_system = f"{system}{retry_note}\n\n{_format_local_agent_context(local_context)}"
                stream_had_error = False
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
                        attempt_collected.append(data)
                        if stream_tokens_immediately:
                            yield chunk
                    elif chunk.startswith("event: error"):
                        stream_had_error = True
                        yield chunk
                    else:
                        yield chunk
                if not stream_had_error:
                    anthropic_breaker.record_success()
                else:
                    anthropic_breaker.record_failure()

            candidate_response = "".join(attempt_collected)
            checks_context = dict(local_context or {})
            checks_context["source_diversity_required"] = require_source_diversity
            checks = _source_checks(candidate_response, checks_context)
            reflection = _reflection_label(checks)
            yield _sse("step", json.dumps({
                "id": "reflection",
                "label": reflection,
                "checks": checks,
                "attempt": attempt,
            }))
            if reflection.startswith("Reflection found gaps"):
                reflection_note = (
                    "\n\n### Reflection Check\n"
                    "- Some requested evidence was not fully cited in the generated text above.\n"
                    "- Use the Sources / Citations section or rerun with a narrower filing/news request for stricter provenance."
                )
                candidate_response += reflection_note
                if stream_tokens_immediately:
                    yield _sse("token", reflection_note)

            candidate_response, evidence_enforcement = _apply_evidence_enforcement(candidate_response, checks, require_evidence)
            if evidence_enforcement.get("status") == "downgraded" and stream_tokens_immediately:
                suffix = "\n\n### Evidence Enforcement\n- Response downgraded due to missing required evidence links."
                if suffix in candidate_response:
                    yield _sse("token", suffix)

            post_guarded, post_report = validate_and_repair_response(candidate_response, context)
            if post_guarded != candidate_response:
                delta = post_guarded[len(candidate_response):] if post_guarded.startswith(candidate_response) else "\n\n### Guardrail Update\n- Response adjusted after final validation."
                candidate_response = post_guarded
                if delta and stream_tokens_immediately:
                    yield _sse("token", delta)

            citation_objects = extract_citations(candidate_response)
            unique_attempt_tools = sorted({tool for tool in attempt_observed_tools if tool})
            eval_case = EvalCase(
                id=f"{run_id}-attempt-{attempt}",
                prompt=req.message,
                response=candidate_response,
                expected_tools=expected_tool_ids,
                observed_tools=unique_attempt_tools,
                citations=citation_objects,
                allowed_tickers=set(_allowed_ticker_context(ticker, snap, user_scope=user_scope)),
                ticker=ticker,
                require_memo_schema=_requires_memo_schema(req.message),
                require_citations=require_evidence,
            )
            eval_result = evaluate_case(eval_case).to_dict()
            metrics = eval_result.get("metrics", {})
            tool_details = ((metrics.get("tool_routing_accuracy") or {}).get("details") or {})
            missing_tools = [str(t) for t in (tool_details.get("missing_tools") or []) if str(t)]
            citation_passed = bool((metrics.get("citation_precision") or {}).get("passed", True))
            retry_require_citations = retry_require_citations or (not citation_passed) or evidence_enforcement.get("status") in {"downgraded", "blocked"}
            retry_missing_tools = set(missing_tools)

            bad_reflection = reflection.startswith("Reflection found gaps")
            bad_evidence = evidence_enforcement.get("status") in {"downgraded", "blocked"}
            eval_passed = bool(eval_result.get("passed"))

            strict_gate = _strict_quality_gate(
                message=req.message,
                candidate_response=candidate_response,
                checks=checks,
                eval_result=eval_result,
                require_evidence=require_evidence,
            )
            strict_failed = not strict_gate.get("passed", True)
            should_retry = attempt < max_attempts and (not eval_passed or bad_reflection or bad_evidence or strict_failed)

            all_tool_trace.extend(attempt_tool_trace)
            all_observed_tools.extend(unique_attempt_tools)

            final_response = candidate_response
            final_checks = checks
            final_evidence_enforcement = evidence_enforcement
            final_post_report = post_report
            final_eval_result = eval_result
            final_citation_objects = citation_objects
            final_used_agentic = used_agentic_attempt
            final_strict_gate = strict_gate

            if should_retry:
                failure = {
                    "attempt": attempt,
                    "eval_passed": eval_passed,
                    "overall_score": eval_result.get("overall_score"),
                    "missing_tools": sorted(retry_missing_tools),
                    "evidence_status": evidence_enforcement.get("status"),
                    "reflection": reflection,
                    "strict_quality_gate": strict_gate,
                }
                attempt_failures.append(failure)
                yield _sse("step", json.dumps({
                    "id": "retry_decision",
                    "label": "Quality gate failed; retrying with replanned tool strategy",
                    "attempt": attempt,
                    "next_attempt": attempt + 1,
                    "failure": failure,
                }))
                continue

            break

        if (
            not final_strict_gate.get("passed", True)
            and not str(final_response).startswith("Policy block: evidence requirements")
        ):
            final_response = (
                "Policy block: strict quality requirements were not satisfied. "
                + "; ".join(final_strict_gate.get("violations", []))
            )

        _maybe_write_conviction(
            ticker=ticker,
            sig_cache_path=BASE / ".model_cache" / f"{ticker}.pkl",
            response_text=final_response,
        )
        try:
            memory.remember_interaction(
                user_text=req.message,
                assistant_text=final_response,
                source="chat",
                metadata={
                    "ticker": ticker,
                    "agentic": final_used_agentic,
                    "thread_id": req.thread_id,
                    "attempts": selected_attempt,
                    "retries": max(0, selected_attempt - 1),
                },
                user_id=user_scope,
                importance=0.64,
            )
        except Exception:
            pass

        governance = assess_output(final_response, output_kind="chat")
        review_state = {"status": "not_required"}
        if governance.get("high_risk") and _require_human_review(request):
            queued = enqueue_review(
                run_id,
                kind="chat",
                user_id=f"{tenant_id}:{raw_user_id}",
                role=user_role,
                summary=final_response,
                assessment=governance,
            )
            review_state = {"status": queued.get("status", "pending")}
            yield _sse("step", json.dumps({
                "id": "human_review",
                "label": "Output queued for human review",
                "review_status": queued.get("status", "pending"),
            }))
            if _governance_block_mode_enabled():
                final_response = (
                    "Policy block: high-risk investment guidance is held for human review. "
                    "No actionable recommendation is released until approved."
                )

        compliance_pass, compliance_report = _apply_regulated_controls(
            user_scope=user_scope,
            ticker=ticker,
            request_text=req.message,
            candidate_text=final_response,
        )
        if not compliance_pass:
            final_response = (
                "Policy block: regulated advisory controls prevented release of this response. "
                + " ".join(compliance_report.get("findings", []))
            )

        if not stream_tokens_immediately:
            for chunk in _chunk_text(final_response):
                yield _sse("token", chunk)

        unique_observed_tools = sorted({tool for tool in all_observed_tools if tool})
        citation_records = [c.__dict__ for c in final_citation_objects]
        latency_ms = int((time.perf_counter() - run_started) * 1000)

        run_record = build_run_record(
            run_id=run_id,
            prompt=req.message,
            final_response=final_response,
            expected_tools=expected_tool_ids,
            observed_tools=unique_observed_tools,
            tool_trace=all_tool_trace,
            citations=citation_records,
            ticker=ticker,
            allowed_tickers=list(_allowed_ticker_context(ticker, snap, user_scope=user_scope)),
            memo_schema=_requires_memo_schema(req.message),
            guardrail_repairs=(final_post_report.repairs if final_post_report else []),
            guardrail_warnings=(final_post_report.warnings if final_post_report else []),
            guardrail_missing_sections=(final_post_report.missing_sections if final_post_report else []),
            guardrail_unknown_tickers=(final_post_report.unknown_tickers if final_post_report else []),
            evidence_enforcement=final_evidence_enforcement,
            thread_id=req.thread_id,
            session_id=getattr(request, "session_id", None),
            user_id=f"{tenant_id}:{raw_user_id}",
            user_role=user_role,
            model_path=_select_chat_model(req.message, mode=req.response_mode),
            provider_path="anthropic/claude-agent-sdk",
            latency_ms=latency_ms,
            eval_result=final_eval_result,
            governance=governance,
            review={
                **review_state,
                "compliance": compliance_report,
                "attempts": selected_attempt,
                "retry_failures": attempt_failures,
                "strict_quality_gate": final_strict_gate,
            },
            quality={
                # Measured from actual run artefacts — no invented proxies
                "sec_citation_used": bool(final_checks.get("sec_citation_used")),
                "citation_count": len(citation_records),
                "tool_trace_steps": len(all_tool_trace),
                "agentic_path_used": bool(final_used_agentic),
                "eval_passed": bool(final_eval_result.get("passed")),
                "eval_score": final_eval_result.get("overall_score"),
            },
        )
        log_eval_run(run_record)
        yield _sse("step", json.dumps({
            "id": "run_summary",
            "run_id": run_id,
            "label": "Run captured and scored",
            "latency_ms": latency_ms,
            "observed_tools": unique_observed_tools,
            "citation_count": len(citation_records),
            "eval_score": final_eval_result.get("overall_score"),
            "eval_passed": final_eval_result.get("passed"),
            "attempts_used": selected_attempt,
            "max_attempts": max_attempts,
            "retry_failures": attempt_failures,
        }))
        yield _sse("done", "")

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Investment memo (streaming) ───────────────────────────────────────
@api.post("/memo/{ticker}")
@limiter.limit("5/minute")
async def generate_memo(ticker: str, request: Request):
    import json, anthropic as _anth
    ticker = _ticker_symbol(ticker)
    user_role = _request_role(request)
    tenant_id, raw_user_id, user_scope = _request_identity_scope(request)
    if user_role == "viewer":
        raise HTTPException(403, "Viewer role cannot generate investment memos")
    api_key_anthropic = _anthropic_api_key_for_scope(user_scope)
    if not api_key_anthropic or not anthropic_breaker.allow():
        raise HTTPException(503, "AI synthesis service is temporarily unavailable")

    detail = market.stock_detail(ticker)
    news   = market.stock_news(ticker, limit=5)
    _, _, user_scope = _request_identity_scope(request)
    snap   = _portfolio_snapshot_for_scope(user_scope)
    recent_filings = []
    try:
        recent_filings = sec.ticker_filings(ticker, limit=6)
    except Exception:
        recent_filings = []
    sig    = {}
    sig_cache = BASE / ".model_cache" / f"{ticker}.pkl"
    if sig_cache.exists():
        import pickle
        try:
            with open(sig_cache, "rb") as f:
                sig = pickle.load(f)
        except Exception:
            pass

    permanent_memory = _memory_context(f"investment memo {ticker}", user_scope, limit=6)
    run_id = new_run_id()
    run_started = time.perf_counter()
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
SEC filings: {json.dumps(recent_filings[:3], indent=2)[:1200]}
Portfolio: {_fmt_portfolio(snap)}
Permanent graph memory:
{permanent_memory or 'No relevant permanent memory found.'}"""
    memo_context = _guardrail_context(
        ticker,
        snap,
        source_summary=f"market data, news, portfolio, ML cache, permanent memory for {ticker}",
        require_memo_schema=True,
        user_scope=user_scope,
    )

    async def generate():
        collected: list = []
        stream_tokens_immediately = not _should_buffer_output()
        yield _sse("step", json.dumps({
            "id": "run_start",
            "run_id": run_id,
            "label": "Memo run started",
            "ticker": ticker,
        }))
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
            anthropic_breaker.record_success()
        except Exception as e:
            anthropic_breaker.record_failure(str(e))
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
                user_id=user_scope,
                importance=0.66,
            )
        except Exception:
            pass

        checks = _source_checks(final_text, {
            "sec": {"recent_filings": recent_filings},
            "market": {"detail": detail, "news": news},
        })
        final_text, evidence_enforcement = _apply_evidence_enforcement(final_text, checks, required=True)
        if evidence_enforcement.get("status") == "downgraded" and stream_tokens_immediately:
            yield _sse("token", "\n\n### Evidence Enforcement\n- Response downgraded due to missing required evidence links.")

        citation_objects = extract_citations(final_text)
        citation_records = [c.__dict__ for c in citation_objects]
        latency_ms = int((time.perf_counter() - run_started) * 1000)
        observed_tools = ["market", "sec", "ml", "portfolio", "memory"]
        expected_tools = ["market", "sec", "ml", "portfolio", "memory"]

        eval_case = EvalCase(
            id=run_id,
            prompt=f"Generate investment memo for {ticker}",
            response=final_text,
            expected_tools=expected_tools,
            observed_tools=observed_tools,
            citations=citation_objects,
            allowed_tickers=set(_allowed_ticker_context(ticker, snap, user_scope=user_scope)),
            ticker=ticker,
            require_memo_schema=True,
            require_citations=True,
        )
        eval_result = evaluate_case(eval_case).to_dict()
        governance = assess_output(final_text, output_kind="memo")
        review_state = {"status": "not_required"}
        if governance.get("high_risk") and _require_human_review(request):
            queued = enqueue_review(
                run_id,
                kind="memo",
                user_id=f"{tenant_id}:{raw_user_id}",
                role=user_role,
                summary=final_text,
                assessment=governance,
            )
            review_state = {"status": queued.get("status", "pending")}
            yield _sse("step", json.dumps({
                "id": "human_review",
                "label": "Memo queued for human review",
                "review_status": queued.get("status", "pending"),
            }))
            if _governance_block_mode_enabled():
                final_text = (
                    "Policy block: high-risk memo is held for human review. "
                    "No actionable recommendation is released until approved."
                )

        compliance_pass, compliance_report = _apply_regulated_controls(
            user_scope=user_scope,
            ticker=ticker,
            request_text=f"Generate investment memo for {ticker}",
            candidate_text=final_text,
        )
        if not compliance_pass:
            final_text = (
                "Policy block: regulated advisory controls prevented release of this memo. "
                + " ".join(compliance_report.get("findings", []))
            )

        for chunk in _chunk_text(final_text):
            yield _sse("token", chunk)

        run_record = build_run_record(
            run_id=run_id,
            prompt=f"Generate investment memo for {ticker}",
            final_response=final_text,
            expected_tools=expected_tools,
            observed_tools=observed_tools,
            tool_trace=[
                {"phase": "memo", "id": "market", "label": "Loaded market/news context", "tool": "market", "raw": _compact_text(json.dumps({"detail": detail, "news": news[:3]}, default=str), 1200)},
                {"phase": "memo", "id": "sec", "label": "Loaded recent SEC filing context", "tool": "sec", "raw": _compact_text(json.dumps({"filings": recent_filings[:3]}, default=str), 1200)},
                {"phase": "memo", "id": "ml", "label": "Loaded cached model signal", "tool": "ml", "raw": _compact_text(json.dumps(sig, default=str), 1200)},
                {"phase": "memo", "id": "portfolio", "label": "Loaded portfolio risk context", "tool": "portfolio", "raw": _compact_text(json.dumps(snap, default=str), 1200)},
                {"phase": "memo", "id": "memory", "label": "Loaded durable memory context", "tool": "memory", "raw": _compact_text(permanent_memory, 1200)},
            ],
            citations=citation_records,
            ticker=ticker,
            allowed_tickers=list(_allowed_ticker_context(ticker, snap, user_scope=user_scope)),
            memo_schema=True,
            guardrail_repairs=report.repairs,
            guardrail_warnings=report.warnings,
            guardrail_missing_sections=report.missing_sections,
            guardrail_unknown_tickers=report.unknown_tickers,
            evidence_enforcement=evidence_enforcement,
            thread_id=None,
            session_id=getattr(request, "session_id", None),
            user_id=f"{tenant_id}:{raw_user_id}",
            user_role=user_role,
            model_path=_select_chat_model(f"investment memo {ticker}"),
            provider_path="anthropic/direct-chat",
            latency_ms=latency_ms,
            eval_result=eval_result,
            governance=governance,
            review={**review_state, "compliance": compliance_report},
            quality={
                "sec_citation_used": bool(checks.get("sec_citation_used")),
                "citation_count": len(citation_records),
                "tool_trace_steps": len(observed_tools),
                "agentic_path_used": False,
                "eval_passed": bool(eval_result.get("passed")) if eval_result else None,
                "eval_score": eval_result.get("overall_score") if eval_result else None,
            },
        )
        log_eval_run(run_record)
        yield _sse("step", json.dumps({
            "id": "run_summary",
            "run_id": run_id,
            "label": "Memo run captured and scored",
            "latency_ms": latency_ms,
            "observed_tools": observed_tools,
            "citation_count": len(citation_records),
            "eval_score": eval_result.get("overall_score"),
            "eval_passed": eval_result.get("passed"),
        }))
        yield _sse("done", "")

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _build_memo_export(ticker: str, user_scope: str = "global") -> dict:
    detail = market.stock_detail(ticker)
    news = market.stock_news(ticker, limit=5)
    snap = _portfolio_snapshot_for_scope(user_scope)
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
    _, _, user_scope = _request_identity_scope(request)
    payload = _build_memo_export(ticker, user_scope=user_scope)
    if format.lower() in {"json", "data"}:
        return payload
    markdown = _memo_export_markdown(payload)
    filename = f"raphi-memo-{ticker.lower()}-{pd.Timestamp.now('UTC').strftime('%Y-%m-%d')}.md"
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class ComplianceBody(BaseModel):
    regulated_advice_mode: bool = False
    attested: bool = False
    allow_recommendations: bool = False
    risk_tolerance: str = "moderate"
    restricted_tickers: list[str] = []


@api.get("/compliance")
@limiter.limit("60/minute")
def get_compliance(request: Request):
    _, _, user_scope = _request_identity_scope(request)
    return _load_compliance_for_scope(user_scope)


@api.put("/compliance")
@limiter.limit("30/minute")
def update_compliance(body: ComplianceBody, request: Request):
    _enforce_side_effect_approval(request, "update_compliance")
    _, _, user_scope = _request_identity_scope(request)
    payload = {
        "regulated_advice_mode": bool(body.regulated_advice_mode),
        "attested": bool(body.attested),
        "allow_recommendations": bool(body.allow_recommendations),
        "client_profile": {
            "risk_tolerance": str(body.risk_tolerance or "moderate").strip().lower() or "moderate",
            "restricted_tickers": _sanitize_ticker_list(body.restricted_tickers or []),
        },
    }
    _save_compliance_for_scope(payload, user_scope)
    return {"ok": True, "compliance": _load_compliance_for_scope(user_scope)}


class DeleteUserDataBody(BaseModel):
    confirm: str = ""


def _delete_user_run_records(caller: str) -> dict:
    removed_files = 0
    eval_runs_dir = BASE / "data" / "eval_runs"
    if eval_runs_dir.exists():
        for path in eval_runs_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("user_id") or "") == caller:
                try:
                    path.unlink(missing_ok=True)
                    removed_files += 1
                except Exception:
                    pass

    jsonl_path = BASE / "eval_runs.jsonl"
    removed_lines = 0
    if jsonl_path.exists():
        kept: list[str] = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                kept.append(line)
                continue
            if str(payload.get("user_id") or "") == caller:
                removed_lines += 1
            else:
                kept.append(line)
        jsonl_path.write_text(("\n".join(kept) + ("\n" if kept else "")), encoding="utf-8")

    return {"removed_run_files": removed_files, "removed_run_lines": removed_lines}


@api.get("/user-data/export")
@limiter.limit("20/minute")
def export_user_data(request: Request):
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to export user data")

    tenant_id, raw_user_id, user_scope = _request_identity_scope(request)
    caller = f"{tenant_id}:{raw_user_id}"
    settings_payload = _load_settings_for_scope(user_scope)
    settings_payload.pop("anthropic_api_key", None)

    portfolio_payload = _portfolio_get_positions_for_scope(user_scope)
    compliance_payload = _load_compliance_for_scope(user_scope)
    citation_payload = citations.export_user_data(user_scope, limit=1000)
    memory_payload = memory.export_user_data(user_scope, limit=1000)

    run_records: list[dict] = []
    eval_runs_dir = BASE / "data" / "eval_runs"
    if eval_runs_dir.exists():
        for path in sorted(eval_runs_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if role == "admin" or str(payload.get("user_id") or "") == caller:
                run_records.append(payload)

    return {
        "exported_at": pd.Timestamp.now("UTC").isoformat(),
        "scope": user_scope,
        "settings": settings_payload,
        "portfolio": portfolio_payload,
        "compliance": compliance_payload,
        "citations": citation_payload,
        "memory": memory_payload,
        "runs": run_records,
    }


@api.delete("/user-data")
@limiter.limit("10/minute")
def delete_user_data(body: DeleteUserDataBody, request: Request):
    _enforce_side_effect_approval(request, "delete_user_data")
    role = _request_role(request)
    if role not in {"analyst", "admin"}:
        raise HTTPException(403, "Role is not permitted to delete user data")
    if str(body.confirm or "").strip().upper() != "DELETE":
        raise HTTPException(422, "Deletion requires confirm='DELETE'")

    tenant_id, raw_user_id, user_scope = _request_identity_scope(request)
    caller = f"{tenant_id}:{raw_user_id}"

    settings_file = user_settings_path(user_scope)
    portfolio_file = _portfolio_file_for_scope(user_scope)
    compliance_file = user_compliance_path(user_scope)

    deleted_files: list[str] = []
    for path in (settings_file, portfolio_file, compliance_file):
        if path.exists():
            try:
                path.unlink(missing_ok=True)
                deleted_files.append(str(path))
            except Exception:
                pass

    citation_result = citations.delete_user_data(user_scope)
    memory_result = memory.delete_user_data(user_scope)
    runs_result = _delete_user_run_records(caller)

    return {
        "ok": True,
        "scope": user_scope,
        "deleted_files": deleted_files,
        "citations": citation_result,
        "memory": memory_result,
        "runs": runs_result,
    }


# ── settings ──────────────────────────────────────────────────────────
@api.get("/settings")
@limiter.limit("60/minute")
def get_settings(request: Request):
    _, _, user_scope = _request_identity_scope(request)
    s = _load_settings_for_scope(user_scope)
    s.pop("anthropic_api_key", None)
    s["anthropic_api_key_set"] = bool(_anthropic_api_key_for_scope(user_scope))
    return s


class SettingsBody(BaseModel):
    watchlist:         list = []
    anthropic_api_key: str  = ""


@api.put("/settings")
@limiter.limit("60/minute")
def update_settings(body: SettingsBody, request: Request):
    _enforce_side_effect_approval(request, "update_settings")
    _, _, user_scope = _request_identity_scope(request)
    s = _load_settings_for_scope(user_scope)
    if body.watchlist:
        s["watchlist"] = [t.upper() for t in body.watchlist]
    if body.anthropic_api_key:
        s["anthropic_api_key"] = body.anthropic_api_key.strip()
    _save_settings_for_scope(s, user_scope)
    return {"ok": True}


# ── Agentic query endpoint ────────────────────────────────────────────

class AgenticQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    history: list = Field(default_factory=list)
    user_context: dict = Field(default_factory=dict)
    tickers: list = Field(default_factory=list)
    universe: list = Field(default_factory=list)


@api.post("/agentic/query")
async def agentic_query(req: AgenticQueryRequest):
    from backend.security import sanitize_user_input
    from backend.input_guardrail import classify_input_bucket
    from raphi.orchestrators.agent_loop import run_agentic_query

    try:
        clean_query = sanitize_user_input(req.query)
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(exc))

    # Control Plane 1 — entry gate. A non-finance query is handled here and never
    # enters the agentic loop, so it cannot produce a Perceive→Plan→Execute state
    # dump. The natural-language response handler is a separate plane; this is an
    # interim placeholder reply.
    if classify_input_bucket(clean_query) == "general":
        import uuid

        from raphi.orchestrators.state import WorkflowState

        state = WorkflowState(
            run_id=str(uuid.uuid4()),
            user_query=clean_query,
            intent="general",
            risk_class="low",
            entities=[],
            tickers=[],
            final_answer=(
                "I'm RAPHI. I help with financial-markets research — ask me about a "
                "company, a filing, or your portfolio and I'll take it from there."
            ),
        )
        return state.to_dict()

    # Merge explicit tickers/universe into user_context so the planner can use them.
    user_ctx: dict = dict(req.user_context or {})
    if req.tickers:
        user_ctx["provided_tickers"] = [str(t).strip().upper() for t in req.tickers if t]
    if req.universe:
        user_ctx["universe"] = [str(t).strip().upper() for t in req.universe if t]

    state = run_agentic_query(
        clean_query,
        history=req.history or None,
        user_context=user_ctx or None,
    )
    return state.to_dict()


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
