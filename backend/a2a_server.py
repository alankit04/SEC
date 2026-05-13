"""
a2a_server.py  —  RAPHI A2A Server (port 9999)

Security fixes applied:
  C2  TokenAuth middleware — X-API-Key required (set RAPHI_API_KEY env var)
  C4  CORS locked to localhost origins only
  C4  Binds to 127.0.0.1 (not 0.0.0.0)
  Sentry  init_sentry() called at startup

Run:
    export RAPHI_API_KEY=your-secret-key
    export SENTRY_DSN=https://xxx@o0.ingest.sentry.io/yyy   # from raphi.sentry.io
    cd "/Users/alan/Desktop/SEC Data"
    .venv/bin/python -m backend.a2a_server
"""

import os
import sys
from pathlib import Path

import uvicorn
from starlette.middleware.cors import CORSMiddleware

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

sys.path.insert(0, str(Path(__file__).parent))

from market_data import MarketData
from sec_data import SECData
from ml_model import SignalEngine
from portfolio_manager import PortfolioManager
from a2a_executor_v2 import RaphiAgent, RaphiAgentExecutor  # Agent SDK + MCP
from security import TokenAuth, init_sentry                  # C2, Sentry

# ── Sentry: initialise before anything else ──────────────────────────
init_sentry()

# ── Singletons (same pattern as main.py) ─────────────────────────────
BASE = Path(__file__).parent.parent
market = MarketData()
sec    = SECData(BASE)
engine = SignalEngine()
portfolio = PortfolioManager()

# ── Agent Skills ─────────────────────────────────────────────────────
market_skill = AgentSkill(
    id="market_intel",
    name="Market Intelligence",
    description="Real-time market data, stock prices, charts, fundamentals, and news sentiment analysis.",
    tags=["market", "stocks", "prices", "news", "sentiment"],
    examples=[
        "What's the current price of NVDA?",
        "Give me a market overview",
        "Show me AAPL news and sentiment",
    ],
)

sec_skill = AgentSkill(
    id="sec_research",
    name="SEC Filings Research",
    description="Search SEC EDGAR database with 15 quarters of filings (2022-2025). Company financials from 10-K/10-Q XBRL data.",
    tags=["sec", "filings", "edgar", "financials", "10-K", "10-Q"],
    examples=[
        "Find SEC filings for Apple",
        "What are Tesla's latest financials?",
        "Search for semiconductor companies in SEC",
    ],
)

ml_skill = AgentSkill(
    id="ml_signals",
    name="ML Trading Signals",
    description="XGBoost + LSTM ensemble predictions with confidence scores and SHAP feature explainability.",
    tags=["ml", "signals", "predictions", "xgboost", "lstm"],
    examples=[
        "Generate a trading signal for MSFT",
        "What's the ML prediction for GOOGL?",
    ],
)

portfolio_skill = AgentSkill(
    id="portfolio_analysis",
    name="Portfolio Analysis",
    description="Portfolio snapshot with positions, P&L, VaR (95%/99%), Sharpe ratio, and alpha vs S&P 500.",
    tags=["portfolio", "risk", "positions", "var", "sharpe"],
    examples=[
        "Show my portfolio",
        "What's my portfolio risk?",
        "How are my positions performing?",
    ],
)

memo_skill = AgentSkill(
    id="investment_memo",
    name="Investment Memo",
    description="Comprehensive buy/sell/hold analysis combining market data, ML signals, SEC filings, and news sentiment.",
    tags=["memo", "analysis", "recommendation", "investment"],
    examples=[
        "Write an investment memo for GOOGL",
        "Should I buy or sell TSLA?",
        "Give me a full analysis of AMZN",
    ],
)

# ── Agent Card ────────────────────────────────────────────────────────
agent_card = AgentCard(
    name="RAPHI",
    description=(
        "AI-powered institutional financial analysis platform with real-time market data, "
        "SEC EDGAR filings (9,457+ companies), ML trading signals (XGBoost + LSTM), "
        "and portfolio risk management."
    ),
    url="http://localhost:9999/",
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[market_skill, sec_skill, ml_skill, portfolio_skill, memo_skill],
)

# ── Server Wiring ────────────────────────────────────────────────────
agent    = RaphiAgent()
executor = RaphiAgentExecutor(agent)
handler  = DefaultRequestHandler(
    agent_executor=executor,
    task_store=InMemoryTaskStore(),
)
server = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=handler,
)

if __name__ == "__main__":
    api_key        = os.environ.get("RAPHI_API_KEY", "")
    internal_token = os.environ.get("RAPHI_INTERNAL_TOKEN", "")
    if not api_key:
        print("⚠️  WARNING: RAPHI_API_KEY not set — server is UNPROTECTED")
        print("   Set it: export RAPHI_API_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')")
    else:
        print(f"✓  Auth: X-API-Key enabled ({len(api_key)} chars)")

    print("RAPHI A2A Server starting on http://127.0.0.1:9999")
    print("Agent card: http://127.0.0.1:9999/.well-known/agent.json")

    starlette_app = server.build()

    # C2 + H1/M3 — Token authentication (must be added before CORS)
    starlette_app.add_middleware(TokenAuth, api_key=api_key, internal_token=internal_token)

    # C4 — CORS locked to localhost only (not wildcard)
    starlette_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8000",   # dashboard origin
            "http://localhost:3000",
            "http://localhost:8080",
            "http://localhost:9999",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    )

    # C4 — Bind to 127.0.0.1 (loopback only, not network-visible)
    uvicorn.run(starlette_app, host="127.0.0.1", port=9999)
