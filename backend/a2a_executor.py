"""
a2a_executor.py  —  RAPHI A2A Agent Executor

RaphiAgent: wraps RAPHI backend modules behind Claude tool-use.
RaphiAgentExecutor: A2A protocol AgentExecutor implementation.
"""

import json
import os
import sys
from pathlib import Path

import anthropic

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

sys.path.insert(0, str(Path(__file__).parent))

from market_data import MarketData
from sec_data import SECData
from ml_model import SignalEngine
from portfolio_manager import PortfolioManager

# ── Anthropic API key (same pattern as main.py) ──────────────────────

SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"


def _get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key and SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            key = json.load(f).get("anthropic_api_key", "")
    return key


# ── Claude tool definitions ──────────────────────────────────────────

TOOLS = [
    {
        "name": "market_overview",
        "description": "Get real-time market overview with major indices (S&P 500, Nasdaq, VIX, 10Y yield, Gold, DXY).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "stock_detail",
        "description": "Get comprehensive stock data: price, chart, P/E, market cap, sector, fundamentals.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol (e.g. NVDA)"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "stock_news",
        "description": "Get recent news articles for a stock with sentiment analysis.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "sec_search",
        "description": "Search SEC EDGAR database for companies by name.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Company name to search for"}},
            "required": ["query"],
        },
    },
    {
        "name": "sec_filings",
        "description": "Get recent SEC filings (10-K, 10-Q) for a company by ticker.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "company_financials",
        "description": "Extract financial metrics (revenue, net income, EPS, assets, equity, cash) from SEC XBRL filings.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "ml_signal",
        "description": "Generate ML trading signal (LONG/SHORT/HOLD) using XGBoost + LSTM ensemble with SHAP explainability.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "portfolio_snapshot",
        "description": "Get current portfolio snapshot with positions, P&L, VaR (95%/99%), Sharpe ratio, and alpha.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """You are RAPHI (Real-time Agentic Platform for Human Investment Intelligence), an institutional-grade financial analysis agent.

You have access to tools for:
- Real-time market data and stock fundamentals
- SEC EDGAR filings (15 quarters, 9,457+ companies)
- ML trading signals (XGBoost + LSTM ensemble with SHAP explainability)
- Portfolio analysis with risk metrics (VaR, Sharpe, alpha)

When a user asks a question, use the appropriate tools to gather data, then provide a clear, concise, professional analysis. Use institutional investment language. Always cite specific data points from your tool results."""

MAX_TOOL_ROUNDS = 5


class RaphiAgent:
    """Wraps RAPHI backend modules behind Claude tool-use."""

    def __init__(
        self,
        market: MarketData,
        sec: SECData,
        engine: SignalEngine,
        portfolio: PortfolioManager,
    ):
        self.market = market
        self.sec = sec
        self.engine = engine
        self.portfolio = portfolio

    def _dispatch_tool(self, name: str, params: dict):
        """Route a tool call to the appropriate backend method."""
        ticker = params.get("ticker", "").upper()

        if name == "market_overview":
            return self.market.market_overview()
        elif name == "stock_detail":
            result = self.market.stock_detail(ticker)
            result.pop("chart", None)  # strip large chart array
            return result
        elif name == "stock_news":
            return self.market.stock_news(ticker)
        elif name == "sec_search":
            return self.sec.search_companies(params["query"])
        elif name == "sec_filings":
            return self.sec.ticker_filings(ticker)
        elif name == "company_financials":
            return self.sec.company_financials(ticker)
        elif name == "ml_signal":
            detail = self.market.stock_detail(ticker)
            funds = {
                "pe_ratio": detail.get("pe_ratio"),
                "revenue_growth": detail.get("revenue_growth"),
            }
            return self.engine.train_and_predict(ticker, funds)
        elif name == "portfolio_snapshot":
            return self.portfolio.snapshot()
        else:
            return {"error": f"Unknown tool: {name}"}

    async def invoke(self, user_message: str) -> str:
        """Process a user message through Claude with tool-use loop."""
        api_key = _get_api_key()
        if not api_key:
            return "Error: No Anthropic API key configured. Set ANTHROPIC_API_KEY or update settings.json."

        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": user_message}]

        for _ in range(MAX_TOOL_ROUNDS):
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            # If Claude's response has no tool_use blocks, we're done
            if response.stop_reason == "end_turn":
                text_parts = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_parts) if text_parts else "No response generated."

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = self._dispatch_tool(block.name, block.input)
                        result_str = json.dumps(result, default=str)
                        if len(result_str) > 4000:
                            result_str = result_str[:4000] + "...(truncated)"
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            # Add assistant response and tool results to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "Analysis complete — reached maximum tool call depth."


class RaphiAgentExecutor(AgentExecutor):
    """A2A protocol AgentExecutor for RAPHI."""

    def __init__(self, agent: RaphiAgent):
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Extract user text from A2A message parts
        # Part is a RootModel wrapping TextPart/FilePart/DataPart
        user_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                inner = part.root if hasattr(part, "root") else part
                if hasattr(inner, "text"):
                    user_text += inner.text

        if not user_text:
            user_text = "What can you help me with?"

        result = await self.agent.invoke(user_text)
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise Exception("cancel not supported")
