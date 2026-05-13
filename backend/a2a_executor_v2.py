"""
a2a_executor_v2.py — RAPHI A2A Executor (Claude Agent SDK — production)

Security fixes applied:
  C3  permission_mode changed from bypassPermissions → acceptEdits
  C3  can_use_tool callback blocks any tool not in the approved ALLOWED_TOOLS list
  H2  sanitize_user_input() guards against prompt injection before query()
  H3  SessionCipher encrypts sessions.json at rest (Fernet symmetric key)
  L4  user_text hard-capped at 4000 chars (via sanitize_user_input)
"""

import json
import logging
import os
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from claude_agent_sdk.types import (
    McpStdioServerConfig,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from security import SessionCipher, sanitize_user_input
from graph_memory import get_graph_memory
from llm_guardrails import GuardrailContext, validate_and_repair_response

logger = logging.getLogger("raphi.executor")

BASE_DIR           = Path(__file__).parent.parent
SETTINGS_FILE      = BASE_DIR / "settings.json"
SESSION_STORE_DIR  = BASE_DIR / ".raphi_audit"
SESSION_STORE_FILE = SESSION_STORE_DIR / "sessions.json"
CLAUDE_CLI         = Path.home() / ".local" / "bin" / "claude"

# MCP server config — absolute path, no cwd dependency
MCP_SERVERS: dict[str, McpStdioServerConfig] = {
    "raphi": {
        "type": "stdio",
        "command": str(BASE_DIR / ".venv" / "bin" / "python"),
        "args": [str(BASE_DIR / "backend" / "raphi_mcp_server.py")],
        "env": {
            # H1: pass internal token so MCP server can auth with FastAPI
            "RAPHI_INTERNAL_TOKEN": os.environ.get("RAPHI_INTERNAL_TOKEN", ""),
        },
    }
}

# C3: Explicit allowlist — any tool NOT in this list is denied by can_use_tool
ALLOWED_TOOLS = [
    "Task",
    "mcp__raphi__market_overview",
    "mcp__raphi__stock_detail",
    "mcp__raphi__stock_news",
    "mcp__raphi__sec_filings",
    "mcp__raphi__sec_search",
    "mcp__raphi__sec_universe",
    "mcp__raphi__sec_industries",
    "mcp__raphi__ml_signal",
    "mcp__raphi__gnn_signal",
    "mcp__raphi__gnn_status",
    "mcp__raphi__gnn_train",
    "mcp__raphi__portfolio_snapshot",
    "mcp__raphi__portfolio_alerts",
    "mcp__raphi__memory_status",
    "mcp__raphi__memory_retrieve",
]
_ALLOWED_TOOLS_SET = set(ALLOWED_TOOLS)

SYSTEM_PROMPT = """You are RAPHI (Real-time Agentic Platform for Human Investment Intelligence).

You have specialist subagents (dispatch via Task tool):
- @market-analyst: real-time prices, technicals, fundamentals, news sentiment
- @sec-researcher: SEC EDGAR XBRL financials (15 quarters, 9,457+ companies)
- @ml-signals: XGBoost+LSTM trading signals with SHAP feature attribution plus GraphSAGE neighbor influence
- @portfolio-risk: VaR, P&L, Sharpe ratio, stop-loss alerts
- @memo-synthesizer: full investment memo (orchestrates all four above in parallel)

Decision rule:
- Single-dimension query (price only, one metric) → call mcp__raphi__* directly
- SEC universe/screener query → call mcp__raphi__sec_universe or mcp__raphi__sec_industries before narrowing to tickers
- Graph/peer-influence query → call mcp__raphi__gnn_signal; call mcp__raphi__gnn_train if status says the graph is not trained
- Multi-source analysis or investment recommendation → delegate to @memo-synthesizer

Always cite specific data points. Use institutional investment language.
Never fabricate numbers — if a tool fails, state it explicitly.

Format responses for the RAPHI web console:
- Clean Markdown with short headings and concise bullets.
- For memos use: Recommendation, Key Evidence, GNN / Peer Influence, Risks, Trade Plan.
- Put recommendation, confidence, and target/stop in the first 2 lines.
- Do not use ASCII diagrams, pipe-delimited chains, raw graph art, or dense one-paragraph blocks."""


# ── C3: Tool guard callback ───────────────────────────────────────────
async def _tool_guard(
    tool_name: str,
    tool_input: dict,
    context: ToolPermissionContext,
) -> PermissionResultAllow | PermissionResultDeny:
    """Block any tool not in the approved allowlist."""
    if tool_name in _ALLOWED_TOOLS_SET:
        return PermissionResultAllow()
    logger.warning("Blocked unapproved tool request: %s", tool_name)
    return PermissionResultDeny(
        reason=f"Tool '{tool_name}' is not in RAPHI's approved tool list."
    )


# ── H3: Encrypted session store ───────────────────────────────────────
_cipher = SessionCipher()


class SessionStore:
    """Persist A2A task_id → Claude session_id, encrypted at rest."""

    def __init__(self) -> None:
        SESSION_STORE_DIR.mkdir(parents=True, exist_ok=True)
        # chmod 700 the audit dir so only owner can read it
        SESSION_STORE_DIR.chmod(0o700)
        self._data: dict[str, str] = {}
        if SESSION_STORE_FILE.exists():
            try:
                raw = SESSION_STORE_FILE.read_text().strip()
                decrypted = _cipher.decrypt(raw)
                self._data = json.loads(decrypted)
            except Exception:
                pass

    def get(self, task_id: str) -> str | None:
        return self._data.get(str(task_id))

    def save(self, task_id: str, session_id: str) -> None:
        self._data[str(task_id)] = session_id
        payload = json.dumps(self._data)
        SESSION_STORE_FILE.write_text(_cipher.encrypt(payload))
        SESSION_STORE_FILE.chmod(0o600)  # owner read/write only


_session_store = SessionStore()
_graph_memory = get_graph_memory()


def _get_api_key() -> str:
    """
    Read API key from env var only — never from settings.json (C1 fix).
    Returns empty string when using Claude OAuth login (claude-agent-sdk
    will use the CLI's OAuth session automatically when no key is set).
    """
    return os.environ.get("ANTHROPIC_API_KEY", "")


class RaphiAgent:
    """RAPHI agent powered by Claude Agent SDK."""

    async def invoke(self, user_message: str, task_id: str | None = None) -> str:
        response_parts: list[str] = []
        async for event in self.stream(user_message, task_id=task_id):
            if event["event"] == "token":
                response_parts.append(event["data"])
            elif event["event"] == "error":
                return event["data"]
        return "".join(response_parts) if response_parts else "Analysis complete."

    async def stream(self, user_message: str, task_id: str | None = None):
        """Run the real Claude Agent SDK path and yield browser-friendly events.

        The SDK gives structured assistant/result messages rather than guaranteed
        token deltas, so RAPHI collects, validates, stores memory, then emits
        small chunks for the web console. The work is still the A2A/MCP agent
        orchestration path, not the old single-call chat path.
        """
        # H2 — sanitize before touching the AI (raises ValueError on injection)
        try:
            user_message = sanitize_user_input(user_message)
        except ValueError as e:
            yield {"event": "error", "data": f"Request rejected: {e}"}
            return

        original_user_message = user_message
        try:
            memories = _graph_memory.retrieve_context(user_message, limit=6)
            memory_context = _graph_memory.format_context(memories)
        except Exception:
            memory_context = ""

        yield {
            "event": "step",
            "data": json.dumps({
                "id": "memory",
                "label": "Permanent memory retrieved" if memory_context else "Permanent memory checked",
            }),
        }

        if memory_context:
            user_message = (
                f"{user_message}\n\n"
                "PERMANENT GRAPH MEMORY CONTEXT:\n"
                f"{memory_context}\n\n"
                "Use this memory only when it is relevant to the user's request."
            )

        resume_id = _session_store.get(task_id) if task_id else None

        # API key is optional — Claude CLI uses OAuth session when not set.
        # Confirmed working: `claude --print` succeeds via OAuth without a key.
        api_key = _get_api_key()
        env: dict[str, str] = {}
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key  # explicit key takes priority over OAuth

        yield {
            "event": "step",
            "data": json.dumps({
                "id": "agentic",
                "label": "A2A agent swarm engaged via Claude Agent SDK",
            }),
        }

        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers=MCP_SERVERS,
            allowed_tools=ALLOWED_TOOLS,
            permission_mode="acceptEdits",    # C3: was bypassPermissions
            can_use_tool=_tool_guard,          # C3: hard deny for unlisted tools
            max_turns=15,
            cwd=str(BASE_DIR),
            cli_path=str(CLAUDE_CLI),
            setting_sources=["project"],
            env=env,                           # empty dict = use CLI OAuth session
            **({"resume": resume_id} if resume_id else {}),
        )

        # can_use_tool requires streaming mode — wrap string in async generator
        async def _prompt_iter():
            yield user_message

        response_parts: list[str] = []
        latest_session_id: str | None = None

        try:
            async for event in query(prompt=_prompt_iter(), options=options):
                if isinstance(event, AssistantMessage):
                    if event.session_id:
                        latest_session_id = event.session_id
                    for block in event.content:
                        if isinstance(block, TextBlock) and block.text:
                            response_parts.append(block.text)
                elif isinstance(event, ResultMessage):
                    if event.result and event.result not in response_parts:
                        response_parts.append(event.result)
        except Exception as exc:
            yield {"event": "error", "data": f"Agentic chat failed: {exc}"}
            return

        if latest_session_id and task_id:
            _session_store.save(task_id, latest_session_id)

        if not response_parts:
            yield {"event": "error", "data": "Agentic chat produced no assistant text."}
            return

        result = "\n".join(response_parts)
        result, guardrail_report = validate_and_repair_response(
            result,
            GuardrailContext(
                allowed_tickers=set(),
                source_summary="A2A MCP tools and permanent memory",
                require_memo_schema=bool(
                    any(term in original_user_message.lower() for term in ("memo", "investment thesis", "recommendation"))
                ),
            ),
        )
        if guardrail_report.repairs or guardrail_report.warnings:
            yield {
                "event": "step",
                "data": json.dumps({
                    "id": "guardrails",
                    "label": "LLM guardrails validated and repaired the response",
                    "repairs": guardrail_report.repairs,
                    "warnings": guardrail_report.warnings,
                }),
            }

        try:
            _graph_memory.remember_interaction(
                user_text=original_user_message,
                assistant_text=result,
                source="a2a",
                metadata={"task_id": task_id},
                importance=0.62,
            )
        except Exception:
            pass

        for idx in range(0, len(result), 600):
            yield {"event": "token", "data": result[idx:idx + 600]}


class RaphiAgentExecutor(AgentExecutor):
    """A2A protocol AgentExecutor backed by the Claude Agent SDK."""

    def __init__(self, agent: RaphiAgent) -> None:
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        user_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                inner = part.root if hasattr(part, "root") else part
                if hasattr(inner, "text"):
                    user_text += inner.text

        if not user_text:
            user_text = "Provide a market overview."

        task_id = str(context.task_id) if getattr(context, "task_id", None) else None
        result  = await self.agent.invoke(user_text, task_id=task_id)
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise Exception("cancel not supported")
