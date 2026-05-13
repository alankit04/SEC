"""
audit_log.py — PostToolUse hook: append financial tool calls to JSONL audit log.

Records every invocation of decision-relevant MCP tools for compliance and
session memory reconstruction.

Environment variables set by Claude Code:
  CLAUDE_TOOL_NAME    — tool that was called
  CLAUDE_TOOL_INPUT   — JSON-encoded tool arguments
  CLAUDE_TOOL_RESULT  — JSON-encoded tool output
  CLAUDE_SESSION_ID   — current session ID
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent.parent
AUDIT_DIR  = BASE_DIR / ".raphi_audit"
AUDIT_FILE = AUDIT_DIR / "financial_decisions.jsonl"

# Only log these tools — skip noisy market_overview, stock_detail, etc.
AUDITED_TOOLS = {
    "mcp__raphi__ml_signal",
    "mcp__raphi__portfolio_snapshot",
    "mcp__raphi__portfolio_alerts",
    "mcp__raphi__sec_filings",
}


def _summarize(tool: str, result: dict) -> dict:
    """Extract key fields to keep audit log compact."""
    if "ml_signal" in tool:
        return {
            "ticker":            result.get("ticker"),
            "direction":         result.get("direction"),
            "confidence":        result.get("confidence"),
            "ensemble_accuracy": result.get("ensemble_accuracy"),
        }
    if "portfolio_snapshot" in tool or "portfolio_alerts" in tool:
        return {
            "total_value":    result.get("total_value"),
            "total_pnl_pct":  result.get("total_pnl_pct"),
            "var_95":         result.get("var_95"),
            "sharpe":         result.get("sharpe"),
            "alerts_count":   len(result.get("alerts", [])) if isinstance(result.get("alerts"), list) else None,
        }
    if "sec_filings" in tool:
        return {
            "cik":           result.get("cik"),
            "filings_count": len(result.get("filings", [])),
        }
    return {}


def main():
    tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
    if tool_name not in AUDITED_TOOLS:
        sys.exit(0)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    tool_input  = os.environ.get("CLAUDE_TOOL_INPUT", "{}")
    tool_result = os.environ.get("CLAUDE_TOOL_RESULT", "{}")
    session_id  = os.environ.get("CLAUDE_SESSION_ID", "unknown")

    try:
        result_data = json.loads(tool_result)
    except Exception:
        result_data = {"raw": str(tool_result)[:500]}

    try:
        input_data = json.loads(tool_input)
    except Exception:
        input_data = {}

    entry = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "session_id":     session_id,
        "tool":           tool_name,
        "input":          input_data,
        "result_summary": _summarize(tool_name, result_data),
    }

    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
