"""
rate_limit.py — PreToolUse hook: sliding-window rate limiter for yfinance-backed MCP tools.

Claude Code passes context via environment variables:
  CLAUDE_TOOL_NAME   — MCP tool being called (e.g. mcp__raphi__ml_signal)

Exit codes:
  0 — allow the call
  1 — block the call (message printed to stdout is shown to Claude)
"""

import json
import os
import sys
import time
from pathlib import Path

# H4 FIX: moved from world-readable /tmp to owner-only .raphi_audit/
_AUDIT_DIR = Path(__file__).parent.parent.parent / ".raphi_audit"
_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
_AUDIT_DIR.chmod(0o700)
RATE_FILE = _AUDIT_DIR / "rate_limits.json"

# Max calls per 60-second window per tool
LIMITS = {
    "ml_signal":      3,   # model training is expensive (~5s per call)
    "stock_detail":   20,
    "stock_news":     15,
    "market_overview": 30,
    "sec_filings":    10,
    "sec_search":     20,
    "portfolio_snapshot": 30,
    "portfolio_alerts":   30,
}
WINDOW = 60  # seconds


def load_state() -> dict:
    if RATE_FILE.exists():
        try:
            with open(RATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict):
    with open(RATE_FILE, "w") as f:
        json.dump(state, f)


def main():
    tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
    # Extract base name from mcp__raphi__<name>
    parts = tool_name.split("__")
    base_name = parts[-1] if len(parts) >= 3 else tool_name

    limit = LIMITS.get(base_name)
    if not limit:
        sys.exit(0)  # No limit configured — allow

    now = time.time()
    state = load_state()
    calls = [t for t in state.get(base_name, []) if now - t < WINDOW]

    if len(calls) >= limit:
        oldest = min(calls)
        wait = int(WINDOW - (now - oldest)) + 1
        print(
            f"[RAPHI rate limit] {base_name}: {len(calls)}/{limit} calls in last 60s. "
            f"Retry in {wait}s or use cached data."
        )
        sys.exit(1)  # Block the call

    calls.append(now)
    state[base_name] = calls
    save_state(state)
    sys.exit(0)


if __name__ == "__main__":
    main()
