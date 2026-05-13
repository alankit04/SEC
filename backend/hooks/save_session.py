"""
save_session.py — Stop hook: persist session summary to project memory.

Reads the audit log for the current session and appends a dated summary
to .claude/agent-memory/MEMORY.md so future sessions have context on
prior analyses and signal history.

Environment variables:
  CLAUDE_SESSION_ID  — session that is ending
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent.parent
MEMORY_DIR  = BASE_DIR / ".claude" / "agent-memory"
MEMORY_FILE = MEMORY_DIR / "MEMORY.md"
AUDIT_FILE  = BASE_DIR / ".raphi_audit" / "financial_decisions.jsonl"
MAX_LINES   = 180  # stay under 200-line auto-load limit


def load_session_actions(session_id: str) -> list:
    if not AUDIT_FILE.exists():
        return []
    actions = []
    with open(AUDIT_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("session_id") == session_id:
                    actions.append(entry)
            except Exception:
                pass
    return actions


def build_summary(session_id: str, actions: list) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tickers = sorted({
        a["input"].get("ticker", "")
        for a in actions
        if a["input"].get("ticker")
    })
    signals = [
        a["result_summary"] for a in actions
        if "ml_signal" in a.get("tool", "")
    ]
    portfolio_checks = [
        a["result_summary"] for a in actions
        if "portfolio" in a.get("tool", "")
    ]

    lines = [f"\n## Session {now} (id: {session_id[:8]}...)"]
    if tickers:
        lines.append(f"- Analyzed: {', '.join(t for t in tickers if t)}")
    for s in signals:
        if s.get("ticker"):
            lines.append(
                f"- Signal: {s['ticker']} → {s.get('direction')} "
                f"({s.get('confidence')}% conf, {s.get('ensemble_accuracy')}% acc)"
            )
    for p in portfolio_checks[:1]:
        # M2 FIX: store only % changes, never absolute $ amounts in MEMORY.md
        if p.get("total_pnl_pct") is not None:
            lines.append(
                f"- Portfolio: P&L {p.get('total_pnl_pct')}% | "
                f"Sharpe {p.get('sharpe', 'n/a')} | snapshot captured"
            )
        elif p:
            lines.append("- Portfolio snapshot captured")
    return "\n".join(lines)


def main():
    session_id = os.environ.get("CLAUDE_SESSION_ID", "unknown")
    actions = load_session_actions(session_id)
    if not actions:
        sys.exit(0)

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary(session_id, actions)
    existing = MEMORY_FILE.read_text() if MEMORY_FILE.exists() else \
        "# RAPHI Agent Memory\nPersistent context across sessions.\n"

    updated = existing.rstrip() + "\n" + summary + "\n"
    lines = updated.splitlines(keepends=True)
    if len(lines) > MAX_LINES:
        # Keep header (first 3 lines) + most recent entries
        lines = lines[:3] + lines[-(MAX_LINES - 3):]

    MEMORY_FILE.write_text("".join(lines))
    sys.exit(0)


if __name__ == "__main__":
    main()
