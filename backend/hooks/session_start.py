"""
session_start.py — SessionStart hook: inject watchlist and risk context.

Prints context that Claude Code includes in the initial system prompt,
giving every session immediate awareness of the user's preferences.
"""

import json
import sys
from pathlib import Path

BASE_DIR      = Path(__file__).parent.parent.parent
SETTINGS_FILE = BASE_DIR / "settings.json"
MEMORY_FILE   = BASE_DIR / ".claude" / "agent-memory" / "MEMORY.md"


def main():
    settings = {}
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE) as f:
                settings = json.load(f)
        except Exception:
            pass

    watchlist  = settings.get("watchlist", ["NVDA", "AAPL", "MSFT", "META", "TSLA"])
    risk_tol   = settings.get("risk_tolerance", "moderate")

    print(f"[RAPHI] Watchlist: {', '.join(watchlist)}")
    print(f"[RAPHI] Risk tolerance: {risk_tol}")
    # L3 FIX: backend URL removed — Claude doesn't need it and it leaks infra details

    if MEMORY_FILE.exists():
        lines = MEMORY_FILE.read_text().splitlines()
        session_lines = [l for l in lines if l.startswith("## Session")]
        if session_lines:
            print(f"[RAPHI] {len(session_lines)} prior analysis session(s) in memory")
            # Print the most recent session summary
            last_idx = max(i for i, l in enumerate(lines) if l.startswith("## Session"))
            recent = lines[last_idx:last_idx + 5]
            for line in recent:
                print(f"[RAPHI] {line}")

    sys.exit(0)


if __name__ == "__main__":
    main()
