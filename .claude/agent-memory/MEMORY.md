# RAPHI Agent Memory
Persistent context across sessions. Auto-updated by Stop hook after each session.

## User Preferences
- Default watchlist: NVDA, AAPL, MSFT, META, TSLA, AMZN, GOOGL
- Risk tolerance: moderate
- Preferred memo format: institutional (5-section)
- Model confidence threshold: >60% for LONG/SHORT | 50-60% → HOLD
- Rate limit thresholds: ml_signal 3/min, stock_detail 20/min, stock_news 15/min

## Data Coverage
- SEC EDGAR: 2022Q1–2025Q4 (16 quarters), 9,460 companies, local XBRL num.txt/sub.txt
- Total filings: 112,220 | Estimated data points: ~54.9 million
- dashboard_data.js last regenerated: 2026-04-13 (includes 2024q4)
- ML models: XGBoost + GradientBoosting ensemble, 3-year history, 24h cache
- Market data: yfinance (60s TTL prices, 15m news, 1h fundamentals)
- Primary server: raphi_server.py on port 9999 (unified A2A + FastAPI)

## Portfolio Risk Rules
- VaR > 2% of total portfolio value → CRITICAL alert
- Position P&L < -5% → WARNING: review investment thesis
- Stop-loss within 3% of current price → WARNING
- Single position weight > 30% → NOTE: concentration risk
- Sharpe < 0.5 → NOTE: poor risk-adjusted return

## ML Signal Interpretation
- Confidence > 60% + ensemble accuracy > 75% → actionable LONG/SHORT
- Confidence 50-60% → HOLD regardless of direction
- n_train < 200 → flag insufficient training data
- Always report top 3 SHAP drivers by absolute value

## Architecture Notes
- MCP server: stdio transport, spawned per session by Agent SDK
- A2A + FastAPI unified server: port 9999 (raphi_server.py) — PRIMARY
- Legacy FastAPI: port 8000 (main.py) — kept for backward compat only
- Worktree: zealous-bose — all development happens here; has full hooks + .env
- Subagent hierarchy: memo-synthesizer orchestrates 4 specialists via Task tool
- Python venv: /Users/alan/Desktop/SEC Data/.venv (Python 3.14.3)

## Full Project Status (Audited 2026-04-13)
### ✅ COMPLETE
- Backend (13 files): raphi_server.py, main.py, a2a_server.py, a2a_executor_v2.py,
  raphi_mcp_server.py, market_data.py, sec_data.py, ml_model.py, portfolio_manager.py,
  security.py, conviction_store.py, a2a_test_client.py, hooks/ (4 scripts)
- Agents (6): data-validator, market-analyst, sec-researcher, ml-signals,
  portfolio-risk, memo-synthesizer
- Skills (5): investment-memo, portfolio-review, sector-screen, add-quarter, refresh-data
- Hooks (4): session_start, rate_limit, audit_log, save_session — wired in worktree settings.json
- Security (10+ fixes): H1-H4, M1-M2, C2-C4, L3-L4, Sentry integrated
- Tests: 8/8 passing (conviction_store tests, 2.33s)
- Dashboard: fresh 2026-04-13, 16 quarters, 112K filings, 9.5K companies

### ✅ FIXED 2026-04-14 (round 4 — H1/M3 server-side enforcement)
- X-Internal-Token now validated SERVER-SIDE in TokenAuth (security.py)
- TokenAuth updated: accepts internal_token param (RAPHI_INTERNAL_TOKEN env var)
- Logic: X-Internal-Token checked first (MCP path), X-API-Key/Bearer second (external)
- Wrong internal token → immediate 401 (no fallthrough to API key check)
- raphi_server.py + a2a_server.py both pass internal_token to TokenAuth
- Verified: internal-token-only → 200 ✅, wrong token → 401 ✅,
            API-key-only → 200 ✅, no token → 401 ✅, MCP live call → ✅
- All 3 files synced to worktree

### ✅ FIXED 2026-04-14 (round 3 — full validation)
- Hook commands in worktree settings.json now cd into WORKTREE root (not parent project)
  using absolute .venv path: /Users/alan/Desktop/SEC Data/.venv/bin/python
- Hook BASE_DIR resolves correctly to worktree root: settings.json ✅ MEMORY.md ✅
- a2a_executor_v2 BASE_DIR resolves to worktree root: settings.json ✅
- End-to-end validation PASSED:
    server.js     → port 9999, cwd=PROJECT_DIR ✅
    a2a_server.js → port 9999, cwd=PROJECT_DIR ✅
    :9999 health  → {"status":"ok","a2a":true} ✅
    8/8 FastAPI endpoints → 200 OK ✅
    8/8 MCP tools → registered and reachable ✅
    6/6 hook paths → resolve correctly ✅
    8/8 domain modules → import cleanly ✅
- MEMORY.md synced from root to worktree

### ✅ FIXED 2026-04-14 (round 2)
- raphi_mcp_server.py: BASE_URL updated from localhost:8000 → localhost:9999 (unified server)
- All 8 MCP tools verified reachable: market_overview, stock_detail, stock_news, sec_filings,
  sec_search, ml_signal, portfolio_snapshot, portfolio_alerts
- Worktree synced: generate_dashboard_data.py, dashboard_data.js, portfolio.json,
  raphi_dashboard.html now present in worktree root
- Worktree is now fully self-contained — no external file dependencies

### ✅ FIXED 2026-04-14 (round 1)
- Root server.js updated to port 9999 (matches worktree)
- portfolio-risk agent model aligned: both root + worktree now use claude-haiku-4-5-20251001
- Worktree backend synced: raphi_server.py, market_data.py, sec_data.py, ml_model.py,
  portfolio_manager.py, security.py, conviction_store.py, static/, all 4 hooks (with H4+L3 fixes)
- settings.json copied to worktree root (executor + hooks can find it)
- mcp pinned to >=1.0.0,<2.0.0 in requirements.txt

### ⚠️ REMAINING (needs user action)
- ANTHROPIC_API_KEY had zero credits — AI chat endpoint (/api/chat SSE stream) never fully
  tested end-to-end with live AI responses. Once credits are added, test with:
  curl -sN -H "X-API-Key: $RAPHI_API_KEY" http://localhost:9999/api/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"What is the current NVDA price?"}' | cat

## Security Fixes Implemented
- H1/M3: X-Internal-Token on all FastAPI→MCP calls
- H2: Prompt injection guard (sanitize_user_input, _INJECTION_PATTERNS regex)
- H3: Session encryption (SessionCipher/Fernet in security.py)
- H4: Rate limit file in .raphi_audit/ with chmod 0o700
- M1: Ticker validation regex ^[A-Z]{1,5}$ in raphi_mcp_server.py
- M2: MEMORY.md stores only % changes, never $ amounts
- C2: TokenAuth pure-ASGI middleware (Bearer/X-API-Key)
- C3: Explicit ALLOWED_TOOLS list in a2a_executor_v2.py
- C4: CORS restricted to localhost origins
- L3: Backend URL not exposed in session_start hook
- L4: Input capped at MAX_INPUT_LENGTH=4000 chars
