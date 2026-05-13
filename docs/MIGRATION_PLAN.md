# MIGRATION PLAN — RAPHI A2A Repository Restructure

## 1. Current State

```
.                                     # FLAT — mixed concerns at root
├── .env                              # secrets (UNPROTECTED by gitignore)
├── .model_cache/                     # ML pkl caches (UNPROTECTED)
├── .raphi_audit/                     # conviction ledger data (UNPROTECTED)
├── a2a_server.js                     # legacy Node spawn wrapper
├── server.js                         # legacy Node spawn wrapper
├── dashboard_data.js                 # generated JS asset
├── generate_dashboard_data.py        # data pipeline script
├── raphi_dashboard.html              # legacy 3400-line HTML dashboard
├── test_50.py                        # QA validation suite
├── portfolio.json                    # portfolio data (mutable)
├── settings.json                     # user config (mutable)
├── backend/
│   ├── raphi_server.py               # main FastAPI hub (A2A + API)
│   ├── a2a_server.py                 # alternative A2A-only entry
│   ├── a2a_executor_v2.py            # Claude Agent SDK executor
│   ├── raphi_mcp_server.py           # MCP stdio server
│   ├── market_data.py                # yfinance wrapper
│   ├── sec_data.py                   # SEC EDGAR reader
│   ├── ml_model.py                   # XGBoost + ensemble signals
│   ├── portfolio_manager.py          # portfolio P&L / VaR
│   ├── conviction_store.py           # JSONL conviction ledger
│   ├── security.py                   # auth, sanitization, cipher
│   ├── requirements.txt              # Python dependencies
│   ├── hooks/                        # Claude Code hook scripts
│   │   ├── audit_log.py
│   │   ├── rate_limit.py
│   │   ├── save_session.py
│   │   └── session_start.py
│   └── static/
│       ├── index.html                # legacy HTML dashboard
│       └── dist/                     # React build output
├── frontend/                         # React + Vite + shadcn/ui
│   ├── src/
│   ├── package.json
│   └── ...
└── (no .gitignore at root)
```

### Problems Identified
1. **No root .gitignore** — .env, .model_cache, __pycache__ exposed
2. **Flat root** — scripts, configs, legacy wrappers, test files all at root
3. **Data singletons duplicated** in 3 files (raphi_server, a2a_server, raphi_mcp_server)
4. **SEC root-finding logic duplicated** in raphi_server.py + raphi_mcp_server.py
5. **`backend/` mixes concerns** — agents, domain services, infra, and hooks all flat
6. **Legacy files at root** — server.js, a2a_server.js, raphi_dashboard.html, dashboard_data.js
7. **No separation** between config data (portfolio.json, settings.json) and source code
8. **Generated artifacts** (dashboard_data.js, .model_cache/, dist/) in source tree

---

## 2. Target Structure

```
.
├── .gitignore                        # NEW — comprehensive
├── .env                              # stays at root (gitignored)
├── README.md                         # (future)
│
├── backend/
│   ├── requirements.txt
│   │
│   ├── services/                     # domain data layers
│   │   ├── __init__.py
│   │   ├── market_data.py
│   │   ├── sec_data.py
│   │   ├── ml_model.py
│   │   ├── portfolio_manager.py
│   │   └── conviction_store.py
│   │
│   ├── agents/                       # agent runtimes
│   │   ├── __init__.py
│   │   ├── a2a_executor_v2.py
│   │   └── a2a_server.py
│   │
│   ├── mcp/                          # MCP server
│   │   ├── __init__.py
│   │   └── raphi_mcp_server.py
│   │
│   ├── infra/                        # platform utilities
│   │   ├── __init__.py
│   │   ├── security.py
│   │   └── singletons.py            # NEW — shared data singletons
│   │
│   ├── hooks/                        # Claude Code hooks (unchanged)
│   │   ├── __init__.py
│   │   ├── audit_log.py
│   │   ├── rate_limit.py
│   │   ├── save_session.py
│   │   └── session_start.py
│   │
│   ├── server.py                     # renamed from raphi_server.py
│   └── static/                       # served assets
│       ├── index.html                # legacy fallback
│       └── dist/                     # React build (gitignored)
│
├── frontend/                         # React app (unchanged)
│   ├── src/
│   ├── package.json
│   └── ...
│
├── scripts/                          # one-off / pipeline scripts
│   └── generate_dashboard_data.py
│
├── tests/                            # test suites
│   └── test_50.py
│
├── data/                             # mutable runtime data
│   ├── portfolio.json
│   └── settings.json
│
├── legacy/                           # deprecated files (kept for reference)
│   ├── server.js
│   ├── a2a_server.js
│   ├── dashboard_data.js
│   └── raphi_dashboard.html
│
└── docs/                             # documentation
    ├── MIGRATION_PLAN.md
    ├── RESTRUCTURE_REPORT.md
    └── VALIDATION_REPORT.md
```

---

## 3. File Move Mapping

| Current Path | Target Path | Notes |
|---|---|---|
| `backend/market_data.py` | `backend/services/market_data.py` | domain layer |
| `backend/sec_data.py` | `backend/services/sec_data.py` | domain layer |
| `backend/ml_model.py` | `backend/services/ml_model.py` | domain layer |
| `backend/portfolio_manager.py` | `backend/services/portfolio_manager.py` | domain layer |
| `backend/conviction_store.py` | `backend/services/conviction_store.py` | domain layer |
| `backend/a2a_executor_v2.py` | `backend/agents/a2a_executor_v2.py` | agent runtime |
| `backend/a2a_server.py` | `backend/agents/a2a_server.py` | agent runtime |
| `backend/raphi_mcp_server.py` | `backend/mcp/raphi_mcp_server.py` | MCP server |
| `backend/security.py` | `backend/infra/security.py` | platform util |
| `backend/raphi_server.py` | `backend/server.py` | unified entry |
| `test_50.py` | `tests/test_50.py` | tests dir |
| `generate_dashboard_data.py` | `scripts/generate_dashboard_data.py` | scripts dir |
| `portfolio.json` | `data/portfolio.json` | runtime data |
| `settings.json` | `data/settings.json` | runtime data |
| `server.js` | `legacy/server.js` | deprecated |
| `a2a_server.js` | `legacy/a2a_server.js` | deprecated |
| `dashboard_data.js` | `legacy/dashboard_data.js` | generated |
| `raphi_dashboard.html` | `legacy/raphi_dashboard.html` | superseded |
| (new) | `backend/infra/singletons.py` | extract shared singletons |
| (new) | `.gitignore` | root gitignore |

---

## 4. Import Updates Required

### backend/server.py (was raphi_server.py)
```python
# OLD                                  →  NEW
from market_data import MarketData     →  from backend.services.market_data import MarketData
from sec_data import SECData           →  from backend.services.sec_data import SECData
from ml_model import SignalEngine      →  from backend.services.ml_model import SignalEngine
from portfolio_manager import ...      →  from backend.services.portfolio_manager import PortfolioManager
from a2a_executor_v2 import ...        →  from backend.agents.a2a_executor_v2 import ...
from security import ...               →  from backend.infra.security import ...
from conviction_store import ...       →  from backend.services.conviction_store import ...
```

### backend/agents/a2a_executor_v2.py
```python
from security import ...               →  from backend.infra.security import ...
from market_data import ...            →  from backend.services.market_data import ...
# etc.
```

### backend/agents/a2a_server.py
```python
# All data imports → backend.services.*
# Executor import → backend.agents.a2a_executor_v2
```

### backend/mcp/raphi_mcp_server.py
```python
# All data imports → backend.services.*
```

### backend/server.py path references
```python
# portfolio.json → ../data/portfolio.json  OR  data/portfolio.json (relative to worktree root)
# settings.json  → ../data/settings.json
```

---

## 5. Risks

| Risk | Mitigation |
|------|-----------|
| Import paths break | Use relative imports within backend package; verify with `python -c "from backend.server import app"` |
| portfolio.json/settings.json path changes | Update `_WORKTREE_ROOT` references; use a `DATA_DIR` constant |
| Hooks use env-based paths | Hooks read from stdin/env — no file imports, safe to leave in place |
| Node.js wrappers break | Moved to legacy/; they're dev convenience, not production |
| 50-test suite hits localhost:9999 | Only URL changes needed if entry module renamed |
| React build output path | `frontend/vite.config.ts` already outputs to `../backend/static/dist` — still correct |

---

## 6. Rollout Order

1. **Create .gitignore** (zero risk)
2. **Create directory structure** (mkdir services/, agents/, mcp/, infra/, scripts/, tests/, data/, legacy/, docs/)
3. **Create backend/infra/singletons.py** (new file, no existing imports break)
4. **Move domain services** (market_data, sec_data, ml_model, portfolio_manager, conviction_store → services/)
5. **Move agents** (a2a_executor_v2, a2a_server → agents/)
6. **Move MCP** (raphi_mcp_server → mcp/)
7. **Move security** (security.py → infra/)
8. **Move root files** (test, script, data, legacy)
9. **Rename raphi_server.py → server.py**
10. **Update all imports** in server.py, a2a_executor_v2.py, a2a_server.py, raphi_mcp_server.py
11. **Update path constants** (DATA_DIR, SETTINGS_FILE, etc.)
12. **Update server.js/a2a_server.js references** (in legacy/)
13. **Verify: `python -c "from backend.server import app"`**
14. **Run test_50.py** to validate runtime
