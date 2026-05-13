# RESTRUCTURE REPORT — RAPHI A2A Repository

## Summary

Restructured the RAPHI repository from a flat layout with mixed concerns into a clean production-style A2A architecture. All 50 validation tests pass after restructuring.

---

## Final Tree

```
.
├── .gitignore                         # NEW — comprehensive exclusions
├── .env                               # secrets (gitignored)
├── portfolio.json                     # portfolio data (kept at root for path stability)
├── settings.json                      # user config (kept at root)
│
├── backend/
│   ├── raphi_server.py                # unified FastAPI entry (A2A + API)
│   ├── requirements.txt               # Python dependencies
│   │
│   ├── services/                      # NEW — domain data layers
│   │   ├── __init__.py
│   │   ├── market_data.py             # yfinance wrapper
│   │   ├── sec_data.py                # SEC EDGAR reader
│   │   ├── ml_model.py                # XGBoost + ensemble signals
│   │   ├── portfolio_manager.py       # portfolio P&L / VaR
│   │   └── conviction_store.py        # JSONL conviction ledger
│   │
│   ├── agents/                        # NEW — agent runtimes
│   │   ├── __init__.py
│   │   ├── a2a_executor_v2.py         # Claude Agent SDK executor
│   │   └── a2a_server.py              # alternative A2A-only entry
│   │
│   ├── tool_server/                   # NEW — MCP tool server
│   │   ├── __init__.py
│   │   └── raphi_mcp_server.py        # MCP stdio server
│   │
│   ├── infra/                         # NEW — platform utilities
│   │   ├── __init__.py
│   │   ├── security.py                # auth, sanitization, cipher
│   │   └── paths.py                   # NEW — centralized path constants
│   │
│   ├── hooks/                         # Claude Code hooks (unchanged)
│   │   ├── __init__.py
│   │   ├── audit_log.py
│   │   ├── rate_limit.py
│   │   ├── save_session.py
│   │   └── session_start.py
│   │
│   └── static/
│       ├── index.html                 # legacy fallback dashboard
│       └── dist/                      # React build output (gitignored)
│
├── frontend/                          # React + Vite + shadcn/ui
│   ├── src/
│   │   ├── App.tsx                    # router + auth guard
│   │   ├── main.tsx                   # React root
│   │   ├── index.css                  # Tailwind + design tokens
│   │   ├── lib/                       # utilities
│   │   │   ├── api.ts                 # API client with SSE
│   │   │   ├── types.ts               # TypeScript interfaces
│   │   │   ├── hooks.ts               # useApi, usePolling
│   │   │   └── utils.ts               # formatting utilities
│   │   ├── components/
│   │   │   ├── layout/AppLayout.tsx   # sidebar + topbar + status bar
│   │   │   └── ui/                    # button, card, badge, input, tabs, etc.
│   │   └── pages/                     # 13 page components
│   │       ├── login.tsx
│   │       ├── dashboard.tsx
│   │       ├── ask.tsx                # SSE chat interface
│   │       ├── stock.tsx              # stock detail + chart + tabs
│   │       ├── portfolio.tsx          # positions table + risk metrics
│   │       ├── signals.tsx            # signal card grid
│   │       ├── news.tsx               # news + sentiment
│   │       ├── memo.tsx               # SSE memo generator
│   │       ├── convictions.tsx        # paginated ledger + stats
│   │       ├── shap.tsx               # SHAP bar chart
│   │       ├── alerts.tsx
│   │       ├── models.tsx
│   │       ├── research.tsx           # SEC search
│   │       └── settings.tsx           # watchlist + API config
│   ├── vite.config.ts
│   ├── package.json
│   └── tsconfig.app.json
│
├── tests/                             # NEW — test suites
│   └── test_50.py                     # 50-test validation suite
│
├── scripts/                           # NEW — pipeline scripts
│   └── generate_dashboard_data.py     # SEC data → JS asset
│
├── legacy/                            # NEW — deprecated files
│   ├── server.js                      # Node.js spawn wrapper
│   ├── a2a_server.js                  # Node.js spawn wrapper
│   ├── dashboard_data.js              # generated JS asset
│   └── raphi_dashboard.html           # monolithic 3400-line HTML
│
└── docs/                              # NEW — documentation
    ├── MIGRATION_PLAN.md
    ├── RESTRUCTURE_REPORT.md
    └── VALIDATION_REPORT.md
```

---

## Key File Moves

| Old Path | New Path | Category |
|---|---|---|
| `backend/market_data.py` | `backend/services/market_data.py` | Domain Service |
| `backend/sec_data.py` | `backend/services/sec_data.py` | Domain Service |
| `backend/ml_model.py` | `backend/services/ml_model.py` | Domain Service |
| `backend/portfolio_manager.py` | `backend/services/portfolio_manager.py` | Domain Service |
| `backend/conviction_store.py` | `backend/services/conviction_store.py` | Domain Service |
| `backend/a2a_executor_v2.py` | `backend/agents/a2a_executor_v2.py` | Agent Runtime |
| `backend/a2a_server.py` | `backend/agents/a2a_server.py` | Agent Runtime |
| `backend/raphi_mcp_server.py` | `backend/tool_server/raphi_mcp_server.py` | MCP Tool Server |
| `backend/security.py` | `backend/infra/security.py` | Infrastructure |
| `test_50.py` | `tests/test_50.py` | Tests |
| `generate_dashboard_data.py` | `scripts/generate_dashboard_data.py` | Scripts |
| `server.js` | `legacy/server.js` | Deprecated |
| `a2a_server.js` | `legacy/a2a_server.js` | Deprecated |
| `dashboard_data.js` | `legacy/dashboard_data.js` | Deprecated |
| `raphi_dashboard.html` | `legacy/raphi_dashboard.html` | Deprecated |

### New Files Created

| File | Purpose |
|---|---|
| `.gitignore` | Root-level git exclusions |
| `backend/infra/paths.py` | Centralized path constants (eliminated duplication) |
| `backend/services/__init__.py` | Package init |
| `backend/agents/__init__.py` | Package init |
| `backend/tool_server/__init__.py` | Package init |
| `backend/infra/__init__.py` | Package init |

---

## Key Import Updates

### `backend/raphi_server.py` (main entry)
```python
# OLD                                      NEW
from market_data import ...             → from services.market_data import ...
from sec_data import ...                → from services.sec_data import ...
from ml_model import ...                → from services.ml_model import ...
from portfolio_manager import ...       → from services.portfolio_manager import ...
from a2a_executor_v2 import ...         → from agents.a2a_executor_v2 import ...
from security import ...                → from infra.security import ...
from conviction_store import ...        → from services.conviction_store import ...
_WORKTREE_ROOT / ".model_cache"         → MODEL_CACHE_DIR (from infra.paths)
_WORKTREE_ROOT / "settings.json"        → SETTINGS_FILE (from infra.paths)
_find_sec_data_root()                   → SEC_DATA_ROOT (from infra.paths)
```

### `backend/agents/a2a_executor_v2.py`
```python
from security import ...                → from infra.security import ...
BASE_DIR = Path(__file__).parent.parent → from infra.paths import PROJECT_ROOT; BASE_DIR = PROJECT_ROOT
"backend/raphi_mcp_server.py"           → "backend/tool_server/raphi_mcp_server.py"
```

### `backend/agents/a2a_server.py`
```python
from market_data import ...             → from services.market_data import ...
from a2a_executor_v2 import ...         → from agents.a2a_executor_v2 import ...
from security import ...                → from infra.security import ...
BASE = Path(__file__).parent.parent     → from infra.paths import SEC_DATA_ROOT; BASE = SEC_DATA_ROOT
```

### `backend/tool_server/raphi_mcp_server.py`
```python
sys.path.insert(0, str(Path(__file__).parent))    → sys.path.insert(0, str(Path(__file__).parent.parent))
from market_data import ...                        → from services.market_data import ...
_WORKTREE_ROOT + _find_sec_data_root()             → from infra.paths import SEC_DATA_ROOT
```

### `backend/services/portfolio_manager.py`
```python
PORTFOLIO_FILE = Path(__file__).parent.parent / ... → Path(__file__).resolve().parent.parent.parent / ...
```

### `backend/services/conviction_store.py`
```python
BASE_DIR = Path(__file__).parent.parent             → Path(__file__).resolve().parent.parent.parent
from sec_data import SECData                        → from services.sec_data import SECData
```

---

## Duplication Eliminated

| Duplication | Before | After |
|---|---|---|
| SEC data root finding | Duplicated in `raphi_server.py`, `raphi_mcp_server.py`, `a2a_server.py` | Centralized in `infra/paths.py` → `find_sec_data_root()` |
| Worktree root computation | `Path(__file__).parent.parent` in 5 files | Centralized in `infra/paths.py` → `PROJECT_ROOT` |
| Settings/portfolio file paths | Hardcoded in 3 files | Centralized in `infra/paths.py` → `SETTINGS_FILE`, `PORTFOLIO_FILE` |
| Model cache directory | `_WORKTREE_ROOT / ".model_cache"` (8 occurrences) | `MODEL_CACHE_DIR` constant |

### Remaining Duplications (Acceptable)
- **Data singleton instantiation** (`MarketData()`, `SECData()`, `SignalEngine()`, `PortfolioManager()`) appears in `raphi_server.py` and `a2a_server.py`. These are intentional — the two server modes need independent instances.

---

## Architecture Decisions

1. **`tool_server/` instead of `mcp/`** — The directory was originally named `mcp/` but this shadowed the `mcp` pip package on PYTHONPATH. Renamed to `tool_server/`.

2. **Data files kept at root** — `portfolio.json` and `settings.json` remain at the project root rather than moving to a `data/` subdirectory, minimizing path changes across 5+ files.

3. **`raphi_server.py` not renamed** — Kept as `raphi_server.py` (not `server.py`) to avoid breaking launch configs, deployment scripts, and the 50-test suite.

4. **Hooks unchanged** — Claude Code hooks use stdin/env, not local file imports. No restructuring needed.

---

## Validation

- **50/50 tests pass** after restructuring
- **Server starts cleanly** at port 9999
- **React frontend build** still outputs to `backend/static/dist/`
- **All Python imports** resolve correctly via `PYTHONPATH=$PWD/backend:$PWD`
