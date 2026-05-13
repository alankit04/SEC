# AgenticAIFinancialExplainantion

Internal product name: RAPHI.

RAPHI is a local-first, agentic investment intelligence platform that combines live market data, local SEC EDGAR data, explainable ML signals, portfolio risk analytics, durable memory, and an A2A/MCP-based multi-agent layer.

## What Is Implemented

- Unified primary backend on port 9999 via `backend/raphi_server.py`
- Browser dashboard and chat UI via `backend/static/index.html`
- Real-time market data and news sentiment via `backend/market_data.py`
- Local SEC filing search and XBRL-derived financial extraction via `backend/sec_data.py`
- XGBoost + GradientBoosting signal engine with SHAP-style explanations via `backend/ml_model.py`
- Graph-based GNN signal engine via `backend/gnn_model.py`
- Portfolio snapshot, VaR, Sharpe, alerts, and position management via `backend/portfolio_manager.py`
- Conviction ledger via `backend/conviction_store.py`
- Durable graph memory with Neo4j primary and local JSON fallback via `backend/graph_memory.py`
- Deterministic LLM response guardrails via `backend/llm_guardrails.py`
- A2A executor and MCP bridge via `backend/a2a_executor_v2.py` and `backend/raphi_mcp_server.py`

## Architecture

### Primary runtime

- `server.js`
  Starts the unified RAPHI server with your local `.env`.

- `backend/raphi_server.py`
  Main production-style app surface.
  Exposes:
  - A2A routes
  - FastAPI data API under `/api/*`
  - browser chat and memo generation
  - conviction ledger endpoints
  - graph memory endpoints

### Secondary runtime

- `backend/main.py`
  Secondary dev API surface on port 8000.
  It includes direct GNN endpoints and remains useful for development and isolated testing.

### Agent and tool layer

- `backend/a2a_executor_v2.py`
  Claude Agent SDK executor with tool allowlist, session encryption, and prompt sanitization.

- `backend/raphi_mcp_server.py`
  MCP stdio bridge that exposes market, SEC, ML, portfolio, and memory tools to the agent layer.

## Tech Stack

### Backend and API

- Python
- FastAPI / Starlette
- Uvicorn
- Pydantic
- SlowAPI
- SSE streaming responses

### AI and agent orchestration

- Anthropic SDK
- Claude Agent SDK
- A2A protocol server
- MCP server bridge

### Finance and data

- yfinance
- Local SEC EDGAR quarterly datasets in `data/`
- XBRL parsing and filing search

### Machine learning and graph modeling

- scikit-learn
- XGBoost
- SHAP
- NumPy
- pandas
- GraphSAGE-style GNN engine with optional PyTorch Geometric backend

### Memory and persistence

- Neo4j HTTP-backed graph memory
- Local JSON memory fallback in `.raphi_memory/`
- Pickle-based model cache in `.model_cache/`
- JSON and JSONL state for settings, portfolio, and conviction ledger
- Fernet-encrypted A2A session mapping

### Frontend

- Single-file browser app in `backend/static/index.html`
- Vanilla HTML, CSS, and JavaScript
- Fetch + streaming UI

### Security and operations

- API key auth
- internal MCP token auth
- prompt injection sanitization
- deterministic output guardrails
- rate limits
- Sentry support

### Testing

- pytest
- browser-level validation can be run against the local app

## Multi-Agent Status

Yes, this project has real multi-agent capability.

- The A2A path is implemented in `backend/a2a_executor_v2.py`.
- The browser chat in `backend/raphi_server.py` routes through the A2A agent stream first.
- If the A2A path returns no text, the server falls back to a local specialist evidence pass that combines market, SEC, ML, GNN, portfolio, and memory context before generating the final answer.

Practical interpretation:

- A2A orchestration exists.
- MCP tools exist.
- Browser chat is agentic, with a local fallback path.

## Memory Status

Yes, the project can remember past conversation context.

- Primary memory backend: Neo4j
- Fallback memory backend: local durable JSON store
- Entry point: `backend/graph_memory.py`

That means memory is still available even when Neo4j is offline. When Neo4j is unavailable, RAPHI uses the local fallback instead of silently disabling durable memory.

## Response Speed and Caching

### Implemented today

- SSE streaming for chat and memo responses
- market/fundamental/news TTL caches in `backend/market_data.py`
- model artifact cache in `.model_cache/`
- GNN state cache in `.model_cache/gnn_state.pkl`

### Not implemented today

- direct LLM KV cache control
- explicit Anthropic prompt-caching policy in the repo

Practical interpretation:

- The app streams quickly and reuses local data/model caches.
- It does not currently implement true application-managed KV caching for the LLM itself.

## Guardrails Status

Yes, guardrails are implemented.

### Input and tool guardrails

- `sanitize_user_input()` in `backend/security.py`
- MCP internal token protection
- explicit tool allowlist in `backend/a2a_executor_v2.py`
- rate limiting on API routes

### Output guardrails

- deterministic post-generation checks in `backend/llm_guardrails.py`
- memo schema repair for required sections
- overconfidence softening
- risk framing enforcement
- unknown ticker detection
- provenance notes

## Browser Chat Threading

The `New Thread` button now starts a truly fresh browser conversation.

Current behavior:

- clears local chat history
- rotates the persistent `thread_id`
- aborts any in-flight chat stream
- prevents stale responses from leaking into the new thread
- resets the console so the next question starts cleanly

## Repository Layout

```text
backend/
  a2a_executor.py
  a2a_executor_v2.py
  a2a_server.py
  conviction_store.py
  gnn_model.py
  graph_memory.py
  hooks/
  main.py
  market_data.py
  ml_model.py
  paths.py
  portfolio_manager.py
  raphi_mcp_server.py
  raphi_server.py
  requirements.txt
  sec_data.py
  security.py
  static/
data/
docs/
scripts/
tests/
server.js
package.json
portfolio.json
settings.json
```

## Local Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Configure environment variables

Create `.env` at the project root.

Common variables:

- `RAPHI_API_KEY`
- `ANTHROPIC_API_KEY`
- `RAPHI_INTERNAL_TOKEN`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `SENTRY_DSN`

### 3. Start the primary app

```bash
node server.js
```

Open:

- `http://localhost:9999`

### 4. Optional: start the secondary dev API

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Neo4j Memory Setup

Neo4j is optional because local durable fallback memory exists.

To use Neo4j as the primary memory backend:

```bash
export NEO4J_PASSWORD='choose-a-local-password'
docker-compose -f docker-compose.neo4j.yml up -d
```

Then run RAPHI with matching `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`.

## Running Tests

```bash
source .venv/bin/activate
pytest
```

Current repo test coverage includes:

- conviction ledger behavior
- graph memory behavior
- SEC data path and ticker mapping behavior

## Key API Surfaces

### Primary server: `backend/raphi_server.py`

- `GET /api/health`
- `GET /api/market/overview`
- `GET /api/stock/{ticker}`
- `GET /api/stock/{ticker}/news`
- `GET /api/stock/{ticker}/signals`
- `GET /api/stock/{ticker}/filings`
- `GET /api/portfolio`
- `PUT /api/portfolio`
- `GET /api/signals`
- `GET /api/alerts`
- `POST /api/chat`
- `POST /api/memo/{ticker}`
- `GET /api/convictions/stats`
- `GET /api/convictions/ledger`
- `GET /api/memory/status`
- `GET /api/memory/retrieve`

### Secondary dev server: `backend/main.py`

- `GET /api/stock/{ticker}/gnn`
- `POST /api/gnn/train`
- `GET /api/gnn/status`

## Current Notes

- `backend/raphi_server.py` is the authoritative app surface.
- `backend/main.py` remains useful for development and direct GNN access.
- The GNN is implemented and blended into signal generation when its cached graph model is available.
- Durable memory is available even without Neo4j because of the local fallback backend.

## License

No license file is included in this repository.