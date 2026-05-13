# RAPHI

**RAPHI** is a local-first financial intelligence platform that combines market data, SEC filings, machine learning signals, portfolio risk analytics, durable memory, and an agent/tool execution layer into one research-grade investment analysis system.

The system is designed to answer questions such as:

- What changed in a company’s market, filing, or signal profile?
- Which portfolio positions are showing elevated risk?
- What evidence supports or weakens a buy/sell/hold thesis?
- How do model signals, SEC fundamentals, market data, and historical memory align?

RAPHI is not a trading bot. It is an explainable investment analysis and decision-support system.

---

## Core Capabilities

| Area | Description |
|---|---|
| Market Intelligence | Retrieves market data, ticker-level summaries, and news sentiment. |
| SEC Analysis | Searches local SEC EDGAR data and extracts XBRL-derived financial information. |
| ML Signal Engine | Uses XGBoost and Gradient Boosting models with SHAP-style explanations. |
| Graph Signals | Includes a GraphSAGE-style GNN signal engine with optional PyTorch Geometric support. |
| Portfolio Risk | Tracks holdings, portfolio snapshots, VaR, Sharpe ratio, alerts, and position-level risk. |
| Conviction Tracking | Maintains a conviction ledger for investment reasoning and thesis history. |
| Memory | Uses Neo4j as the primary graph memory backend with local JSON fallback. |
| Agent Layer | Supports A2A-style agent execution and MCP-based tool access. |
| Guardrails | Applies input sanitization, tool allowlists, deterministic output checks, and risk framing. |
| Browser UI | Provides a local dashboard and streaming chat interface. |

---

## System Architecture

RAPHI runs through a primary FastAPI application exposed by `backend/raphi_server.py`.

```text
Browser UI
  |
  |  /api/chat, /api/memo, /api/stock, /api/portfolio
  v
Primary RAPHI Server
backend/raphi_server.py
  |
  |-- Market Data Layer
  |     backend/market_data.py
  |
  |-- SEC Filing Layer
  |     backend/sec_data.py
  |
  |-- ML Signal Layer
  |     backend/ml_model.py
  |
  |-- GNN Signal Layer
  |     backend/gnn_model.py
  |
  |-- Portfolio Risk Layer
  |     backend/portfolio_manager.py
  |
  |-- Conviction Ledger
  |     backend/conviction_store.py
  |
  |-- Memory Layer
  |     backend/graph_memory.py
  |
  |-- Guardrails
  |     backend/security.py
  |     backend/llm_guardrails.py
  |
  |-- Agent Execution
        backend/a2a_executor_v2.py
        backend/raphi_mcp_server.py
```

---

## Runtime Surfaces

### Primary Runtime

```text
server.js
backend/raphi_server.py
```

The primary runtime starts RAPHI on port `9999`.

It serves:

- Browser dashboard
- Streaming chat
- Market data APIs
- Portfolio APIs
- Signal APIs
- SEC filing APIs
- Conviction ledger APIs
- Memory APIs
- A2A/MCP-backed agent routes

Start it with:

```bash
node server.js
```

Then open:

```text
http://localhost:9999
```

---

### Secondary Development Runtime

```text
backend/main.py
```

The secondary runtime is used for isolated development, especially direct GNN testing.

Start it with:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Primary application work should happen through `backend/raphi_server.py`.

---

## Agent and Tool Execution

RAPHI includes a real agent/tool execution path.

The main components are:

```text
backend/a2a_executor_v2.py
backend/raphi_mcp_server.py
```

The A2A executor manages:

- Tool allowlisting
- Session encryption
- Prompt sanitization
- Agent execution flow

The MCP bridge exposes tools for:

- Market data
- SEC filings
- ML signals
- Portfolio state
- Memory retrieval

The browser chat first attempts the A2A agent stream. If that path does not return usable text, RAPHI falls back to a local evidence pass that combines market, SEC, ML, GNN, portfolio, and memory context before generating the final response.

---

## Memory System

RAPHI supports durable memory.

Primary memory backend:

```text
Neo4j
```

Fallback memory backend:

```text
.raphi_memory/
```

Entry point:

```text
backend/graph_memory.py
```

If Neo4j is unavailable, RAPHI uses local JSON-backed memory instead of disabling memory completely. This keeps local development usable without requiring a database for every run.

---

## Machine Learning and Signal Generation

RAPHI includes multiple signal layers.

### Traditional ML

Implemented in:

```text
backend/ml_model.py
```

Uses:

- XGBoost
- Gradient Boosting
- scikit-learn
- pandas
- NumPy
- SHAP-style explanations

### Graph Signal Engine

Implemented in:

```text
backend/gnn_model.py
```

Supports:

- GraphSAGE-style modeling
- Cached graph state
- Optional PyTorch Geometric backend
- Blended signal generation when graph artifacts are available

Cached artifacts are stored under:

```text
.model_cache/
```

---

## Portfolio and Risk Analytics

Implemented in:

```text
backend/portfolio_manager.py
```

RAPHI supports:

- Portfolio snapshots
- Position management
- VaR estimation
- Sharpe ratio calculation
- Portfolio alerts
- Signal-aware portfolio review

Portfolio state is stored locally through JSON-backed state files.

---

## Guardrails and Safety Controls

RAPHI includes both input-side and output-side controls.

### Input and Execution Controls

Implemented across:

```text
backend/security.py
backend/a2a_executor_v2.py
backend/raphi_mcp_server.py
```

Includes:

- User input sanitization
- MCP internal token protection
- Explicit tool allowlist
- API key authentication
- Rate limiting
- Session encryption for A2A mappings

### Output Controls

Implemented in:

```text
backend/llm_guardrails.py
```

Includes:

- Deterministic post-generation checks
- Memo schema repair
- Overconfidence softening
- Risk framing enforcement
- Unknown ticker detection
- Provenance notes

The goal is not to make the model “always right.” The goal is to make unsupported, overconfident, or poorly framed outputs easier to detect and constrain.

---

## Streaming, Caching, and Performance

RAPHI supports streaming responses for chat and memo generation.

Implemented today:

- SSE streaming for chat responses
- SSE streaming for memo responses
- Market data TTL caches
- Fundamental data TTL caches
- News TTL caches
- Local model artifact cache
- GNN state cache

Relevant paths:

```text
backend/market_data.py
.model_cache/
.model_cache/gnn_state.pkl
```

Not implemented today:

- Direct LLM KV cache control
- Explicit Anthropic prompt caching policy

Practical implication:

RAPHI can reuse local data/model artifacts and stream responses quickly, but it does not currently manage provider-level LLM KV caching.

---

## Browser Chat Threading

The browser UI supports fresh conversation threads.

The `New Thread` action:

- Clears local chat history
- Rotates the persistent `thread_id`
- Aborts any in-flight chat stream
- Prevents stale responses from leaking into the next thread
- Resets the console state

This matters because financial analysis conversations often carry ticker, portfolio, and thesis context that should not accidentally bleed across sessions.

---

## Tech Stack

### Backend

- Python
- FastAPI
- Starlette
- Uvicorn
- Pydantic
- SlowAPI
- Server-Sent Events

### Agent and LLM Layer

- Anthropic SDK
- Claude Agent SDK
- A2A protocol server
- MCP server bridge

### Data and Finance

- yfinance
- Local SEC EDGAR datasets
- XBRL parsing
- Filing search

### Machine Learning

- scikit-learn
- XGBoost
- SHAP
- pandas
- NumPy
- Optional PyTorch Geometric support

### Memory and Persistence

- Neo4j
- Local JSON fallback memory
- Pickle-based model cache
- JSON / JSONL state files
- Fernet-encrypted session mapping

### Frontend

- Single-file browser application
- HTML
- CSS
- JavaScript
- Fetch API
- Streaming response handling

### Operations

- API key authentication
- Internal MCP token authentication
- Prompt sanitization
- Rate limiting
- Sentry support

### Testing

- pytest

---

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

---

## Local Setup

### 1. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Configure environment variables

Create a `.env` file at the project root.

Common variables:

```bash
RAPHI_API_KEY=
ANTHROPIC_API_KEY=
RAPHI_INTERNAL_TOKEN=

NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=

SENTRY_DSN=
```

Neo4j is optional. RAPHI can fall back to local JSON memory when Neo4j is not configured.

### 4. Start the primary application

```bash
node server.js
```

Open:

```text
http://localhost:9999
```

---

## Optional Neo4j Setup

To use Neo4j as the primary graph memory backend:

```bash
export NEO4J_PASSWORD='choose-a-local-password'
docker-compose -f docker-compose.neo4j.yml up -d
```

Then make sure the following variables match your local Neo4j configuration:

```bash
NEO4J_URI=
NEO4J_USER=
NEO4J_PASSWORD=
```

If Neo4j is offline or unavailable, RAPHI uses local memory fallback.

---

## Running Tests

```bash
source .venv/bin/activate
pytest
```

Current test coverage includes:

- Conviction ledger behavior
- Graph memory behavior
- SEC data path behavior
- Ticker mapping behavior

---

## Key API Surfaces

### Health

```http
GET /api/health
```

### Market and Stock Data

```http
GET /api/market/overview
GET /api/stock/{ticker}
GET /api/stock/{ticker}/news
GET /api/stock/{ticker}/signals
GET /api/stock/{ticker}/filings
```

### Portfolio

```http
GET /api/portfolio
PUT /api/portfolio
GET /api/signals
GET /api/alerts
```

### Chat and Memo Generation

```http
POST /api/chat
POST /api/memo/{ticker}
```

### Conviction Ledger

```http
GET /api/convictions/stats
GET /api/convictions/ledger
```

### Memory

```http
GET /api/memory/status
GET /api/memory/retrieve
```

### Secondary Development API

Available through `backend/main.py`:

```http
GET /api/stock/{ticker}/gnn
POST /api/gnn/train
GET /api/gnn/status
```

---

## Current Implementation Notes

- `backend/raphi_server.py` is the authoritative application surface.
- `backend/main.py` is retained for development and direct GNN testing.
- The GNN engine is implemented and can be blended into signal generation when cached graph artifacts exist.
- Memory remains available without Neo4j through local fallback storage.
- The system supports streaming responses, but does not currently implement application-managed LLM KV caching.
- The agent path is implemented through A2A execution and MCP tool exposure, with a deterministic fallback path for evidence-based response generation.

---

## Limitations

RAPHI is still a local-first research and engineering system. Current limitations include:

- No direct LLM KV cache management.
- No explicit Anthropic prompt-caching policy.
- Local SEC data depends on the datasets available under `data/`.
- The GNN signal layer depends on available cached graph state.
- Financial outputs should be treated as decision-support analysis, not investment advice.
- No license file is currently included in the repository.

---
