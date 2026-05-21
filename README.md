# RAPHI: Agentic AI Research System for SEC Intelligence with A2A/MCP Tooling

RAPHI is a local-first agentic AI research system for SEC and financial intelligence. It combines A2A/MCP tool orchestration, Claude Agent SDK execution, SEC EDGAR retrieval, market data, local model signal context, portfolio exposure, durable thesis memory, citation indexing, and deterministic guardrails into one explanation-driven workflow.

> **What it does**
>
> RAPHI plans financial research tasks, routes them through specialist tools, retrieves structured evidence, preserves useful context in memory, and generates sourced research memos with explicit uncertainty, citations, and risk framing. SEC research is the first serious domain implementation; the broader architecture is an agentic AI research workflow, not a simple chatbot.

> **What it does not do**
>
> - RAPHI is not a trading bot.
> - RAPHI does not execute live trades.
> - RAPHI does not guarantee financial correctness.
> - RAPHI does not provide licensed investment advice.
> - RAPHI does not fine-tune, distill, or quantize hosted LLMs.
> - RAPHI does not replace human review of filings, sources, risk, or portfolio decisions.

## Why This Exists

Agentic AI systems are most useful when they can plan, call tools, preserve memory, check evidence, and expose what happened. RAPHI applies that pattern to SEC and financial research, where unsupported claims are especially risky.

The project is intended as an applied AI infrastructure prototype for agent/tool interoperability, citation memory, specialist routing, local model context, durable thesis tracking, and evaluation-oriented memo generation. Its current domain is SEC EDGAR retrieval, XBRL financial extraction, market data, portfolio context, and citation-backed financial research workflows.

**Principle:** RAPHI separates evidence collection from synthesis. Tools retrieve structured data first; the LLM is used to reason over retrieved evidence, not to invent facts.

## How It Works

```text
Ticker / thesis
    ↓
SEC EDGAR + XBRL retrieval
    ↓
Market data + news context
    ↓
Local model signal context
    ↓
Portfolio exposure and risk context
    ↓
Durable memory retrieval
    ↓
A2A/MCP tool orchestration
    ↓
Guardrailed memo generation
    ↓
Evaluation: citations, unsupported claims, schema, routing
```

## Architecture Overview

RAPHI has three layers:

- **Architecture layer:** agentic AI orchestration through A2A/MCP tooling, Claude Agent SDK execution, specialist tool routing, durable memory, and deterministic guardrails.
- **Domain layer:** SEC EDGAR retrieval, XBRL financial extraction, market/news context, portfolio exposure, local signal context, citation-backed memo generation, and financial research workflows.
- **Evaluation layer:** reliability posture around citation precision, unsupported-claim rate, memo-schema compliance, tool-routing accuracy, and guardrail repair behavior.

- **Unified server:** `backend/raphi_server.py` exposes the browser UI, FastAPI routes, A2A entry point, chat, memo, memory, citation, SEC, market, model, GNN, and portfolio surfaces.
- **SEC retrieval:** `backend/sec_data.py` reads local SEC/XBRL datasets; `backend/edgar_live.py` adds live EDGAR submissions and full-text filing search.
- **Citation memory:** `backend/citation_index.py` stores citation sources in Postgres when configured, with SQLite FTS5 fallback; `backend/web_citations.py` searches the local index first.
- **Optional web refresh:** `backend/firecrawl_client.py` is used only as an ingestion or refresh tool when configured.
- **Agent orchestration:** `backend/a2a_executor_v2.py` routes agent requests through allowed MCP tools; `backend/raphi_mcp_server.py` exposes those tools.
- **Local signal context:** `backend/ml_model.py`, `backend/gnn_model.py`, and `backend/model_optimization.py` provide research signals, graph context, and local signal-layer optimization artifacts.
- **Memory and track record:** `backend/graph_memory.py` provides durable thesis memory; `backend/conviction_store.py` stores and resolves research convictions.
- **Guardrails:** `backend/security.py` and `backend/llm_guardrails.py` apply authentication, prompt sanitization, deterministic response repair, and risk framing.
- **Browser client:** `backend/static/index.html` is the local dashboard and chat interface.

## Current Implementation Status

| Capability | Purpose | Evidence in repo / file path | Status |
|---|---|---|---|
| Unified FastAPI/A2A server | Single local runtime for API, dashboard, chat, memo, A2A, and tool bridge | `backend/raphi_server.py`, `server.js` | Implemented |
| Browser dashboard and chat | Local UI for research workflows and smoke-tested navigation | `backend/static/index.html`, `scripts/e2e-smoke.mjs` | Implemented |
| Local SEC/XBRL retrieval | Query local SEC datasets and metric-level filing citations | `backend/sec_data.py`, `tests/test_sec_data_paths.py` | Implemented |
| Live EDGAR retrieval | Retrieve recent company submissions and EDGAR full-text search results | `backend/edgar_live.py`, routes in `backend/raphi_server.py` | Implemented |
| Market and news context | Fetch price, overview, history, and news context through provider APIs | `backend/market_data.py` | Implemented, provider-dependent |
| Citation index | Store and search durable citation evidence before using live web refresh | `backend/citation_index.py`, `backend/web_citations.py`, `tests/test_citation_index.py`, `tests/test_web_citations.py` | Implemented |
| Firecrawl ingestion | Optional source refresh and page ingestion when a local citation gap exists | `backend/firecrawl_client.py`, `backend/raphi_mcp_server.py` | Implemented, optional |
| Local model signal context | Provide research-context signals from tabular and temporal tree-based features | `backend/ml_model.py` | Implemented as research context |
| GNN graph context | Model peer and correlation relationships for graph-based signal context | `backend/gnn_model.py`, `tests/test_gnn_model.py`, `tests/test_raphi_server_gnn.py` | Implemented |
| Local signal optimization | Contextual bandit policy, student distillation, and int8 quantization for local signal artifacts | `backend/model_optimization.py`, `tests/test_model_optimization.py` | Implemented for local signal layer only |
| Portfolio exposure context | Add portfolio holdings, exposure, and risk metrics to research answers | `backend/portfolio_manager.py`, `tests/test_raphi_server_live_surfaces.py` | Implemented |
| Durable thesis memory | Store and retrieve prior discussion facts with Neo4j or local fallback | `backend/graph_memory.py`, `tests/test_graph_memory.py` | Implemented |
| A2A/MCP orchestration | Route requests through allowlisted specialist tools | `backend/a2a_executor_v2.py`, `backend/raphi_mcp_server.py`, `tests/test_agent_gnn_tools.py` | Implemented |
| Deterministic guardrails | Sanitize inputs and repair output for unsupported claims, memo shape, and risk framing | `backend/security.py`, `backend/llm_guardrails.py`, `tests/test_llm_guardrails.py` | Implemented |
| Conviction ledger | Persist research calls and evaluate resolved outcomes over time | `backend/conviction_store.py`, `tests/test_conviction_store.py` | Implemented |
| Formal eval harness | Measure citation precision, unsupported-claim rate, schema compliance, and routing accuracy | Current tests cover pieces; no unified eval harness file yet | Planned |
| Live trade execution | Place orders or execute trades | No broker integration in repo | Not implemented by design |

## Key Features

- Local-first SEC research over EDGAR filings, XBRL facts, accession numbers, filing dates, and SEC URLs.
- Citation memory that searches saved sources before optional web refresh.
- Agentic routing through A2A and MCP tools rather than a single unconstrained prompt.
- Guardrailed memo generation with explicit risk framing and source expectations.
- Portfolio-aware responses that can account for current local holdings.
- Durable memory for prior research context, with Neo4j primary storage and local fallback.
- Local signal-layer context from tabular models, graph relationships, and resolved conviction outcomes.
- Browser UI plus API endpoints for reproducible local testing.

## Guardrails And Safety

RAPHI uses deterministic guardrails around the LLM and tool layer:

- API-key and internal-token checks in `backend/security.py`.
- Prompt-injection sanitization in `backend/security.py` and `backend/a2a_executor_v2.py`.
- MCP tool allowlisting in `backend/a2a_executor_v2.py`.
- Response repair and risk framing in `backend/llm_guardrails.py`.
- Memo schema checks for generated research memos.
- Citation and provenance checks in chat and memo paths.
- Rate limiting and secret scrubbing support in the server stack.

These guardrails reduce avoidable failure modes, but they do not prove that a financial conclusion is correct.

## Evaluation Strategy

RAPHI is designed around an evaluation-first workflow. The next implementation step is a formal eval harness for citation precision, unsupported-claim rate, memo-schema compliance, and tool-routing accuracy.

Implemented validation today includes:

- Tool registration and allowlist tests: `tests/test_agent_gnn_tools.py`.
- Chat routing, local fallback, identity handling, and guardrail behavior: `tests/test_raphi_server_chat_agentic.py`.
- Deterministic LLM guardrails: `tests/test_llm_guardrails.py`.
- SEC pathing and filing citation behavior: `tests/test_sec_data_paths.py`.
- Citation index and local-first web citation behavior: `tests/test_citation_index.py`, `tests/test_web_citations.py`.
- GNN behavior and API coverage: `tests/test_gnn_model.py`, `tests/test_raphi_server_gnn.py`.
- Local signal optimization behavior: `tests/test_model_optimization.py`.
- Portfolio, memo export, model status, and live API surfaces: `tests/test_raphi_server_live_surfaces.py`.
- Browser workflow smoke tests: `scripts/e2e-smoke.mjs`.

Last local validation observed for this repo: `61 passed, 2 warnings`.

## Setup And Run

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Create a local `.env` file with the values you want enabled:

```bash
RAPHI_API_KEY=local-development-key
RAPHI_INTERNAL_TOKEN=local-internal-token
ANTHROPIC_API_KEY=your_anthropic_key

# Optional integrations
FIRECRAWL_API_KEY=
CITATION_DATABASE_URL=
NEO4J_URL=
NEO4J_USER=
NEO4J_PASSWORD=
SENTRY_DSN=
FIGMA_ACCESS_TOKEN=
FIGMA_FILE_KEY=
```

Run the primary local server:

```bash
node server.js
```

If you want to run FastAPI directly:

```bash
set -a
source .env
set +a
.venv/bin/uvicorn backend.raphi_server:app --host 127.0.0.1 --port 9999
```

Open:

```text
http://localhost:9999
```

## API Surfaces

Representative local API routes include:

- Health and settings: `GET /api/health`, `GET /api/settings`, `POST /api/settings`.
- Market data: `GET /api/market/overview`, `GET /api/stock/{ticker}`, `GET /api/stock/{ticker}/news`.
- SEC data: `GET /api/stock/{ticker}/filings`, `GET /api/stock/{ticker}/live-filings`, `GET /api/edgar/search`.
- Citation memory: `GET /api/citations/status`, `POST /api/citations/index`, `POST /api/citations/sec/{ticker}/index`, `GET /api/citations/search`, `POST /api/web/citations`.
- Optional Firecrawl refresh: `POST /api/firecrawl/scrape`, `POST /api/firecrawl/search`.
- Model and graph context: `GET /api/stock/{ticker}/signals`, `GET /api/stock/{ticker}/gnn`, `GET /api/gnn/status`, `POST /api/gnn/train`.
- Portfolio and risk: `GET /api/portfolio`, `PUT /api/portfolio`, `POST /api/portfolio/positions`.
- Cross-asset and alerts: `GET /api/signals`, `GET /api/cross-asset/signals`, `GET /api/alerts`.
- Model optimization status: `GET /api/models/optimization`, `POST /api/models/rl/update`, `GET /api/stock/{ticker}/optimization`.
- Conviction ledger: conviction routes exposed from `backend/raphi_server.py`.
- Memory: graph memory routes exposed from `backend/raphi_server.py`.
- Agent and memo generation: `POST /api/chat`, `POST /api/memo/{ticker}`, `GET /api/memo/{ticker}/export`.
- MCP bridge: `POST /mcp`.

## Testing

Run the Python test suite:

```bash
source .venv/bin/activate
pytest
```

Run the browser smoke test against a running local server:

```bash
npm run test:e2e
```

The tests validate behavior across retrieval, citations, guardrails, agent routing, memory, graph context, portfolio surfaces, and UI navigation. They are not a substitute for a formal financial correctness evaluation.

## Known Limitations

- RAPHI is a local-first prototype, not a hosted production service.
- Financial outputs depend on provider availability and data quality.
- Model signals are research context, not trade recommendations.
- Guardrails reduce overconfidence but do not prove correctness.
- The formal evaluation harness is partial and pending.
- Hosted LLM inference is provider-dependent and requires a valid API key for full responses.
- Optional integrations such as Firecrawl, Neo4j, Postgres, Figma, and Sentry only run when configured.
- No live trading execution is implemented.
- No licensed investment advice is provided.

## Roadmap

- Add a formal eval harness for citation precision, unsupported-claim rate, memo-schema compliance, and tool-routing accuracy.
- Add golden SEC research tasks with expected citations and failure labels.
- Expand citation indexing jobs for company IR pages, earnings transcripts, and press releases.
- Improve retrieval ranking across SEC filings, local citation memory, and optional web refresh.
- Add clearer UI evidence panels for tool calls, retrieved citations, and guardrail repairs.
- Replace any remaining legacy UI copy that implies unsupported model types or trading guarantees.
- Add deployment hardening for secrets, observability, and reproducible demo environments.

## License Note

No investment advice is provided by this repository. Review the project license before using or redistributing the code. Any financial research generated by RAPHI should be independently verified against primary sources and reviewed by a qualified human decision-maker.
