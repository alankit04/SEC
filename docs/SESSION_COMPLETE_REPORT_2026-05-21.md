# RAPHI Complete Session Report

Date: 2026-05-21
Repository: SEC
Branch: main

## 1) Executive Summary

This session delivered full production implementation and validation of the requested performance and efficiency improvements for RAPHI.

Primary outcomes:

- Practical latency improvements were implemented in backend orchestration paths.
- Context compaction was implemented for token and cost reduction.
- A centralized versioned tool-result cache was implemented and integrated across expensive MCP tools.
- Cache behavior includes TTL, stale grace, single-flight deduping, and stale-on-error serving.
- Cache invalidation was implemented for model-changing operations.
- End-to-end validation passed with full tests and a live A/B probe showing strong warm-cache speedups.

Current status at report time:

- Test suite: 90 passed, 2 warnings, 0 failed.
- Live cache A/B probe: average speedup 85.18% across tested tools.

## 2) Goals Addressed

The work addressed all requested themes from the session:

- Run and validate real performance behavior, not placeholders.
- Improve practical latency.
- Reduce token usage with compaction mode.
- Evaluate and implement layered caching with tool-result cache priority.
- Implement fully without scaffolding code.

## 3) Core Architecture Changes

### 3.1 Centralized Tool Result Cache

Implemented in backend/tool_result_cache.py.

Capabilities:

- Stable cache key composition:
  - tool name
  - normalized arguments hash
  - data version
  - model version
  - user scope
- Freshness lifecycle:
  - fresh
  - stale
  - expired
- TTL and stale-grace support per request.
- Single-flight per key to prevent concurrent stampedes.
- Stale-on-error fallback to preserve resilience.
- Tool-specific invalidation and global invalidation operations.
- Metadata emitted per record for observability:
  - cache_hit
  - freshness_state
  - age_ms
  - source
  - version fields

### 3.2 MCP Integration of Cache Layer

Implemented in backend/raphi_mcp_server.py.

Added:

- Global cache instance with default TTL and stale-grace env controls.
- Tool-specific TTL mapping and source mapping.
- Data/model version derivation helpers using:
  - environment version tags for remote providers
  - file mtimes for local data/model artifacts
- Scope strategy for user-scoped tools.
- Generic cached wrapper for tool JSON producers.
- Optional cache metadata exposure gate.

Cache integrated for the following expensive tools:

- market_overview
- stock_detail
- stock_news
- sec_filings
- sec_search
- sec_universe
- sec_industries
- ml_signal
- gnn_signal
- gnn_status
- portfolio_snapshot
- portfolio_alerts
- memory_status
- memory_retrieve
- edgar_live_filings
- edgar_search_fulltext
- firecrawl_scrape
- firecrawl_search
- web_citations

Invalidation behavior:

- gnn_train now invalidates gnn_status, gnn_signal, and ml_signal cache groups.

### 3.3 Latency and Compaction in Chat/Agent Flow

Implemented across backend/raphi_server.py and backend/a2a_executor_v2.py.

Key changes:

- Compact mode and prompt/history/memory compaction helpers.
- Reduced tool/turn overhead in compact path.
- Parallelized pre-chat I/O gathers to reduce serial waiting.
- Lightweight market prefetch TTL cache in server path.
- Lower turn budget in executor for faster completions.
- Role-scoped MCP tool permissions in executor.

### 3.4 Runtime Visibility and Governance Extensions

Added production modules:

- backend/eval_harness.py
- backend/eval_logger.py
- backend/release_gates.py
- backend/governance.py
- backend/provider_controls.py

Integrated with chat/memo streaming:

- Run IDs and tool traces.
- Eval scoring and quality metrics in run summaries.
- Optional review queue flow for high-risk outputs.
- Provider circuit-breaker health handling.

Frontend update in backend/static/index.html:

- Added live run trace panel rendering for plan/tools/guardrail/eval visibility.

## 4) Validation and Testing

### 4.1 Automated Tests

Test status after implementation:

- 90 passed
- 2 warnings
- 0 failed

Important test additions/updates:

- tests/test_tool_result_cache.py added:
  - key stability
  - cache hit behavior
  - concurrent single-flight dedupe
  - stale-on-error fallback
- tests/test_raphi_server_gnn.py updated for strict ticker validation compliance.
- Additional eval/governance/provider/release-gates tests added:
  - tests/test_eval_harness.py
  - tests/test_eval_logger.py
  - tests/test_governance.py
  - tests/test_provider_controls.py
  - tests/test_release_gates.py

### 4.2 Live A/B Cache Probe

Probe method:

- Repeated identical /mcp calls for cold then warm path.
- Measured elapsed_ms for each phase.
- Captured cache metadata when surfaced in response payload.

Results:

| Tool | Cold ms | Warm ms | Speedup |
| --- | ---: | ---: | ---: |
| stock_detail | 866.74 | 8.13 | 99.06% |
| stock_news | 271.81 | 15.74 | 94.21% |
| sec_filings | 8389.96 | 6.30 | 99.92% |
| web_citations | 10.90 | 5.72 | 47.52% |

Aggregate:

- avg_cold_ms: 2384.85
- avg_warm_ms: 8.97
- avg_speedup_percent: 85.18

Health endpoint clarification captured during probe:

- /health returned 404 in this app layout.
- /api/health is the correct route and returned 200.

## 5) Changed Files Summary

Major files touched in this session:

- backend/tool_result_cache.py
- backend/raphi_mcp_server.py
- backend/raphi_server.py
- backend/a2a_executor_v2.py
- backend/eval_harness.py
- backend/eval_logger.py
- backend/release_gates.py
- backend/governance.py
- backend/provider_controls.py
- backend/static/index.html
- scripts/run_eval_harness.py
- scripts/run_release_gates.py
- tests/test_tool_result_cache.py
- tests/test_raphi_server_gnn.py
- tests/test_raphi_server_chat_agentic.py
- tests/test_eval_harness.py
- tests/test_eval_logger.py
- tests/test_governance.py
- tests/test_provider_controls.py
- tests/test_release_gates.py

## 6) Operational Notes

- The backend server process on port 9999 was verified as:
  - .venv/bin/uvicorn backend.raphi_server:app --host 127.0.0.1 --port 9999 --log-level warning
- Cache metadata exposure for probing used:
  - RAPHI_CACHE_EXPOSE_META=1

## 7) Outcome

Requested implementation is complete for this session scope:

- practical latency improvements: complete
- token-saving compaction: complete
- layered central tool-result cache: complete
- production integration and invalidation: complete
- tests and live benchmark evidence: complete

Saved artifacts:

- This report: docs/SESSION_COMPLETE_REPORT_2026-05-21.md
- Probe snapshot: docs/CACHE_AB_PROBE_2026-05-21.json
