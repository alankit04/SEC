# Conviction Ledger — Design Specification
**Date:** 2026-04-08
**Status:** Approved for implementation
**Feature:** RAPHI Conviction Ledger — self-evaluating research accuracy tracking

---

## Executive Summary

The Conviction Ledger tracks every research output RAPHI generates and checks it against reality at 30, 60, and 90-day windows. Over time RAPHI builds a per-signal, per-ticker, per-regime accuracy record that is surfaced inline on every research output and on a dedicated dashboard page. No AI financial research tool in 2026 tracks its own accuracy. This is how RAPHI becomes defensible, not just useful.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| What to track | ML signal direction + SEC revenue trend + Signal View (sentiment implicit via Signal View) | All three are automatable and verifiable. Sentiment excluded from standalone scoring — VADER is weak and scoring it separately risks eroding trust in the ledger. |
| "Proved true" for ML | Price direction vs entry price at 30/60/90 calendar days | Unambiguous. Fully automatable via yfinance. |
| "Proved true" for SEC | Next EDGAR XBRL filing confirms/contradicts revenue trend direction (±3% noise band) | Authoritative source. Automatable from local XBRL files already in RAPHI. |
| "Proved true" for Signal View | Stock return vs SPY at 30/60/90 days | Risk-adjusted. Removes market beta from the accuracy signal. |
| Storage approach | Append-only JSONL (two files) | Immutable audit trail. Fits existing `financial_decisions.jsonl` pattern. No new dependencies. Migration to SQLite is a one-time script when query complexity demands it. |
| Where shown | Both: compact inline on every research output + dedicated Conviction Ledger page | Right information at the right moment. Ledger page is built first (forces correct data model). Inline is trivial once data exists. |
| Lookback trigger | `checkPendingConvictions()` in existing `liveRefresh()` Promise.all | Zero new infrastructure. Fits current architecture. Nightly scheduled job added as second trigger in a future production deployment. |

---

## Section 1 — Data Model

### Two files, both append-only, never mutated

**Location:** `.raphi_audit/conviction_ledger/` (existing chmod 700 directory)

```
.raphi_audit/
  conviction_ledger/
    convictions.jsonl    ← one line per research output, write-once
    resolutions.jsonl    ← one line per resolved lookback window, append-only
```

### convictions.jsonl schema

One JSON object per line. Written once. Never modified.

```json
{
  "id": "cvx-20260408-NVDA-a3f",
  "ticker": "NVDA",
  "date": "2026-04-08T14:32:11Z",
  "entry_price": 890.50,
  "ml": {
    "direction": "LONG",
    "probability": 0.71,
    "model_version": "xgb_v2.1"
  },
  "sec": {
    "trend": "accelerating",
    "latest_revenue": 44200000000,
    "quarters_used": 8,
    "next_filing_due": "2026-07-15"
  },
  "signal_view": "Positive",
  "conviction": "MEDIUM",
  "source": "memo",
  "vix_at_creation": 15.8,
  "lookbacks_due": {
    "30d": "2026-05-08",
    "60d": "2026-06-07",
    "90d": "2026-07-07"
  }
}
```

**Field notes:**
- `id` format: `cvx-{YYYYMMDD}-{TICKER}-{rand3}` — human-readable, unique, sortable by date
- `source`: `"memo"` | `"signal_query"` | `"chat"` — tracks what generated the conviction. A `"chat"` conviction is only written when a chat response includes an explicit ticker + ML direction + Signal View conclusion (i.e. the agent called `ml_signal` and `sec_filings` tools and returned a structured research conclusion). General chat without a structured research output does not create a conviction.
- `vix_at_creation`: VIX closing value fetched from `yf.Ticker("^VIX")` at `write_conviction()` call time. Used for regime breakdown on the ledger page. Stored at creation, not resolution, so each conviction belongs to exactly one VIX regime.
- `sec.next_filing_due`: projected from last known filing date by sec-researcher. `null` if unavailable — SEC resolution is permanently skipped for this conviction
- `lookbacks_due`: calendar days, not trading days

### resolutions.jsonl schema

One JSON object per line. One line per resolved lookback window. Up to 4 per conviction: `30d`, `60d`, `90d`, `sec`.

```json
// ML price check (30d, 60d, or 90d)
{
  "conviction_id": "cvx-20260408-NVDA-a3f",
  "lookback": "30d",
  "resolved_date": "2026-05-08",
  "ml_result": "CONFIRMED",
  "price_at_check": 962.30,
  "vs_entry_pct": 8.1,
  "vs_spy_pct": 6.8,
  "vix_at_check": 16.2
}

// SEC filing check
{
  "conviction_id": "cvx-20260408-NVDA-a3f",
  "lookback": "sec",
  "resolved_date": "2026-07-16",
  "sec_result": "CONFIRMED",
  "actual_revenue": 49800000000,
  "revenue_delta_pct": 12.7
}
```

**Field notes:**
- `vix_at_check`: stored on every ML resolution for future regime analysis without reprocessing
- `ml_result` and `sec_result` values: `"CONFIRMED"` | `"CONTRADICTED"` | `"INCONCLUSIVE"` (SEC only, within ±3% noise band)
- `INCONCLUSIVE` is excluded from accuracy denominator — does not count for or against
- Deduplication: `(conviction_id, lookback)` pair is checked before any resolution is written — safe to call repeatedly

---

## Section 2 — Backend

### New file: `backend/conviction_store.py`

Single responsibility: all reads and writes to the two JSONL files. Nothing else touches them directly.

#### Functions

**`write_conviction(ticker, ml, sec, signal_view, source, entry_price) → str`**

Called after every validated research output from `a2a_executor_v2.py`. Builds the conviction object, computes lookback due dates, appends to `convictions.jsonl`. Returns `conviction_id`.

**`check_pending() → dict`**

Called by `GET /api/convictions/check`. Algorithm:
1. Read `convictions.jsonl` → `dict[id → conviction]`
2. Read `resolutions.jsonl` → `set[(conviction_id, lookback)]` of already-resolved pairs
3. For each unresolved (conviction, window) pair where `today >= due_date`:
   - For ML windows: fetch price via `yf.Ticker(ticker).history(period="5d")`, fetch SPY and VIX same window, compute result, append to `resolutions.jsonl`
   - For SEC window: check local XBRL data for post-conviction quarter, compare revenue, append result
4. Return `{resolved: int, still_pending: int, errors: list[str]}`

**`get_accuracy_stats(ticker=None, lookback=None) → AccuracyStats`**

Joins both files by `conviction_id`. Computes accuracy by signal type, lookback window, and ticker. Denominator rule: only resolved windows count. Pending windows are excluded from both numerator and denominator.

**`get_ledger(page=1, ticker=None) → LedgerPage`**

Full conviction history joined with resolutions. Sorted by date descending. Paginated at 50 per page. Returns per-conviction status per lookback window.

#### Error handling rules

| Condition | Behaviour |
|---|---|
| yfinance fails during ML check | Skip conviction this cycle, add to `errors[]`, retry next liveRefresh |
| EDGAR data not yet updated for SEC check | Skip, `still_pending++`, retry |
| `(conviction_id, lookback)` already in resolutions | Skip. Fully idempotent. |
| File write error | Log to Sentry. Return error in response. UI shows "ledger unavailable" gracefully. |
| One agent output fails schema validation | Field set to `null` in conviction. `source` set to `"partial_memo"`. That specific resolution type permanently skipped. |

### New API endpoints in `raphi_server.py`

All require auth (X-API-Key). Rate limited at 60/min (data tier).

```
POST /api/convictions
  → write_conviction() from validated agent outputs
  → returns { conviction_id }

GET  /api/convictions/check
  → check_pending()
  → returns { resolved, still_pending, errors }
  → idempotent, safe to call on every liveRefresh cycle

GET  /api/convictions/stats?ticker=NVDA
  → get_accuracy_stats(ticker)
  → ticker param optional — omit for portfolio-wide stats
  → powers both ledger page aggregate view and inline compact line

GET  /api/convictions/ledger?page=1&ticker=NVDA
  → get_ledger(page, ticker)
  → full history for ledger page, paginated at 50/page
```

### Integration points in existing code

**`a2a_executor_v2.py`** — after schema validation passes on research completion:
```python
# After all agent outputs validated:
conviction_id = write_conviction(
    ticker=ticker,
    ml=ml_output,
    sec=sec_output,
    signal_view=memo.signal_view,
    source="memo",
    entry_price=market_output.price
)
# conviction_id included in audit log entry
```

**`index.html` `liveRefresh()`** — two additions to existing Promise.all:
```js
Promise.all([
    loadMarketMetrics(sig),
    loadDashboardSignals(sig),
    loadWatchlistPrices(sig),
    loadConvictionStats(sig),      // ← new: writes sidebar badge + stat cards
    resolveConvictionsPoll(sig),   // ← new: fires /api/convictions/check silently
])
```

---

## Section 3 — Frontend

### Dedicated Conviction Ledger page

Added to sidebar navigation after Decision Memo. Uses stacked-layers SVG icon. `nav-badge` with id `nav-pending-count` appears only when pending count > 0.

#### Top aggregate stats bar (7 cards)

| Card | ID | Colour rule |
|---|---|---|
| ML accuracy 30d | `cl-stat-ml-30` | ≥65% green · 50–64% amber · <50% red |
| ML accuracy 60d | `cl-stat-ml-60` | same |
| ML accuracy 90d | `cl-stat-ml-90` | same |
| SEC trend accuracy | `cl-stat-sec` | same |
| Signal View vs SPY | `cl-stat-spy` | always purple (excess return %) |
| Total convictions | `cl-stat-total` | neutral white |
| Pending count | `cl-stat-pending` | always amber — shows urgency |

#### Conviction history table

Columns: Date · Ticker · ML Signal · SEC Rev Trend · Signal View · 30d · 60d · 90d · Source

Cell rendering per lookback window:
- **Confirmed correct:** green chip `✓ +11.4%`, `rgba(60,225,181,0.1)` background
- **Contradicted:** red chip `✗ +6.3%`, `rgba(255,90,122,0.1)` background, faint red row background
- **Pending (nearest window):** amber chip `⏳ Apr 30`
- **Pending (further windows):** muted gray due-date text

Filters: three `<select>` dropdowns — All Tickers / specific, All Signal Types / ML / SEC / Signal View, All Windows / 30d / 60d / 90d. Each `onchange` re-calls `loadConvictionLedger()` with updated params.

#### Regime breakdown section

Three cards below the table: VIX < 15 (green) · VIX 15–25 (amber) · VIX > 25 (red). Each shows ML accuracy, SEC accuracy, excess return vs SPY for convictions grouped by `vix_at_creation` (the VIX level when the conviction was written). Each conviction belongs to exactly one regime. Powered by `get_accuracy_stats()` which groups by `vix_at_creation` ranges.

### Inline compact view

Appears as a 1-line footer strip immediately below the Signal View section on every memo output and signal card. Rendered by `loadTickerConvictionBadge(ticker, containerEl)`.

**When ticker has prior resolved convictions:**
```
[⚖] RAPHI track record on NVDA: 71% ML acc · 74% SEC acc · +4.2% vs SPY over 6 prior calls    view ledger →
```
Purple strip (`rgba(124,108,240,0.06)`), purple top border.

**When ticker has zero resolved convictions (first call):**
```
[⚖] Conviction recorded. First RAPHI call on SMCI · resolution due in 30 / 60 / 90 days    view ledger →
```
Amber strip (`rgba(255,180,84,0.04)`), amber top border. No accuracy numbers shown.

### JavaScript functions

**`loadConvictionStats(signal)`**
- Calls `GET /api/convictions/stats`
- Writes: `cl-stat-ml-30`, `cl-stat-ml-60`, `cl-stat-ml-90`, `cl-stat-sec`, `cl-stat-spy`, `cl-stat-total`, `cl-stat-pending`, `nav-pending-count`
- Called by: `liveRefresh()` always + `switchPage('convictions')`

**`loadConvictionLedger(signal, page=1, ticker=null, signalType=null, window=null)`**
- Calls `GET /api/convictions/ledger?page=N&ticker=X`
- Writes: `cl-table-body`, `cl-pagination`, `cl-regime-low`, `cl-regime-mid`, `cl-regime-high`
- Called by: `switchPage('convictions')` + filter `onchange` handlers

**`loadTickerConvictionBadge(ticker, containerEl)`**
- Calls `GET /api/convictions/stats?ticker={ticker}`
- Appends inline strip to `containerEl` (no fixed DOM ID — dynamic)
- Branches on `resolved_count === 0` for first-call variant
- Called by: memo output renderer + signal card renderer, immediately after content injection

**`resolveConvictionsPoll(signal)`**
- Calls `GET /api/convictions/check` silently (no spinner)
- If `response.resolved > 0`: shows toast "X convictions resolved — accuracy updated", re-calls `loadConvictionStats()`, re-calls `loadConvictionLedger()` if current page is `convictions`
- Exceptions swallowed with `console.warn` — never blocks liveRefresh
- Called by: `liveRefresh()` always, inside Promise.all

**switchPage loaders addition:**
```js
convictions: () => Promise.all([loadConvictionStats(sig), loadConvictionLedger(sig)]),
```

---

## Section 4 — End-to-End Data Flow

### Scenario A: Conviction Creation

```
User question
  → RaphiAgentExecutor.execute()
  → sanitize_user_input() [4000 char cap, injection guard]
  → Claude Agent SDK query()
  → @memo-synthesizer via Task tool
  → 4 sub-agents dispatched in parallel:
      @market-analyst   → entry_price, sector
      @sec-researcher   → sec.trend, sec.latest_revenue, sec.next_filing_due
      @ml-signals       → ml.direction, ml.probability, ml.model_version
      @portfolio-risk   → signal_view, conviction tier
  → Schema validation gate (all four outputs)
  → write_conviction() → append to convictions.jsonl
  → conviction_id recorded in audit log
  → research output returned to user (includes conviction_id for traceability)
```

**If one agent fails validation:** field is `null`, `source` = `"partial_memo"`, conviction still written. That resolution type permanently skipped. User response includes disclaimer.

### Scenario B: ML Lookback Resolution

```
liveRefresh() fires (30s market-open / 60s market-closed)
  → resolveConvictionsPoll() in Promise.all
  → GET /api/convictions/check
  → check_pending():
      1. Read convictions.jsonl → dict[id → conviction]
      2. Read resolutions.jsonl → set[(id, lookback)] resolved pairs
      3. For each unresolved (conviction × {30d,60d,90d}) where today >= due_date:
          → yf.Ticker(ticker).history(period="5d") → latest close
          → yf.Ticker("SPY").history(period="5d") → SPY return same window
          → yf.Ticker("^VIX").history(period="5d") → VIX at check
          → compute: CONFIRMED / CONTRADICTED
          → append resolution line to resolutions.jsonl
  → return {resolved, still_pending, errors}
  → frontend: if resolved > 0 → toast + refresh stats
```

**Edge cases:**
- yfinance fails → skip, add to errors[], retry next cycle. No write.
- Already resolved → O(1) hash lookup skips. Fully idempotent.
- Market holiday → `history(period="5d")` returns last available close.
- NEUTRAL direction → `vs_entry_pct` within ±1.5% at the lookback date = CONFIRMED (stock stayed flat as predicted). Outside ±1.5% in either direction = CONTRADICTED (stock made a directional move the neutral signal did not anticipate).

### Scenario C: SEC Filing Resolution

```
check_pending() — SEC branch (separate from ML branch):
  For each conviction where today >= sec.next_filing_due
  AND (conviction_id, "sec") not in resolved set:
    → sec_data.company_financials(ticker)
    → filter to periods filed AFTER conviction.date
    → if newest quarter > conviction.date:
        → fetch actual revenue (XBRL tags priority: Revenues > RevenueFromContract > SalesRevenueNet)
        → revenue_delta_pct = (actual - conviction.sec.latest_revenue) / conviction.sec.latest_revenue * 100
        → |delta| <= 3% → INCONCLUSIVE (excluded from denominator)
        → delta > +3% and trend=accelerating → CONFIRMED
        → delta < -3% and trend=decelerating → CONFIRMED
        → mismatch → CONTRADICTED
        → append sec resolution to resolutions.jsonl
    → else: still_pending++ (EDGAR not yet updated)
```

**Edge cases:**
- `next_filing_due` is null → SEC resolution permanently skipped for this conviction
- Revenue XBRL tag absent → INCONCLUSIVE written, excluded from denominator
- EDGAR files not yet refreshed locally → skip, retry next cycle

### Scenario D: Accuracy Stats Computation

```
get_accuracy_stats(ticker=None):
  1. Read convictions.jsonl → dict[id → {ticker, ml, sec, signal_view, date}]
  2. Read resolutions.jsonl → dict[id → list[resolution]]
  3. Join: for each conviction × each resolution:
      - CONFIRMED → increment confirmed[signal_type][lookback][ticker]
      - CONTRADICTED → increment contradicted[...]
      - INCONCLUSIVE / pending → skip (excluded from denominator)
  4. Accuracy = confirmed / (confirmed + contradicted)
  5. Aggregate by: signal_type (ml/sec/signal_view) × lookback × ticker × VIX regime
  6. pending_count = convictions with zero resolutions

Denominator rule: A 35-day-old conviction contributes its 30d result.
Its 60d and 90d windows do not count for or against the score yet.
This prevents survivorship bias.
```

---

## Implementation Order

1. `backend/conviction_store.py` — data layer, all four functions
2. API endpoints in `raphi_server.py` — POST + three GETs
3. Integration hook in `a2a_executor_v2.py` — `write_conviction()` after schema validation
4. `liveRefresh()` additions in `index.html` — `loadConvictionStats()` + `resolveConvictionsPoll()`
5. Conviction Ledger page — sidebar nav + stats bar + table + regime cards
6. Inline compact view — `loadTickerConvictionBadge()` wired to memo and signal card renderers

---

## Out of Scope (Future)

- SQLite migration (when convictions exceed ~500 and regime queries become slow on JSONL)
- Nightly scheduled job as second resolution trigger (for idle-server production deployment)
- Standalone sentiment accuracy scoring (when VADER is replaced with a stronger model)
- Portfolio-level accuracy across all held positions (Phase 2 — Portfolio Narrative Contradiction feature)
