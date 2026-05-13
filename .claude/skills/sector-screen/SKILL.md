---
name: sector-screen
description: >
  Screen a sector or custom ticker list for investment opportunities by running
  ML signals and ranking by confidence. Pass a sector name (e.g. "semiconductors")
  or comma-separated ticker list as the argument.
---

## Sector Screen Process

**Input:** sector name (e.g., "semiconductors", "cloud software") or ticker list (e.g., "NVDA,AMD,INTC,QCOM")

**Steps:**
1. If sector name given, derive relevant tickers from watchlist or SEC SIC codes
2. For each ticker, call `mcp__raphi__ml_signal` and `mcp__raphi__stock_detail` in parallel
3. Filter: keep only LONG or SHORT signals with confidence > 60% and ensemble_accuracy > 75%
4. Rank by confidence descending

**Output template:**

---
# RAPHI Sector Screen: {SECTOR}
**Date:** {DATE} | **Screened:** {N} tickers | **Qualifying signals:** {M}

## Signal Leaderboard
| Rank | Ticker | Signal | Conf% | Acc% | Price | P/E | Top SHAP Driver |
|------|--------|--------|-------|------|-------|-----|-----------------|
| {rows — only qualifying signals} |

## Disqualified (low confidence or accuracy)
{Ticker list with reason — e.g., "INTC: HOLD (52% conf)"}

## Top Pick: {TICKER}
**Rationale:** {2–3 sentences citing specific SHAP drivers, P/E context, and SEC revenue trend}

## Sector Themes
{1–2 sentences on cross-ticker SHAP patterns that indicate sector-wide drivers}

---

**Note:** Only signals with confidence >60% and accuracy >75% are actionable.
Signals with confidence 50–60% are HOLD — position sizing should reflect uncertainty.
