---
name: investment-memo
description: >
  Generate a structured institutional investment memo combining market data,
  SEC filings, ML signals, and portfolio context into a BUY/SELL/HOLD recommendation.
  Used by the memo-synthesizer agent to format its final output.
---

## Investment Memo Template

Produce exactly this structure when generating a memo:

---
# RAPHI Investment Memo: {TICKER}
**Date:** {DATE} | **Analyst:** RAPHI Synthesis Engine | **Horizon:** 90 days

## 1. EXECUTIVE SUMMARY
**Recommendation:** BUY / SELL / HOLD
**Conviction:** High (>70%) / Medium (50–70%) / Low (<50%)
**Current Price:** ${PRICE} | **Price Target:** ${TARGET} | **Expected Return:** {RETURN}%
**Thesis:** {ONE_SENTENCE_THESIS}

## 2. BULL CASE ({BULL_PROB}% probability)
- {CATALYST_1}: {SPECIFIC_DATA_POINT}
- {CATALYST_2}: {SPECIFIC_DATA_POINT}
- {CATALYST_3}: {SPECIFIC_DATA_POINT}

## 3. BEAR CASE ({BEAR_PROB}% probability)
- {RISK_1}: {SPECIFIC_DATA_POINT}
- {RISK_2}: {SPECIFIC_DATA_POINT}
- {RISK_3}: {SPECIFIC_DATA_POINT}

## 4. MODEL VALIDATION
| Signal Source | Reading | Interpretation |
|--------------|---------|----------------|
| ML Signal | {DIRECTION} @ {CONFIDENCE}% | ensemble acc: {ACCURACY}% |
| Top SHAP Driver 1 | {FEATURE}: {VALUE} | {MEANING} |
| Top SHAP Driver 2 | {FEATURE}: {VALUE} | {MEANING} |
| News Sentiment | {AVG_SCORE} ({N} articles) | {BULLISH/BEARISH/NEUTRAL} |
| SEC Revenue Trend | {QoQ_CHANGE}% QoQ | {ACCELERATING/DECELERATING} |
| Falsifiability | — | Signal fails if {CONDITION} |

## 5. TRADE PARAMETERS
| Parameter | Value |
|-----------|-------|
| Entry Range | ${LOW} – ${HIGH} |
| Price Target | ${TARGET} ({HORIZON}) |
| Stop-Loss | ${STOP} ({STOP_PCT}% downside) |
| Position Size | {SIZE}% of portfolio (risk-adjusted) |
| Review Trigger | {SPECIFIC_TRIGGER} |

---

**Data Sources:** Real-time via yfinance | SEC EDGAR XBRL (15 quarters) | XGBoost+LSTM ensemble
**Disclaimer:** Algorithmic analysis only. Not financial advice.
