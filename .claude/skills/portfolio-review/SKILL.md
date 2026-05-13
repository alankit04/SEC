---
name: portfolio-review
description: >
  Run a full portfolio health check: risk metrics, position attribution,
  rebalancing suggestions, and VaR stress scenarios. Invokes portfolio-risk
  and ml-signals agents to generate a comprehensive review report.
---

## Portfolio Review Process

When this skill is invoked:
1. Call `mcp__raphi__portfolio_snapshot` for positions and risk metrics
2. Call `mcp__raphi__portfolio_alerts` for active risk alerts
3. For each position with weight > 15%, call `mcp__raphi__ml_signal`
4. Compile the report using this template:

---
# RAPHI Portfolio Health Report
**Date:** {DATE} | **Total Value:** ${VALUE} | **Positions:** {N}

## Performance Dashboard
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Total P&L | ${PNL} ({PCT}%) | — | — |
| Daily VaR 95% | ${VAR95} ({VAR_PCT}%) | < 2% | ✓/⚠/✗ |
| Daily VaR 99% | ${VAR99} | — | — |
| Sharpe Ratio | {SHARPE} | > 1.0 | ✓/⚠/✗ |
| Alpha vs SPY | {ALPHA}% | > 0% | ✓/⚠/✗ |

## Position Attribution
| Ticker | Direction | Weight | P&L% | ML Signal | Stop Distance |
|--------|-----------|--------|------|-----------|---------------|
| {rows sorted by weight descending} |

## Active Alerts
{List alerts with severity: CRITICAL / WARNING / INFO}
{If none: "No active alerts — portfolio within risk parameters"}

## ML Signal Alignment
{For positions with signal data: table of ticker vs signal direction vs position direction}
{Flag any misaligned positions: e.g., SHORT ML signal on a LONG position}

## Rebalancing Recommendations
1. {SPECIFIC ACTION with sizing — e.g., "Reduce NVDA from 35% to 25% weight (concentration risk)"}
2. {SPECIFIC ACTION with rationale}
3. {SPECIFIC ACTION with rationale}

---
