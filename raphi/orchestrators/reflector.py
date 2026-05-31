from raphi.orchestrators.state import WorkflowState, ReflectionResult


def _normalize_price_signal(direction: str) -> str:
    """Map LONG/SHORT/HOLD to BUY/SELL/HOLD for comparison with filing classifier."""
    return {"LONG": "BUY", "SHORT": "SELL"}.get(direction.upper(), "HOLD")


def reflect(state: WorkflowState) -> ReflectionResult:
    issues: list[dict] = []
    retryable = False
    suggested_plan_changes: list[dict] = []

    # ── Tool errors ──────────────────────────────────────────────────────
    normalized_errors = []
    for item in getattr(state, "errors", []) or []:
        if isinstance(item, dict):
            normalized_errors.append(item)
        elif item:
            normalized_errors.append({"error": str(item)})
    if any(e.get("error") for e in normalized_errors):
        issues.append({"type": "tool_error", "details": normalized_errors})
        retryable = True
        # For each failed tool, mark it non-required so the retry doesn't abort again
        for err in normalized_errors:
            tool = err.get("tool", "")
            if tool:
                suggested_plan_changes.append({
                    "action": "skip_failed_step",
                    "tool_name": tool,
                    "reason": err.get("error", "tool failed on previous attempt"),
                })

    # ── Missing universe for trending stocks ─────────────────────────────
    if state.intent == "trending_stocks" and not state.universe:
        issues.append({"type": "missing_universe", "details": "No universe for trending stocks"})

    # ── Missing model provenance for recommendation ───────────────────────
    if state.intent == "recommendation" and not state.has_model_provenance():
        issues.append({"type": "missing_model_provenance", "details": "No model signal for recommendation"})
        # Suggest adding ml_signal for each validated ticker
        for ticker in (state.validated_tickers or []):
            suggested_plan_changes.append({
                "action": "add_step",
                "tool_name": "ml_signal",
                "ticker": ticker,
                "purpose": f"Add model signal for {ticker} to support recommendation",
                "args": {"ticker": ticker},
                "required": False,
                "expected_output": "Model signal direction and confidence score",
            })

    # ── Signal disagreement: price model vs. filing text classifier ───────
    # When XGBoost/LSTM says BUY but the fine-tuned filing classifier says SELL
    # (or vice versa), the two independent signals conflict.
    # Retry with a deeper edgar_live_summary to resolve the conflict.
    for res in state.retrieval_results.values():
        if not isinstance(res, dict):
            continue
        direction     = res.get("direction", "")
        filing_signal = res.get("filing_signal", "")
        if not (direction and filing_signal):
            continue
        price_signal = _normalize_price_signal(direction)
        # Only flag hard disagreements (BUY vs SELL); HOLD on either side is not a conflict
        if (
            price_signal  != filing_signal
            and price_signal  != "HOLD"
            and filing_signal != "HOLD"
            and filing_signal != "default"
        ):
            ticker = res.get("ticker", "unknown")
            issues.append({
                "type":          "signal_disagreement",
                "ticker":        ticker,
                "details":       (
                    f"Price model says {direction} but filing classifier says {filing_signal} "
                    f"for {ticker} — conflicting signals; deeper analysis warranted."
                ),
                "price_signal":  price_signal,
                "filing_signal": filing_signal,
                "filing_reason": res.get("filing_reason", ""),
            })
            retryable = True
            # Suggest adding edgar_live_summary to get 8-K events + insider activity
            # that might explain the divergence
            suggested_plan_changes.append({
                "action":          "add_step",
                "tool_name":       "edgar_live_summary",
                "ticker":          ticker,
                "purpose":         (
                    f"Resolve signal conflict for {ticker}: price model says {price_signal} "
                    f"but filing text says {filing_signal} — fetch 8-K events and insider activity"
                ),
                "args":            {"ticker": ticker, "days": 90},
                "required":        False,
                "expected_output": "Recent 8-Ks, Form 4 insider trades, latest 10-Q/10-K narrative",
            })

    return ReflectionResult(
        passed=(len(issues) == 0),
        issues=issues,
        retryable=retryable,
        suggested_plan_changes=suggested_plan_changes,
    )
