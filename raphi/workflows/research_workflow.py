from raphi.orchestrators.state import WorkflowState, ToolPlanStep, ClaimCitationMapEntry
from raphi.orchestrators.planner import build_plan
from raphi.orchestrators.tool_executor import execute_plan
from raphi.evals.citation_freshness import infer_freshness_requirement, evaluate_citation_freshness, should_refresh_citation
from raphi.evals.claim_evidence_checks import run_claim_evidence_checks
from raphi.orchestrators.reflector import reflect


def _apply_plan_changes(state: WorkflowState, changes: list[dict]) -> WorkflowState:
    """
    Apply reflector's suggested_plan_changes to the tool plan before a retry.

    Actions:
      add_step        — append a new ToolPlanStep (deduplicated by tool+ticker)
      skip_failed_step — mark an existing step required=False so timeout won't abort
    """
    for change in changes:
        action = change.get("action")

        if action == "add_step":
            tool_name = change["tool_name"]
            ticker    = change.get("ticker", "")
            # Don't add if the same tool+ticker already exists in the plan
            already_present = any(
                s.tool_name == tool_name and s.args.get("ticker", "") == ticker
                for s in state.tool_plan
            )
            if not already_present:
                step_id = f"retry_{tool_name}_{ticker}_{len(state.tool_plan)}"
                state.tool_plan.append(ToolPlanStep(
                    id=step_id,
                    tool_name=tool_name,
                    purpose=change.get("purpose", f"Retry: {tool_name}"),
                    required=change.get("required", False),
                    args=change.get("args", {}),
                    expected_output=change.get("expected_output", ""),
                ))

        elif action == "skip_failed_step":
            # Mark the failing step as optional so a timeout doesn't abort the retry
            for step in state.tool_plan:
                if step.tool_name == change["tool_name"]:
                    step.required = False

    return state


def _clear_failed_results(state: WorkflowState) -> WorkflowState:
    """
    Smart retry preparation: drop only error results, keep successful ones.
    execute_plan skips steps whose result is already in retrieval_results,
    so this ensures only genuinely failed steps are re-run.
    """
    state.retrieval_results = {
        step_id: result
        for step_id, result in state.retrieval_results.items()
        if not (isinstance(result, dict) and result.get("error"))
    }
    state.errors = []
    return state


def run_research_workflow(state: WorkflowState) -> WorkflowState:
    from datetime import datetime, timezone
    from raphi.tools.evidence_collector import collect_dynamic_evidence
    from raphi.evals.citation_freshness import infer_freshness_requirement, evaluate_citation_freshness, should_refresh_citation
    from raphi.evals.claim_evidence_checks import run_claim_evidence_checks
    now = datetime.now(timezone.utc).isoformat()
    # 1. Build plan if needed
    if not state.tool_plan:
        state.tool_plan = build_plan(state)
    # 2. Execute plan with a bounded reflection/retry loop
    if not state.retrieval_results:
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                state = execute_plan(state)
            except Exception as exc:
                state.add_error({"tool": "execute_plan", "error": str(exc), "attempt": attempt})

            state.reflection = reflect(state)

            if state.reflection.passed or not state.reflection.retryable or attempt >= max_attempts:
                break

            # Apply suggested plan changes: add steps to resolve conflicts,
            # mark failing tools as non-required so they don't abort the retry
            if state.reflection.suggested_plan_changes:
                state = _apply_plan_changes(state, state.reflection.suggested_plan_changes)

            # Smart retry: keep successful results, only re-run failed/new steps
            state = _clear_failed_results(state)

            state.add_trace({
                "tool_name":      "reflector",
                "args":           {"attempt": attempt, "changes": len(state.reflection.suggested_plan_changes)},
                "started_at":     now,
                "ended_at":       datetime.now(timezone.utc).isoformat(),
                "latency_ms":     0,
                "ok":             True,
                "error":          None,
                "output_summary": (
                    f"Retry {attempt}: applied {len(state.reflection.suggested_plan_changes)} plan changes. "
                    f"Issues: {[i['type'] for i in state.reflection.issues]}"
                ),
                "status":         "retry",
            })
    # 3. Collect dynamic evidence
    state = collect_dynamic_evidence(state)
    if state.reflection is None:
        state.reflection = reflect(state)
    # 4. Evaluate citation freshness for each evidence packet and write back
    freshness_req = infer_freshness_requirement(state.user_query, state.intent)
    status_list = []
    fresh_count = 0
    stale_count = 0
    unknown_count = 0
    not_time_sensitive_count = 0
    for ep in state.evidence_packets:
        # Support dict or dataclass
        from raphi.evals.citation_freshness import get_field, set_field
        result = evaluate_citation_freshness(ep, freshness_req)
        set_field(ep, "freshness_required", freshness_req.requires_freshness)
        set_field(ep, "freshness_window_hours", freshness_req.max_age_hours)
        set_field(ep, "freshness_status", result.freshness_status)
        set_field(ep, "stale_reason", result.stale_reason)
        set_field(ep, "refresh_attempted", should_refresh_citation(result))
        set_field(ep, "refresh_successful", False)
        set_field(ep, "citation_age_hours", result.age_hours if hasattr(ep, "citation_age_hours") or (isinstance(ep, dict) and "citation_age_hours" in ep) else None)
        status_list.append({
            "evidence_id": get_field(ep, "evidence_id"),
            "freshness_status": result.freshness_status,
            "stale_reason": result.stale_reason
        })
        if result.freshness_status == "fresh":
            fresh_count += 1
        elif result.freshness_status == "stale":
            stale_count += 1
        elif result.freshness_status == "unknown":
            unknown_count += 1
        elif result.freshness_status == "not_time_sensitive":
            not_time_sensitive_count += 1
    state.citation_freshness_status = {
        "requires_freshness": freshness_req.requires_freshness,
        "max_age_hours": freshness_req.max_age_hours,
        "checked_at": now,
        "fresh_count": fresh_count,
        "stale_count": stale_count,
        "unknown_count": unknown_count,
        "not_time_sensitive_count": not_time_sensitive_count,
        "status": status_list
    }
    # 5. Build ClaimCitationMap (list of entries) after freshness
    claim_citation_map = []
    for ep in state.evidence_packets:
        from raphi.evals.citation_freshness import get_field
        freshness_status = get_field(ep, "freshness_status")
        support_status = "supported" if get_field(ep, "supports_claim") and freshness_status == "fresh" else "background"
        notes = ""
        if freshness_req.requires_freshness and freshness_status == "stale":
            support_status = "partially_supported"
            notes = "Stale evidence cannot fully support current/latest claim."
        claim_citation_map.append(ClaimCitationMapEntry(
            claim_id=get_field(ep, "claim_id"),
            claim=get_field(ep, "claim"),
            evidence_ids=[get_field(ep, "evidence_id")],
            citation_urls=[get_field(ep, "url")],
            support_status=support_status,
            freshness_status=freshness_status,
            citation_age_hours=get_field(ep, "citation_age_hours", None),
            stale_reason=get_field(ep, "stale_reason", None),
            notes=notes
        ))
    state.claim_citation_map = claim_citation_map
    # 6. Recommendation downgrade for high-risk
    if state.intent == "recommendation":
        # If no model_signals or governance, downgrade
        if not state.model_signals:
            from raphi.orchestrators.state import Recommendation
            state.recommendation = Recommendation(
                decision="research_only",
                confidence=None,
                confidence_source="",
                rationale=["No model or governance provenance for recommendation"],
                risk_framing=["High risk: unsupported recommendation"],
                allowed=False,
                downgrade_reason="missing_model_or_governance_provenance"
            )
            state.governance_status = {
                "status": "downgraded",
                "reason": "missing_model_or_governance_provenance",
                "allowed": False,
            }
    elif state.intent == "model_signal":
        state.governance_status = {
            "status": "research_only",
            "reason": "model signal is informational and not a trade instruction",
            "allowed": True,
        }
    # 7. Run claim evidence checks
    state.eval_status = run_claim_evidence_checks(state)
    # 8. Render final_answer
    answer_parts = []
    # Insert research-only message for downgraded recommendations
    if state.intent == "recommendation" and state.recommendation and state.recommendation.downgrade_reason:
        answer_parts.append(
            "Research-only: I cannot provide a buy/sell/long/short recommendation because structured model provenance, evidence support, or governance approval is missing."
        )
    if state.intent == "model_signal":
        unavailable = [
            ticker
            for ticker, status in (state.gnn_registration_status or {}).items()
            if not status.get("gnn_signal_available")
        ]
        if unavailable:
            answer_parts.append(
                f"GNN signal unavailable for {', '.join(unavailable)}; retrain or refresh graph coverage before treating peer influence as complete."
            )
    # Standard trace details
    answer_parts.extend([
        f"Query classified as: {state.intent} (risk: {state.risk_class})",
        f"Tickers: {state.tickers}",
        f"Validated: {state.validated_tickers}",
        f"Invalid: {state.invalid_tickers}",
        f"Evidence count: {len(state.evidence_packets)}",
        f"Citation freshness: {state.citation_freshness_status}",
    ])
    if state.intent == "recommendation" and state.recommendation:
        answer_parts.append(f"Recommendation: {state.recommendation.decision} (allowed: {state.recommendation.allowed})")
        if state.recommendation.downgrade_reason:
            answer_parts.append(f"Downgraded: {state.recommendation.downgrade_reason}")
    if state.eval_status and not state.eval_status.get("passed", True):
        answer_parts.append(f"Eval failures: {state.eval_status.get('failures')}")
    state.final_answer = "\n".join(answer_parts)
    return state
