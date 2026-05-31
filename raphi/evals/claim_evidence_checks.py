from raphi.orchestrators.state import WorkflowState


def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def check_recommendation_provenance(state: WorkflowState):
    if state.intent == "recommendation":
        if not state.model_signals:
            return False, "Recommendation without model signal provenance"
    return True, None

def check_confidence_provenance(state: WorkflowState):
    if state.recommendation and state.recommendation.confidence is not None:
        if state.recommendation.confidence_source not in ["ml_model", "gnn_model", "deterministic_score"]:
            return False, "Confidence score without valid provenance"
    return True, None

def check_trending_query_contract(state: WorkflowState):
    if state.intent == "trending_stocks":
        if not state.universe or not state.time_window or not state.ranking_table:
            return False, "Trending workflow missing universe/time_window/ranking_table"
    return True, None

def check_market_claim_sources(state: WorkflowState):
    for ep in state.evidence_packets:
        if _field(ep, "source_type") == "market":
            if not _field(ep, "timestamp") or not _field(ep, "source_name") or not _field(ep, "retrieved_at"):
                return False, "Market claim missing timestamp/source_name/retrieved_at"
    return True, None

def check_sec_claim_sources(state: WorkflowState):
    for ep in state.evidence_packets:
        if _field(ep, "source_type") == "sec":
            if not (_field(ep, "accession") or _field(ep, "url")):
                return False, "SEC claim missing accession or SEC URL"
    return True, None

def check_evidence_packets_exist(state: WorkflowState):
    if state.intent in ["recommendation", "investment_memo", "sec_research", "latest_filing", "trending_stocks"]:
        if not state.evidence_packets:
            return False, "No evidence packets for research/recommendation answer"
    return True, None

def check_citation_freshness_for_current_claims(state: WorkflowState):
    for entry in state.claim_citation_map:
        if _field(entry, "freshness_status") not in ("fresh", "not_time_sensitive"):
            return False, "Stale or unknown citation for current claim"
    return True, None

def run_claim_evidence_checks(state: WorkflowState):
    checks = []
    failures = []
    warnings = []
    for fn in [
        check_recommendation_provenance,
        check_confidence_provenance,
        check_trending_query_contract,
        check_market_claim_sources,
        check_sec_claim_sources,
        check_evidence_packets_exist,
        check_citation_freshness_for_current_claims
    ]:
        ok, msg = fn(state)
        checks.append({"check": fn.__name__, "ok": ok, "msg": msg})
        if not ok:
            failures.append(msg)
    return {
        "passed": len(failures) == 0,
        "checks": checks,
        "failures": failures,
        "warnings": warnings
    }
