import pytest
from raphi.orchestrators.state import WorkflowState, ClaimCitationMapEntry, Recommendation
from raphi.evals.claim_evidence_checks import run_claim_evidence_checks

def make_state(**kwargs):
    state = WorkflowState(run_id="1", user_query="", intent="recommendation", risk_class="high", entities=[], tickers=[])
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state

def test_recommendation_without_model_signal_fails():
    state = make_state(model_signals=[], recommendation=Recommendation(decision="buy", confidence=0.8, confidence_source="ml_model", rationale=[], risk_framing=[], allowed=True, downgrade_reason=None))
    result = run_claim_evidence_checks(state)
    assert not result["passed"]
    assert any("Recommendation without model signal" in f for f in result["failures"])

def test_confidence_without_provenance_fails():
    state = make_state(model_signals=[object()], recommendation=Recommendation(decision="buy", confidence=0.8, confidence_source="unknown", rationale=[], risk_framing=[], allowed=True, downgrade_reason=None))
    result = run_claim_evidence_checks(state)
    assert not result["passed"]
    assert any("Confidence score without valid provenance" in f for f in result["failures"])

def test_trending_workflow_without_universe_fails():
    state = make_state(intent="trending_stocks", universe=[], time_window="", ranking_table=[])
    result = run_claim_evidence_checks(state)
    assert not result["passed"]
    assert any("Trending workflow missing universe" in f for f in result["failures"])

def test_market_claim_without_source_fails():
    class EP:
        source_type = "market"
        timestamp = None
        source_name = None
        retrieved_at = None
    state = make_state(evidence_packets=[EP()])
    result = run_claim_evidence_checks(state)
    assert not result["passed"]
    assert any("Market claim missing timestamp" in f for f in result["failures"])

def test_stale_citation_fails_for_current_claim():
    entry = ClaimCitationMapEntry(claim_id="1", claim="", evidence_ids=[], citation_urls=[], support_status="supported", freshness_status="stale", citation_age_hours=100, stale_reason="old", notes="")
    state = make_state(claim_citation_map=[entry])
    result = run_claim_evidence_checks(state)
    assert not result["passed"]
    assert any("Stale or unknown citation" in f for f in result["failures"])

def test_unknown_date_citation_not_fresh():
    entry = ClaimCitationMapEntry(claim_id="1", claim="", evidence_ids=[], citation_urls=[], support_status="supported", freshness_status="unknown", citation_age_hours=None, stale_reason="no date", notes="")
    state = make_state(claim_citation_map=[entry])
    result = run_claim_evidence_checks(state)
    assert not result["passed"]
    assert any("Stale or unknown citation" in f for f in result["failures"])

def test_valid_state_passes():
    entry = ClaimCitationMapEntry(claim_id="1", claim="", evidence_ids=[], citation_urls=[], support_status="supported", freshness_status="fresh", citation_age_hours=1, stale_reason=None, notes="")
    state = make_state(claim_citation_map=[entry], model_signals=[object()], evidence_packets=[object()], recommendation=Recommendation(decision="buy", confidence=0.8, confidence_source="ml_model", rationale=[], risk_framing=[], allowed=True, downgrade_reason=None))
    result = run_claim_evidence_checks(state)
    assert result["passed"]
