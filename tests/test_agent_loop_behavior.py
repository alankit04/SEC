import pytest
from raphi.orchestrators.agent_loop import run_agentic_query

def test_casual_question_no_onboarding():
    state = run_agentic_query("How are you?")
    assert state.intent == "casual_chat"
    assert not state.tickers

def test_factual_new_ticker_validates_and_registers(monkeypatch):
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": True, "company_name": "Palantir", "cik": "0001321655", "source_used": "company_tickers", "errors": []})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", lambda t, q, user_id="anonymous", source="user_query": {"ticker": t, "registered_to_memory": True, "registered_to_watchlist": True, "memory_id": f"mock-{t}", "error": None})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", lambda t, u: {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": True, "gnn_signal_available": False, "requires_refresh": True, "error": None})
    state = run_agentic_query("What about PLTR?")
    assert "PLTR" in state.validated_tickers
    assert state.intent == "company_factual"

def test_sec_new_ticker_runs_sec_plan(monkeypatch):
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": True, "company_name": "Palantir", "cik": "0001321655", "source_used": "company_tickers", "errors": []})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", lambda t, q, user_id="anonymous", source="user_query": {"ticker": t, "registered_to_memory": True, "registered_to_watchlist": True, "memory_id": f"mock-{t}", "error": None})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", lambda t, u: {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": True, "gnn_signal_available": False, "requires_refresh": True, "error": None})
    state = run_agentic_query("Show me the latest 10-Q for PLTR")
    assert state.intent in ["latest_filing", "sec_research"]
    assert "PLTR" in state.validated_tickers

def test_model_signal_discloses_missing_gnn(monkeypatch):
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": True, "company_name": "Palantir", "cik": "0001321655", "source_used": "company_tickers", "errors": []})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", lambda t, q, user_id="anonymous", source="user_query": {"ticker": t, "registered_to_memory": True, "registered_to_watchlist": True, "memory_id": f"mock-{t}", "error": None})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", lambda t, u: {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": True, "gnn_signal_available": False, "requires_refresh": True, "error": None})
    state = run_agentic_query("Show me the model signal for PLTR")
    assert state.intent == "model_signal"
    # Should disclose missing GNN coverage in uncertainty_flags or final_answer

def test_recommendation_high_risk(monkeypatch):
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": True, "company_name": "Palantir", "cik": "0001321655", "source_used": "company_tickers", "errors": []})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", lambda t, q, user_id="anonymous", source="user_query": {"ticker": t, "registered_to_memory": True, "registered_to_watchlist": True, "memory_id": f"mock-{t}", "error": None})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", lambda t, u: {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": True, "gnn_signal_available": False, "requires_refresh": True, "error": None})
    state = run_agentic_query("Should I buy PLTR?")
    assert state.intent == "recommendation"
    assert state.risk_class == "high"

def test_unsupported_recommendation_downgraded():
    # Simulate a recommendation query with no model provenance, should be downgraded to research_only
    def fake_validate_ticker(t):
        return {"ticker": t, "valid": True, "company_name": "Palantir", "cik": "0001321655", "source_used": "company_tickers", "errors": []}
    def fake_register_ticker_interest(t, q, user_id="anonymous", source="user_query"):
        return {"ticker": t, "registered_to_memory": True, "registered_to_watchlist": True, "memory_id": f"mock-{t}", "error": None}
    def fake_register_gnn_candidate(t, u):
        return {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": True, "gnn_signal_available": False, "requires_refresh": True, "error": None}

    import types
    from raphi.orchestrators import agent_loop
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", fake_validate_ticker)
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", fake_register_ticker_interest)
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", fake_register_gnn_candidate)

    # Patch run_research_workflow to simulate no model_signals
    orig_run_research_workflow = getattr(agent_loop, "run_research_workflow", None)
    def fake_run_research_workflow(state):
        state.model_signals = []
        state.recommendation = None
        state.final_answer = "Research view only — I cannot provide a recommendation because structured model provenance is missing."
        return state
    if orig_run_research_workflow:
        monkeypatch.setattr(agent_loop, "run_research_workflow", fake_run_research_workflow)

    state = agent_loop.run_agentic_query("Should I buy PLTR?")
    assert state.intent == "recommendation"
    assert state.risk_class == "high"
    assert not state.model_signals
    assert "provenance" in state.final_answer.lower() or "research view" in state.final_answer.lower()
    monkeypatch.undo()

def test_self_heal_retries():
    # Simulate a tool failure and check that the agent attempts a self-heal (retry)
    from raphi.orchestrators import agent_loop
    class DummyException(Exception):
        pass
    class DummyState:
        def __init__(self):
            self.tool_plan = ["fail_tool"]
            self.retrieval_results = {}
            self.reflection = None
            self.final_answer = ""
            self.retry_count = 0
    def fake_execute_plan(state):
        if getattr(state, "retry_count", 0) < 1:
            state.retry_count = getattr(state, "retry_count", 0) + 1
            raise DummyException("Tool failed")
        state.retrieval_results = {"fail_tool": "success after retry"}
        return state
    def fake_reflect(state):
        if getattr(state, "retry_count", 0) < 1:
            return types.SimpleNamespace(passed=False, issues=[{"tool": "fail_tool"}], retryable=True, suggested_plan_changes=[])
        return types.SimpleNamespace(passed=True, issues=[], retryable=False, suggested_plan_changes=[])
    orig_execute_plan = getattr(agent_loop, "execute_plan", None)
    orig_reflect = getattr(agent_loop, "reflect", None)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(agent_loop, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(agent_loop, "reflect", fake_reflect)
    state = DummyState()
    try:
        # Simulate the agentic loop with a retry
        for _ in range(2):
            try:
                state = agent_loop.execute_plan(state)
            except DummyException:
                state.reflection = agent_loop.reflect(state)
                if getattr(state.reflection, "retryable", False):
                    continue
                else:
                    break
        assert state.retry_count == 1
        assert state.retrieval_results["fail_tool"] == "success after retry"
    finally:
        if orig_execute_plan:
            monkeypatch.setattr(agent_loop, "execute_plan", orig_execute_plan)
        if orig_reflect:
            monkeypatch.setattr(agent_loop, "reflect", orig_reflect)
        monkeypatch.undo()
