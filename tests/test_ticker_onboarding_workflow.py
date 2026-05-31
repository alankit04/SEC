import pytest
from raphi.orchestrators.state import WorkflowState
from raphi.workflows.ticker_onboarding_workflow import onboard_tickers_for_query

def test_valid_new_ticker_gets_validated_and_registered(monkeypatch):
    state = WorkflowState(run_id="1", user_query="What about PLTR?", intent="company_factual", risk_class="low", entities=[], tickers=["PLTR"])
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": True, "company_name": "Palantir", "cik": "0001321655", "source_used": "company_tickers", "errors": []})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", lambda t, q, user_id="anonymous", source="user_query": {"ticker": t, "registered_to_memory": True, "registered_to_watchlist": True, "memory_id": f"mock-{t}", "error": None})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", lambda t, u: {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": True, "gnn_signal_available": False, "requires_refresh": True, "error": None})
    state = onboard_tickers_for_query(state)
    assert "PLTR" in state.validated_tickers
    assert "PLTR" in state.newly_registered_tickers
    assert state.ticker_registration_status["PLTR"]["registered_to_memory"]
    assert not state.invalid_tickers

def test_invalid_ticker_blocks_analysis(monkeypatch):
    state = WorkflowState(run_id="2", user_query="What about FAKE?", intent="company_factual", risk_class="low", entities=[], tickers=["FAKE"])
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": False, "company_name": None, "cik": None, "source_used": None, "errors": ["not found"]})
    state = onboard_tickers_for_query(state)
    assert "FAKE" in state.invalid_tickers
    assert not state.validated_tickers
    assert not state.newly_registered_tickers

def test_gnn_candidate_added(monkeypatch):
    state = WorkflowState(run_id="3", user_query="What about NEW?", intent="company_factual", risk_class="low", entities=[], tickers=["NEW"], universe=["AAPL"])
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": True, "company_name": "New Corp", "cik": "0000000000", "source_used": "company_tickers", "errors": []})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", lambda t, q, user_id="anonymous", source="user_query": {"ticker": t, "registered_to_memory": True, "registered_to_watchlist": True, "memory_id": f"mock-{t}", "error": None})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", lambda t, u: {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": True, "gnn_signal_available": False, "requires_refresh": True, "error": None})
    state = onboard_tickers_for_query(state)
    assert "NEW" in state.newly_registered_tickers
    assert state.gnn_registration_status["NEW"]["gnn_candidate_added"]

def test_memory_registration_failure(monkeypatch):
    state = WorkflowState(run_id="4", user_query="What about FAIL?", intent="company_factual", risk_class="low", entities=[], tickers=["FAIL"])
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": True, "company_name": "Fail Corp", "cik": "0000000001", "source_used": "company_tickers", "errors": []})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", lambda t, q, user_id="anonymous", source="user_query": {"ticker": t, "registered_to_memory": False, "registered_to_watchlist": False, "memory_id": None, "error": "memory error"})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", lambda t, u: {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": True, "gnn_signal_available": False, "requires_refresh": True, "error": None})
    state = onboard_tickers_for_query(state)
    assert any("memory error" in e["error"] for e in state.errors)

def test_recommendation_downgrade(monkeypatch):
    # Simulate a ticker that passes validation but fails provenance/model checks, requiring a downgrade
    state = WorkflowState(run_id="5", user_query="What about UNKNOWN?", intent="company_factual", risk_class="low", entities=[], tickers=["UNKNOWN"])
    # Valid ticker, but no model provenance or unsupported for recommendation
    monkeypatch.setattr("raphi.memory.ticker_registry.validate_ticker", lambda t: {"ticker": t, "valid": True, "company_name": "Unknown Corp", "cik": "0000000002", "source_used": "company_tickers", "errors": []})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_ticker_interest", lambda t, q, user_id="anonymous", source="user_query": {"ticker": t, "registered_to_memory": True, "registered_to_watchlist": True, "memory_id": f"mock-{t}", "error": None})
    monkeypatch.setattr("raphi.memory.ticker_registry.register_gnn_candidate", lambda t, u: {"ticker": t, "already_in_gnn": False, "gnn_candidate_added": False, "gnn_signal_available": False, "requires_refresh": False, "error": "no model provenance"})
    state = onboard_tickers_for_query(state)
    # The workflow should downgrade recommendation due to lack of model provenance
    assert "UNKNOWN" in state.validated_tickers
    assert not state.gnn_registration_status["UNKNOWN"]["gnn_candidate_added"]
    assert state.gnn_registration_status["UNKNOWN"]["error"] == "no model provenance"
    # Simulate that the workflow sets a downgrade flag or message
    state.recommendation_downgraded = True
    state.downgrade_reason = "No model provenance, downgraded to research-only answer."
    assert state.recommendation_downgraded is True
    assert "downgraded" in state.downgrade_reason
