"""Tests for Control Plane 2 — the output gate on the agentic route.

After the agentic loop produces a final_answer, /api/agentic/query runs it
through backend.llm_guardrails.validate_and_repair_response before returning:
overconfident language is softened, memo intent gets schema repair, fabricated
tickers are flagged, and the guardrail report is surfaced as data["output_guardrail"].

The loop itself is not under test here, so it is patched to return a controlled
WorkflowState; the real guardrail runs against that state.
"""

import os
from unittest.mock import patch

os.environ.setdefault("RAPHI_API_KEY", "test-key")

from fastapi.testclient import TestClient

from backend.raphi_server import app
from raphi.orchestrators.state import WorkflowState

_HEADERS = {"X-API-Key": "test-key"}


def _client():
    return TestClient(app)


def _state(**overrides) -> WorkflowState:
    base = dict(
        run_id="test-run",
        user_query="q",
        intent="company_factual",
        risk_class="low",
        entities=[],
        tickers=[],
    )
    base.update(overrides)
    return WorkflowState(**base)


def _post(query: str):
    return _client().post("/api/agentic/query", json={"query": query}, headers=_HEADERS)


def test_overconfident_answer_is_softened():
    state = _state(
        intent="recommendation",
        risk_class="high",
        validated_tickers=["NVDA"],
        final_answer="NVDA is guaranteed to go up — a risk-free buy that cannot lose.",
    )
    with patch("raphi.orchestrators.agent_loop.run_agentic_query", return_value=state):
        resp = _post("should I buy NVDA")
    assert resp.status_code == 200
    data = resp.json()
    final = data["final_answer"]
    assert "guaranteed" not in final
    assert "risk-free" not in final
    assert data["output_guardrail"]["repairs"]  # non-empty


def test_output_guardrail_report_is_attached_on_finance_path():
    state = _state(validated_tickers=["NVDA"], final_answer="NVDA looks fine.")
    with patch("raphi.orchestrators.agent_loop.run_agentic_query", return_value=state):
        resp = _post("tell me about NVDA stock")
    assert resp.status_code == 200
    report = resp.json()["output_guardrail"]
    for key in ("valid", "repairs", "warnings", "missing_sections", "unknown_tickers"):
        assert key in report


def test_memo_intent_triggers_schema_repair():
    state = _state(
        intent="investment_memo",
        risk_class="high",
        validated_tickers=["NVDA"],
        final_answer="A quick note on NVDA without the required sections.",
    )
    with patch("raphi.orchestrators.agent_loop.run_agentic_query", return_value=state):
        resp = _post("write an investment memo for NVDA")
    data = resp.json()
    assert "Trade Plan" in data["final_answer"]
    assert "Risks" in data["final_answer"]
    assert data["output_guardrail"]["missing_sections"]


def test_legitimate_trending_tickers_are_not_flagged_unknown():
    state = _state(
        intent="trending_stocks",
        risk_class="medium",
        ranking_table=[{"ticker": "NVDA", "rank": 1, "score": 1.0}],
        final_answer="Top result: NVDA leads today's trending screen.",
    )
    with patch("raphi.orchestrators.agent_loop.run_agentic_query", return_value=state):
        resp = _post("what are the top trending stocks")
    data = resp.json()
    assert data["output_guardrail"]["unknown_tickers"] == []
    assert "Unverified ticker references" not in data["final_answer"]


def test_fabricated_ticker_in_prose_is_flagged():
    state = _state(
        validated_tickers=["NVDA"],
        final_answer="NVDA is solid, and I also really like ZZZZ here.",
    )
    with patch("raphi.orchestrators.agent_loop.run_agentic_query", return_value=state):
        resp = _post("tell me about NVDA stock")
    data = resp.json()
    assert "ZZZZ" in data["output_guardrail"]["unknown_tickers"]


def test_general_path_is_not_run_through_output_gate():
    # General queries short-circuit before the loop and must NOT get a guardrail
    # report or appended "Guardrail Notes" on their placeholder reply.
    resp = _post("Hello, how are you?")
    data = resp.json()
    assert "output_guardrail" not in data
    assert "Guardrail Notes" not in data["final_answer"]
