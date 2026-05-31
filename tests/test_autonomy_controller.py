from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
os.environ.setdefault("RAPHI_API_KEY", "test-key")

from autonomy_controller import AutonomyController
import raphi_server


API_HEADERS = {
    "X-API-Key": "test-key",
    "X-User-Id": "unit-test-user",
    "X-Tenant-Id": "unit",
}

ACTION_HEADERS = {
    **API_HEADERS,
    "X-Action-Approval": "approved",
}


def test_autonomy_controller_learns_and_calibrates(tmp_path):
    policy_file = tmp_path / "policy_state.json"
    controller = AutonomyController(policy_file=policy_file)

    run = {
        "prompt": "write a full investment memo for NVDA",
        "user_id": "unit:user",
        "user_role": "analyst",
        "latency_ms": 2100,
        "observed_tools": ["market", "sec", "citation"],
        "eval_result": {
            "overall_score": 0.91,
            "passed": True,
            "metrics": {
                "citation_precision": {"score": 1.0},
            },
        },
        "review": {"retry_failures": []},
    }

    learned = controller.learn_from_run(run)
    assert learned["intent"] == "memo"
    assert learned["pass"] is True
    assert 0.0 <= learned["calibrated_confidence"] <= 1.0

    status = controller.status()
    assert status["global"]["runs"] >= 1
    assert "memo" in status["intent_stats"]
    assert "market" in status["tool_reliability"]


def test_autonomy_objective_changes_by_intent(tmp_path):
    controller = AutonomyController(policy_file=tmp_path / "policy.json")

    lookup = controller.objective_for(
        message="what is nvda price",
        intent="lookup",
        user_scope="unit:user",
        user_role="analyst",
        provider_status={"status": "ok"},
    )
    memo = controller.objective_for(
        message="write full memo with evidence",
        intent="memo",
        user_scope="unit:user",
        user_role="analyst",
        provider_status={"status": "ok"},
    )

    assert lookup.latency_budget_ms < memo.latency_budget_ms
    assert lookup.quality_target < memo.quality_target
    assert memo.tool_budgets["sec"] >= 1


def test_autonomy_learns_user_behavior_preferences(tmp_path):
    controller = AutonomyController(policy_file=tmp_path / "policy.json")

    controller.learn_from_behavior(
        user_scope="unit:user",
        event_type="page_view",
        metadata={"page": "signals"},
    )
    controller.learn_from_behavior(
        user_scope="unit:user",
        event_type="chat_message",
        metadata={
            "ticker": "NVDA",
            "intent": "memo",
            "response_mode": "debug",
            "is_followup": True,
            "message_length": 88,
        },
    )
    controller.learn_from_behavior(
        user_scope="unit:user",
        event_type="chat_message",
        metadata={
            "ticker": "NVDA",
            "intent": "compare",
            "response_mode": "debug",
            "is_followup": True,
            "message_length": 64,
        },
    )

    profile = controller.behavior_profile("unit:user")
    assert profile["preferred_tickers"][0] == "NVDA"
    assert profile["preferred_pages"][0] == "signals"
    assert profile["preferred_response_mode"] == "debug"
    assert profile["preferred_focus"] == "evidence_depth"

    objective = controller.objective_for(
        message="compare NVDA and AMD with evidence",
        intent="compare",
        user_scope="unit:user",
        user_role="analyst",
        provider_status={"status": "ok"},
    )
    assert objective.tool_budgets["citation"] >= 2
    assert objective.critique_loops >= 3


def test_autonomy_behavior_filters_common_words_as_tickers(tmp_path):
    controller = AutonomyController(policy_file=tmp_path / "policy.json")

    controller.learn_from_behavior(
        user_scope="unit:user",
        event_type="chat_message",
        metadata={"tickers": ["WHAT", "WHO", "TELL", "LOCAL", "NVDA"], "intent": "lookup"},
    )

    behavior = controller._behavior_bucket("unit:user")
    behavior.setdefault("ticker_interest", {}).update({"WHAT": 4, "WHO": 3, "GOOGL": 2, "NVDA": 5})

    profile = controller.behavior_profile("unit:user")
    assert profile["preferred_tickers"] == ["NVDA", "GOOGL"]
    assert "WHAT" not in profile["ticker_interest"]
    assert "WHO" not in profile["ticker_interest"]
    assert "TELL" not in profile["ticker_interest"]
    assert "LOCAL" not in profile["ticker_interest"]


def test_autonomy_status_endpoint_available(monkeypatch):
    os.environ["RAPHI_API_KEY"] = "test-key"
    client = TestClient(raphi_server.app)

    response = client.get("/api/autonomy/status", headers=API_HEADERS)

    assert response.status_code == 200
    payload = response.json()
    assert "global" in payload
    assert "intent_stats" in payload


def test_autonomy_behavior_endpoint_records_user_preferences(monkeypatch, tmp_path):
    os.environ["RAPHI_API_KEY"] = "test-key"
    monkeypatch.setattr(raphi_server, "autonomy", AutonomyController(policy_file=tmp_path / "policy.json"))
    client = TestClient(raphi_server.app)

    response = client.post(
        "/api/autonomy/behavior",
        headers=API_HEADERS,
        json={
            "event_type": "chat_submit",
            "metadata": {
                "ticker": "MSFT",
                "intent": "lookup",
                "response_mode": "compact",
                "is_followup": False,
                "message_length": 42,
                "raw_text": "this should not be stored",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "MSFT" in payload["preferred_tickers"]
    assert "raw_text" not in str(payload)


def test_model_retrain_endpoint_uses_behavior_tickers(monkeypatch, tmp_path):
    os.environ["RAPHI_API_KEY"] = "test-key"
    controller = AutonomyController(policy_file=tmp_path / "policy.json")
    controller.learn_from_behavior(
        user_scope="unit:unit-test-user",
        event_type="chat_message",
        metadata={"ticker": "NVDA", "intent": "memo", "response_mode": "debug"},
    )
    monkeypatch.setattr(raphi_server, "autonomy", controller)

    class FakeMarket:
        def stock_detail(self, ticker):
            return {"ticker": ticker, "pe_ratio": 20, "revenue_growth": 0.12}

    class FakeEngine:
        def __init__(self):
            self.calls = []

        def force_retrain(self, ticker, fundamentals):
            self.calls.append((ticker, fundamentals))
            return {
                "ticker": ticker,
                "direction": "LONG",
                "confidence": 61.0,
                "trained_at": "2026-05-25T00:00:00",
                "n_train": 120,
            }

    class FakeGNN:
        def __init__(self):
            self.calls = []

        def ensure_trained(self, universe, force=False):
            self.calls.append((universe, force))

        def status(self):
            return {"trained": True, "graph_nodes": 1, "graph_edges": 0}

    fake_engine = FakeEngine()
    fake_gnn = FakeGNN()
    monkeypatch.setattr(raphi_server, "market", FakeMarket())
    monkeypatch.setattr(raphi_server, "engine", fake_engine)
    monkeypatch.setattr(raphi_server, "gnn", fake_gnn)
    monkeypatch.setattr(
        raphi_server,
        "optimize_from_conviction_ledger",
        lambda convictions, resolutions: {"applied_updates": 0},
    )
    monkeypatch.setattr(raphi_server, "_append_retraining_record", lambda record: None)
    client = TestClient(raphi_server.app)

    response = client.post(
        "/api/models/retrain",
        headers=ACTION_HEADERS,
        json={"source": "behavior", "background": False, "include_gnn": True, "max_tickers": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "trained"
    assert payload["universe_source"] == "behavior"
    assert payload["tickers"] == ["NVDA"]
    assert fake_engine.calls[0][0] == "NVDA"
    assert fake_gnn.calls == [(["NVDA"], True)]


def test_autonomy_monitor_start_and_stop(monkeypatch):
    os.environ["RAPHI_API_KEY"] = "test-key"
    client = TestClient(raphi_server.app)

    start = client.post(
        "/api/autonomy/monitor/start",
        headers=ACTION_HEADERS,
        json={"ticker": "NVDA", "duration_s": 60, "poll_interval_s": 10, "intent": "monitor"},
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    detail = client.get(f"/api/autonomy/monitor/jobs/{job_id}", headers=API_HEADERS)
    assert detail.status_code == 200
    assert detail.json()["job_id"] == job_id

    stop = client.post(f"/api/autonomy/monitor/jobs/{job_id}/stop", headers=ACTION_HEADERS)
    assert stop.status_code == 200
    assert stop.json()["job_id"] == job_id
