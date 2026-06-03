"""Tests for Control Plane 1 — the pre-loop input guardrail.

classify_input_bucket(query) is a deterministic, no-LLM domain classifier that
routes a query into "finance" (enter the agentic loop) or "general" (handled
before the loop, so it never produces a state dump).
"""

import os

from backend.input_guardrail import classify_input_bucket

os.environ.setdefault("RAPHI_API_KEY", "test-key")


# ── Finance bucket: one test per signal group in the spec ──────────────

def test_bare_ticker_is_finance():
    assert classify_input_bucket("Analyze NVDA for me") == "finance"


def test_dollar_ticker_is_finance():
    assert classify_input_bucket("thoughts on $PLTR") == "finance"


def test_sec_filing_terms_are_finance():
    assert classify_input_bucket("show me the latest 10-K") == "finance"
    assert classify_input_bucket("any recent 8-k filings?") == "finance"
    assert classify_input_bucket("pull the SEC filing") == "finance"


def test_market_price_stock_are_finance():
    assert classify_input_bucket("what is the stock price of apple") == "finance"
    assert classify_input_bucket("how is the market today") == "finance"


def test_portfolio_var_sharpe_are_finance():
    assert classify_input_bucket("what is my portfolio VaR") == "finance"
    assert classify_input_bucket("compute the sharpe ratio") == "finance"


def test_recommendation_phrasing_is_finance():
    assert classify_input_bucket("should I buy this") == "finance"
    assert classify_input_bucket("is it a sell or a hold") == "finance"


def test_model_signal_terms_are_finance():
    assert classify_input_bucket("what does the GNN signal say") == "finance"
    assert classify_input_bucket("run the xgboost prediction") == "finance"


# ── General bucket ─────────────────────────────────────────────────────

def test_greeting_is_general():
    assert classify_input_bucket("Hello, how are you?") == "general"


def test_identity_question_is_general():
    assert classify_input_bucket("who are you?") == "general"


def test_off_topic_is_general():
    assert classify_input_bucket("what's the weather today") == "general"


# ── False-positive guards (word boundaries, not substrings) ────────────

def test_var_substring_does_not_trigger_finance():
    # "various" contains "var" but is not a finance signal
    assert classify_input_bucket("tell me about the various options here") == "general"


def test_empty_query_is_general():
    assert classify_input_bucket("") == "general"
    assert classify_input_bucket("    ") == "general"


# ── Case insensitivity ─────────────────────────────────────────────────

def test_classification_is_case_insensitive():
    assert classify_input_bucket("STOCK PRICE OF TESLA") == "finance"
    assert classify_input_bucket("HELLO THERE") == "general"


# ── Route wiring: general short-circuits the loop, finance enters it ───

def _client():
    from fastapi.testclient import TestClient
    from backend.raphi_server import app
    return TestClient(app)


_HEADERS = {"X-API-Key": "test-key"}


def test_general_query_short_circuits_the_loop():
    resp = _client().post(
        "/api/agentic/query", json={"query": "Hello, how are you?"}, headers=_HEADERS
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "general"
    # The agentic loop must NOT have run for a general query.
    assert data["tool_plan"] == []
    assert data["tool_trace"] == []
    # A non-empty, human response is returned instead of a state dump.
    assert isinstance(data["final_answer"], str) and data["final_answer"].strip()
    assert "No tickers detected" not in data["final_answer"]


def test_finance_query_still_enters_the_loop():
    resp = _client().post(
        "/api/agentic/query", json={"query": "Show me SEC filings for NVDA"}, headers=_HEADERS
    )
    assert resp.status_code == 200
    data = resp.json()
    # Finance queries are not short-circuited as "general".
    assert data["intent"] != "general"
