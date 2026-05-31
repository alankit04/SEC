import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RAPHI_API_KEY", "test-key")
from backend.raphi_server import app

_HEADERS = {"X-API-Key": "test-key"}


def test_agentic_research_query():
    client = TestClient(app)
    payload = {
        "query": "Show me SEC filings for NVDA"
    }
    resp = client.post("/api/agentic/query", json=payload, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "final_answer" in data
    # Evidence/citation logic is stubbed, but should not error

def test_agentic_trending_query():
    client = TestClient(app)
    payload = {
        "query": "What are the top trending stocks?"
    }
    resp = client.post("/api/agentic/query", json=payload, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "final_answer" in data
    assert "Top result:" in data["final_answer"]
    assert "Why it ranked first" in data["final_answer"]

def test_agentic_onboarding_query():
    client = TestClient(app)
    payload = {
        "query": "Register NVDA for analysis",
        "tickers": ["NVDA"],
        "universe": ["NVDA", "AAPL"]
    }
    resp = client.post("/api/agentic/query", json=payload, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "ticker_registration_status" in data
    assert "NVDA" in data.get("ticker_registration_status", {})
    assert isinstance(data.get("final_answer", ""), str)
    
