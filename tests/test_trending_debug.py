import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("RAPHI_API_KEY", "test-key")
from backend.raphi_server import app

client = TestClient(app)
_HEADERS = {"X-API-Key": "test-key"}

def test_trending_debug():
    resp = client.post("/api/agentic/query", json={"query": "What are the top trending stocks in 2026?"}, headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    print("FINAL_ANSWER_DEBUG_START")
    print(data["final_answer"])
    print("FINAL_ANSWER_DEBUG_END")
    assert True  # Always pass
