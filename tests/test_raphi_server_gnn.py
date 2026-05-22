import sys
import os
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

os.environ.pop("RAPHI_API_KEY", None)
os.environ.pop("SENTRY_DSN", None)
os.environ["RAPHI_API_KEY"] = "test-key"

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


class FakeGNN:
    def __init__(self):
        self.predict_calls = []
        self.batch_calls = []
        self.train_calls = []

    def predict(self, ticker, universe):
        self.predict_calls.append((ticker, universe))
        return {
            "ticker": ticker,
            "direction": "LONG",
            "confidence": 72.4,
            "neighbors": [{"ticker": "AAA", "influence": 0.12}],
        }

    def predict_batch(self, universe):
        self.batch_calls.append(universe)
        return {
            ticker: {"ticker": ticker, "direction": "LONG", "confidence": 70.0}
            for ticker in universe
        }

    def ensure_trained(self, universe, force=False):
        self.train_calls.append((universe, force))

    def status(self):
        return {
            "trained": True,
            "backend": "numpy-sage",
            "graph_nodes": 3,
            "graph_edges": 2,
            "tickers": ["TSLA", "AAPL", "MSFT"],
            "cache_age_h": 0.1,
            "stale": False,
        }


def test_unified_server_exposes_gnn_prediction_with_requested_ticker(monkeypatch):
    fake = FakeGNN()
    monkeypatch.setattr(raphi_server, "gnn", fake)
    monkeypatch.setattr(raphi_server, "_load_settings_for_scope", lambda _scope: {"watchlist": ["AAPL", "MSFT"]})
    client = TestClient(raphi_server.app)

    response = client.get("/api/stock/tsla/gnn", headers=API_HEADERS)

    assert response.status_code == 200
    assert response.json()["ticker"] == "TSLA"
    assert fake.predict_calls == [("TSLA", ["TSLA", "AAPL", "MSFT"])]


def test_unified_server_exposes_batch_status_and_synchronous_training(monkeypatch):
    fake = FakeGNN()
    monkeypatch.setattr(raphi_server, "gnn", fake)
    monkeypatch.setattr(raphi_server, "_load_settings_for_scope", lambda _scope: {"watchlist": ["AAPL", "GOOGL"]})
    client = TestClient(raphi_server.app)

    batch_response = client.get("/api/gnn/signals?tickers=msft,nvda", headers=API_HEADERS)
    status_response = client.get("/api/gnn/status", headers=API_HEADERS)
    train_response = client.post(
        "/api/gnn/train",
        headers=ACTION_HEADERS,
        json={"tickers": ["msft", "nvda"], "force": False, "background": False},
    )

    assert batch_response.status_code == 200
    assert status_response.status_code == 200
    assert train_response.status_code == 200
    assert fake.batch_calls == [["MSFT", "NVDA", "AAPL", "GOOGL"]]
    assert fake.train_calls == [(["MSFT", "NVDA", "AAPL", "GOOGL"], False)]
    assert train_response.json()["status"] == "trained"
