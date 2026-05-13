import os
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

os.environ.pop("RAPHI_API_KEY", None)
os.environ.pop("SENTRY_DSN", None)

import raphi_server


class FakeMarket:
    def ohlcv(self, ticker, period="1y"):
        idx = pd.date_range("2025-01-01", periods=80, freq="D")
        close = pd.Series(range(100, 180), index=idx, dtype=float)
        return pd.DataFrame(
            {
                "Open": close - 1,
                "High": close + 2,
                "Low": close - 2,
                "Close": close,
                "Volume": pd.Series(range(1_000_000, 1_000_080), index=idx),
            }
        )

    def ticker_price(self, ticker):
        return {"price": 100.0, "change": 1.0, "pct": 1.0}

    def stock_detail(self, ticker):
        return {"ticker": ticker, "price": 100.0}


class FakePortfolio:
    def __init__(self):
        self.positions = [{"ticker": "'; DROP TABLE--"}]

    def get_positions(self):
        return self.positions

    def update_positions(self, positions):
        self.positions = positions

    def snapshot(self):
        return {"positions": self.positions, "total_value": 100.0, "var_95": 2.0, "sharpe": 1.0}


def test_stock_technicals_endpoint_computes_real_indicators(monkeypatch):
    monkeypatch.setattr(raphi_server, "market", FakeMarket())
    client = TestClient(raphi_server.app)

    response = client.get("/api/stock/NVDA/technicals")

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "NVDA"
    assert data["summary"]["trend"] == "bullish"
    assert any(row["name"] == "RSI 14" for row in data["indicators"])


def test_cross_asset_signals_are_generated_from_prices(monkeypatch):
    monkeypatch.setattr(raphi_server, "market", FakeMarket())
    client = TestClient(raphi_server.app)

    response = client.get("/api/cross-asset/signals?asset_class=crypto")

    assert response.status_code == 200
    data = response.json()
    assert data["asset_class"] == "crypto"
    assert {row["ticker"] for row in data["signals"]} >= {"BTC-USD", "ETH-USD"}
    assert all(row["direction"] == "LONG" for row in data["signals"])


def test_portfolio_position_upsert_filters_invalid_saved_rows(monkeypatch):
    fake_market = FakeMarket()
    fake_portfolio = FakePortfolio()
    monkeypatch.setattr(raphi_server, "market", fake_market)
    monkeypatch.setattr(raphi_server, "portfolio", fake_portfolio)
    client = TestClient(raphi_server.app)

    response = client.post("/api/portfolio/positions", json={"ticker": "nvda", "shares": 2})

    assert response.status_code == 200
    assert fake_portfolio.positions == [
        {"ticker": "NVDA", "shares": 2.0, "entry_price": 100.0, "direction": "LONG"}
    ]
