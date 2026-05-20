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
        return {
            "ticker": ticker,
            "price": 100.0,
            "source": "Yahoo Finance via yfinance",
            "quote_url": f"https://finance.yahoo.com/quote/{ticker}",
        }


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


def test_memo_export_markdown_contains_sec_citations(monkeypatch):
    monkeypatch.setattr(
        raphi_server,
        "_build_memo_export",
        lambda ticker: {
            "ticker": ticker,
            "exported_at": "2026-05-19T00:00:00Z",
            "recommendation": "HOLD",
            "confidence": 55,
            "market": {"price": 100.0, "pe_ratio": 20, "sector": "Technology", "industry": "Semiconductors"},
            "sec": {
                "filings": [{
                    "form": "10-Q",
                    "accession": "0001045810-25-000230",
                    "filed": "2025-11-19",
                    "sec_url": "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000230/",
                }],
                "financial_citations": {
                    "revenue": {
                        "form": "10-Q",
                        "accession": "0001045810-25-000230",
                        "filed": "2025-11-19",
                        "sec_url": "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000230/",
                    }
                },
            },
            "gnn": {"direction": "HOLD", "confidence": 54.8, "graph_nodes": 7, "graph_edges": 3},
            "portfolio": {"total_value": 0, "var_95": 0, "sharpe": 0},
            "provenance": {"sec": "SEC Financial Statement Data Sets and SEC EDGAR Archives"},
        },
    )
    client = TestClient(raphi_server.app)

    response = client.get("/api/memo/NVDA/export")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "RAPHI Memo Export: NVDA" in response.text
    assert "0001045810-25-000230" in response.text
    assert "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000230/" in response.text
