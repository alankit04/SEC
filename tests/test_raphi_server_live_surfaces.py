import os
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

os.environ.pop("RAPHI_API_KEY", None)
os.environ.pop("SENTRY_DSN", None)
os.environ["RAPHI_API_KEY"] = "test-key"

import raphi_server
from citation_index import CitationDocument, CitationIndex


API_HEADERS = {
    "X-API-Key": "test-key",
    "X-User-Id": "unit-test-user",
    "X-Tenant-Id": "unit",
}

ACTION_HEADERS = {
    **API_HEADERS,
    "X-Action-Approval": "approved",
}


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


def test_stock_detail_endpoint_serializes_nan_chart_values(monkeypatch):
    class NaNMarket(FakeMarket):
        def stock_detail(self, ticker):
            return {
                "ticker": ticker,
                "price": 100.0,
                "chart": [{"date": "2026-01-01", "open": float("nan"), "close": 100.0}],
            }

    monkeypatch.setattr(raphi_server, "market", NaNMarket())
    client = TestClient(raphi_server.app)

    response = client.get("/api/stock/AAPL", headers=API_HEADERS)

    assert response.status_code == 200
    assert response.json()["chart"][0]["open"] is None


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

    response = client.get("/api/stock/NVDA/technicals", headers=API_HEADERS)

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "NVDA"
    assert data["summary"]["trend"] == "bullish"
    assert any(row["name"] == "RSI 14" for row in data["indicators"])


def test_cross_asset_signals_are_generated_from_prices(monkeypatch):
    monkeypatch.setattr(raphi_server, "market", FakeMarket())
    client = TestClient(raphi_server.app)

    response = client.get("/api/cross-asset/signals?asset_class=crypto", headers=API_HEADERS)

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

    response = client.post("/api/portfolio/positions", headers=ACTION_HEADERS, json={"ticker": "nvda", "shares": 2})

    assert response.status_code == 200
    assert fake_portfolio.positions == [
        {"ticker": "NVDA", "shares": 2.0, "entry_price": 100.0, "direction": "LONG"}
    ]


def test_portfolio_position_requires_explicit_side_effect_approval(monkeypatch):
    fake_market = FakeMarket()
    fake_portfolio = FakePortfolio()
    monkeypatch.setattr(raphi_server, "market", fake_market)
    monkeypatch.setattr(raphi_server, "portfolio", fake_portfolio)
    client = TestClient(raphi_server.app)

    response = client.post("/api/portfolio/positions", headers=API_HEADERS, json={"ticker": "nvda", "shares": 2})

    assert response.status_code == 409
    assert "requires explicit approval" in response.json()["detail"]


def test_memo_export_markdown_contains_sec_citations(monkeypatch):
    monkeypatch.setattr(
        raphi_server,
        "_build_memo_export",
        lambda ticker, user_scope="global": {
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

    response = client.get("/api/memo/NVDA/export", headers=API_HEADERS)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "RAPHI Memo Export: NVDA" in response.text
    assert "0001045810-25-000230" in response.text
    assert "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000230/" in response.text


def test_model_optimization_status_endpoint_reports_real_surfaces():
    client = TestClient(raphi_server.app)

    response = client.get("/api/models/optimization", headers=API_HEADERS)

    assert response.status_code == 200
    data = response.json()
    assert set(data) >= {"rl_policy", "distillation", "quantization"}
    assert data["quantization"]["bits"] == 8


def test_stock_optimization_endpoint_reports_policy_and_cached_artifact_state():
    client = TestClient(raphi_server.app)

    response = client.get("/api/stock/NVDA/optimization", headers=API_HEADERS)

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "NVDA"
    assert "q_values" in data["rl_policy"]
    assert "distilled_student" in data
    assert "quantized_student" in data


def test_citations_search_endpoint_uses_local_index(monkeypatch, tmp_path):
    idx = CitationIndex(database_url="", sqlite_path=tmp_path / "citations.sqlite")
    idx.add_document(CitationDocument(
        ticker="ASST",
        source_type="sec_filing",
        title="ASST 8-K Merger Filing",
        url="https://www.sec.gov/Archives/edgar/data/123/000123/",
        text="ASST Strive merger citation from SEC filing evidence.",
    ), user_scope="local:api-key-user")
    monkeypatch.setattr(raphi_server, "citations", idx)
    client = TestClient(raphi_server.app)

    response = client.get("/api/citations/search?q=Strive merger&ticker=ASST", headers=API_HEADERS)

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "local_citation_index"
    assert data["count"] >= 1
    assert data["results"][0]["url"].startswith("https://www.sec.gov/")
