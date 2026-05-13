"""
portfolio_manager.py  —  Portfolio with real-time P&L and historical VaR.
"""

import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from paths import PORTFOLIO_FILE
except ImportError:  # pragma: no cover - package import path
    from backend.paths import PORTFOLIO_FILE


def _safe_float(v) -> "float | None":
    """Convert NaN/Inf to None so JSON serialization doesn't fail."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None

TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def _load() -> dict:
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {"positions": []}


def _save(data: dict):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
class PortfolioManager:
    def __init__(self):
        self._price_cache:  dict = {}
        self._price_ts:     dict = {}
        self._hist_cache:   dict = {}
        self._hist_ts:      dict = {}
        self._PRICE_TTL = 60
        self._HIST_TTL  = 3600

    # ------------------------------------------------------------------
    def _current_price(self, ticker: str) -> float | None:
        key = ticker
        if key in self._price_ts and (time.time() - self._price_ts[key]) < self._PRICE_TTL:
            return self._price_cache[key]
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if len(hist) == 0:
                return None
            price = _safe_float(hist["Close"].iloc[-1])
            if price is None:
                return None
            self._price_cache[key] = price
            self._price_ts[key]    = time.time()
            return price
        except Exception:
            return None

    def _hist_returns(self, ticker: str) -> pd.Series:
        key = ticker
        if key in self._hist_ts and (time.time() - self._hist_ts[key]) < self._HIST_TTL:
            return self._hist_cache[key]
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            ret  = hist["Close"].pct_change().dropna()
            self._hist_cache[key] = ret
            self._hist_ts[key]    = time.time()
            return ret
        except Exception:
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        data      = _load()
        positions = data.get("positions", [])

        enriched     = []
        total_value  = 0.0
        total_cost   = 0.0
        returns_map  = {}

        for pos in positions:
            ticker  = str(pos.get("ticker", "")).strip().upper()
            if not TICKER_RE.match(ticker):
                continue
            shares  = pos.get("shares", 0)
            entry   = pos.get("entry_price", 0)
            direct  = pos.get("direction", "LONG")

            cur_price = self._current_price(ticker)
            if cur_price is None:
                cur_price = entry

            mv = abs(shares) * cur_price
            if direct == "LONG":
                pnl    = (cur_price - entry) * shares
                pnl_pct = (cur_price - entry) / entry * 100 if entry else 0
            elif direct == "SHORT":
                pnl    = (entry - cur_price) * abs(shares)
                pnl_pct = (entry - cur_price) / entry * 100 if entry else 0
            else:
                pnl     = 0
                pnl_pct = 0

            total_value += mv
            total_cost  += abs(shares) * entry

            ret = self._hist_returns(ticker)
            if len(ret) > 10:
                returns_map[ticker] = ret * (1 if direct == "LONG" else -1)

            enriched.append({
                **pos,
                "ticker": ticker,
                "current_price": round(cur_price, 2),
                "market_value":  round(mv, 0),
                "pnl":           round(pnl, 2),
                "pnl_pct":       round(pnl_pct, 2),
                "weight":        0,   # filled below
            })

        # Weights
        for p in enriched:
            p["weight"] = round(p["market_value"] / total_value * 100, 1) if total_value else 0

        # Total P&L
        total_pnl     = total_value - total_cost
        total_pnl_pct = total_pnl / total_cost * 100 if total_cost else 0

        # VaR (historical simulation)
        var_95, var_99 = self._compute_var(enriched, total_value, returns_map)

        # Sharpe (annualised)
        sharpe = self._compute_sharpe(enriched, returns_map)

        # Portfolio vs SPY alpha
        alpha_pct = self._portfolio_alpha(total_value, total_cost)

        return {
            "positions":     enriched,
            "total_value":   round(total_value,  0),
            "total_cost":    round(total_cost,   0),
            "total_pnl":     round(total_pnl,    2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "var_95":        round(var_95,  0),
            "var_99":        round(var_99,  0),
            "sharpe":        round(sharpe, 2),
            "alpha_pct":     round(alpha_pct, 2),
        }

    # ------------------------------------------------------------------
    def _compute_var(self, positions: list, total_value: float,
                     returns_map: dict) -> tuple[float, float]:
        if not returns_map or total_value == 0:
            return 0, 0
        weights = {}
        for p in positions:
            t = p["ticker"]
            if t in returns_map:
                weights[t] = p["market_value"] / total_value

        tickers = list(weights.keys())
        df      = pd.DataFrame({t: returns_map[t] for t in tickers}).dropna()
        if df.empty:
            return 0, 0

        w       = np.array([weights[t] for t in tickers])
        port_ret = df.values @ w
        var_95  = -np.percentile(port_ret, 5)  * total_value
        var_99  = -np.percentile(port_ret, 1)  * total_value
        return max(var_95, 0), max(var_99, 0)

    def _compute_sharpe(self, positions: list, returns_map: dict) -> float:
        if not returns_map:
            return 0.0
        combined = pd.concat(returns_map.values(), axis=1).mean(axis=1).dropna()
        if combined.std() == 0:
            return 0.0
        return float(combined.mean() / combined.std() * np.sqrt(252))

    def _portfolio_alpha(self, total_value: float, total_cost: float) -> float:
        if total_cost == 0:
            return 0.0
        port_ret = (total_value - total_cost) / total_cost * 100
        try:
            spy = yf.Ticker("SPY").history(period="6mo")
            if len(spy) >= 2:
                spy_ret = (float(spy["Close"].iloc[-1]) - float(spy["Close"].iloc[0])) \
                          / float(spy["Close"].iloc[0]) * 100
                return round(port_ret - spy_ret, 2)
        except Exception:
            pass
        return round(port_ret, 2)

    # ------------------------------------------------------------------
    def update_positions(self, positions: list):
        data = _load()
        clean = []
        for pos in positions:
            ticker = str(pos.get("ticker", "")).strip().upper()
            if not TICKER_RE.match(ticker):
                continue
            clean.append({**pos, "ticker": ticker})
        data["positions"] = clean
        _save(data)

    def get_positions(self) -> list:
        return _load().get("positions", [])
