"""
market_data.py  —  yfinance wrapper with TTL caching.
Provides real-time prices, fundamentals, news, and historical data.
"""

import math
import time
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf


def _safe_float(v) -> "float | None":
    """Convert yfinance NaN/Inf values to None so JSON serialization doesn't fail."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None

# Try VADER for sentiment; fall back to keyword scoring
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

MARKET_TICKERS = {
    "^GSPC": "sp500",
    "^NDX":  "nasdaq",
    "^VIX":  "vix",
    "^TNX":  "ten_year",
    "GLD":   "gold",
    "DX-Y.NYB": "dxy",
}

PRICE_TTL  = 60      # seconds
FUND_TTL   = 3600    # 1 hour
NEWS_TTL   = 900     # 15 minutes
HIST_TTL   = 3600    # 1 hour


class MarketData:
    def __init__(self):
        self._cache: dict = {}
        self._ts:    dict = {}

    # ------------------------------------------------------------------
    def _fresh(self, key: str, ttl: int) -> bool:
        return key in self._ts and (time.time() - self._ts[key]) < ttl

    def _set(self, key: str, value):
        self._cache[key] = value
        self._ts[key] = time.time()
        return value

    # ------------------------------------------------------------------
    def ticker_price(self, ticker: str) -> dict:
        key = f"p:{ticker}"
        if self._fresh(key, PRICE_TTL):
            return self._cache[key]
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d", interval="1d")
            if len(hist) < 2:
                hist = t.history(period="1mo", interval="1d")
            cur  = _safe_float(hist["Close"].iloc[-1])
            if cur is None:
                return self._set(key, {"price": None, "change": None, "pct": None})
            prev = _safe_float(hist["Close"].iloc[-2]) if len(hist) >= 2 else cur
            prev = prev or cur
            chg  = cur - prev
            pct  = chg / prev * 100 if prev else 0
            return self._set(key, {"price": round(cur, 2),
                                   "change": round(chg, 2),
                                   "pct":    round(pct, 2)})
        except Exception as e:
            return {"price": None, "change": None, "pct": None, "error": str(e)}

    # ------------------------------------------------------------------
    def market_overview(self) -> dict:
        key = "overview"
        if self._fresh(key, PRICE_TTL):
            return self._cache[key]
        result = {}
        for sym, label in MARKET_TICKERS.items():
            result[label] = self.ticker_price(sym)
        from datetime import datetime
        now = datetime.now()
        result["market_open"] = now.weekday() < 5 and 9 <= now.hour < 16
        result["timestamp"]   = now.isoformat()
        return self._set(key, result)

    # ------------------------------------------------------------------
    def stock_detail(self, ticker: str) -> dict:
        key = f"d:{ticker}"
        if self._fresh(key, FUND_TTL):
            return self._cache[key]
        try:
            t    = yf.Ticker(ticker.upper())
            info = t.info or {}
            hist = t.history(period="6mo", interval="1d")

            cur  = _safe_float(hist["Close"].iloc[-1]) if len(hist) > 0 else _safe_float(info.get("currentPrice"))
            cur  = cur or 0.0
            prev = _safe_float(hist["Close"].iloc[-2]) if len(hist) > 1 else cur
            prev = prev or cur
            chg  = cur - prev
            pct  = chg / prev * 100 if prev else 0

            chart = []
            for ts, row in hist.iterrows():
                chart.append({
                    "date":   ts.strftime("%Y-%m-%d"),
                    "open":   round(float(row["Open"]),  2),
                    "high":   round(float(row["High"]),  2),
                    "low":    round(float(row["Low"]),   2),
                    "close":  round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                })

            rev_growth = None
            try:
                fin = t.financials
                if fin is not None and not fin.empty and "Total Revenue" in fin.index:
                    rev = fin.loc["Total Revenue"].dropna()
                    if len(rev) >= 2:
                        rev_growth = round((float(rev.iloc[0]) - float(rev.iloc[1])) / abs(float(rev.iloc[1])) * 100, 1)
            except Exception:
                pass

            result = {
                "ticker":        ticker.upper(),
                "name":          info.get("longName", ticker),
                "price":         round(cur, 2),
                "change":        round(chg, 2),
                "pct":           round(pct, 2),
                "market_cap":    info.get("marketCap") or 0,
                "pe_ratio":      info.get("trailingPE"),
                "forward_pe":    info.get("forwardPE"),
                "revenue":       info.get("totalRevenue") or 0,
                "net_income":    info.get("netIncomeToCommon") or 0,
                "eps":           info.get("trailingEps"),
                "volume":        info.get("volume") or 0,
                "avg_volume":    info.get("averageVolume") or 0,
                "week52_high":   info.get("fiftyTwoWeekHigh") or 0,
                "week52_low":    info.get("fiftyTwoWeekLow") or 0,
                "beta":          info.get("beta"),
                "sector":        info.get("sector", ""),
                "industry":      info.get("industry", ""),
                "short_summary": (info.get("longBusinessSummary") or "")[:400],
                "revenue_growth": rev_growth,
                "chart":         chart,
            }
            return self._set(key, result)
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    # ------------------------------------------------------------------
    def stock_news(self, ticker: str, limit: int = 8) -> list:
        key = f"n:{ticker}"
        if self._fresh(key, NEWS_TTL):
            return self._cache[key]
        try:
            t    = yf.Ticker(ticker.upper())
            raw  = t.news or []
            news = []
            for item in raw[:limit]:
                title = item.get("title", "")
                # Sentiment
                if _VADER and title:
                    sc = _VADER.polarity_scores(title)["compound"]
                    if sc >= 0.05:    sentiment, score = "positive", sc
                    elif sc <= -0.05: sentiment, score = "negative", sc
                    else:             sentiment, score = "neutral",  sc
                else:
                    sentiment, score = _keyword_sentiment(title)

                pub = item.get("providerPublishTime", 0)
                age_h = (time.time() - pub) / 3600 if pub else 9999
                if age_h < 1:    age_str = f"{int(age_h*60)}m ago"
                elif age_h < 24: age_str = f"{int(age_h)}h ago"
                else:            age_str = f"{int(age_h/24)}d ago"

                news.append({
                    "title":     title,
                    "publisher": item.get("publisher", ""),
                    "url":       item.get("link", "#"),
                    "sentiment": sentiment,
                    "score":     round(score, 3),
                    "published": age_str,
                    "tickers":   item.get("relatedTickers", [ticker]),
                })
            return self._set(key, news)
        except Exception:
            return []

    # ------------------------------------------------------------------
    def historical_returns(self, tickers: list, period: str = "1y") -> pd.DataFrame:
        frames = {}
        for t in tickers:
            key = f"ret:{t}:{period}"
            if self._fresh(key, HIST_TTL):
                frames[t] = self._cache[key]
                continue
            try:
                h = yf.Ticker(t).history(period=period)
                if len(h) > 0:
                    ret = h["Close"].pct_change().dropna()
                    self._set(key, ret)
                    frames[t] = ret
            except Exception:
                pass
        return pd.DataFrame(frames)

    # ------------------------------------------------------------------
    def ohlcv(self, ticker: str, period: str = "3y") -> pd.DataFrame:
        key = f"ohlcv:{ticker}:{period}"
        if self._fresh(key, HIST_TTL):
            return self._cache[key]
        try:
            h = yf.Ticker(ticker).history(period=period)
            return self._set(key, h)
        except Exception:
            return pd.DataFrame()


# ------------------------------------------------------------------
def _keyword_sentiment(text: str):
    pos = {"beat", "record", "surge", "rally", "growth", "upgrade",
           "buy", "bullish", "strong", "profit", "gain", "rise"}
    neg = {"miss", "loss", "drop", "cut", "downgrade", "sell", "bearish",
           "weak", "crash", "decline", "fall", "risk", "concern"}
    words = set(text.lower().split())
    p = len(words & pos)
    n = len(words & neg)
    score = (p - n) / max(p + n, 1)
    if score > 0.1:    return "positive", round(score, 3)
    elif score < -0.1: return "negative", round(score, 3)
    else:              return "neutral",  round(score, 3)
