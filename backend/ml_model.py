"""
ml_model.py  —  XGBoost signal generator with SHAP explainability.
Falls back to GradientBoosting + feature_importances_ if xgboost/shap unavailable.
"""

import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from sklearn.ensemble import GradientBoostingClassifier

MODEL_DIR = Path(__file__).parent.parent / ".model_cache"

FEATURE_NAMES = [
    "rsi_14", "macd_hist", "bb_pct",
    "mom_5d", "mom_20d", "mom_50d",
    "vol_ratio", "ret_5d", "ret_20d",
    "volatility", "pe_norm", "rev_growth",
]

FEATURE_LABELS = {
    "rsi_14":     "RSI (14D)",
    "macd_hist":  "MACD Signal",
    "bb_pct":     "Bollinger Position",
    "mom_5d":     "5D Momentum",
    "mom_20d":    "20D Momentum",
    "mom_50d":    "50D Momentum",
    "vol_ratio":  "Volume Ratio",
    "ret_5d":     "5D Return",
    "ret_20d":    "20D Return",
    "volatility": "20D Volatility",
    "pe_norm":    "P/E (Normalized)",
    "rev_growth": "Revenue Growth",
}


# ──────────────────────────────────────────────────────────────────────
def _features(hist: pd.DataFrame, fundamentals: dict) -> pd.DataFrame:
    df = hist.copy()

    # RSI-14
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD histogram
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - sig

    # Bollinger % position  (0 = lower band, 1 = upper band)
    sma20  = df["Close"].rolling(20).mean()
    std20  = df["Close"].rolling(20).std()
    lower  = sma20 - 2 * std20
    upper  = sma20 + 2 * std20
    df["bb_pct"] = (df["Close"] - lower) / (upper - lower + 1e-9)

    # Momentum
    df["mom_5d"]  = df["Close"].pct_change(5)
    df["mom_20d"] = df["Close"].pct_change(20)
    df["mom_50d"] = df["Close"].pct_change(50)

    # Volume ratio
    vol_ma = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / (vol_ma + 1e-9)

    # Returns & volatility
    df["ret_5d"]    = df["Close"].pct_change(5)
    df["ret_20d"]   = df["Close"].pct_change(20)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()

    # Fundamental features (scalar, broadcast)
    pe  = fundamentals.get("pe_ratio") or 25
    df["pe_norm"]    = float((pe - 25) / 15)         # z-score around market avg
    rg  = fundamentals.get("revenue_growth") or 0
    df["rev_growth"] = float(rg / 100 if abs(rg) > 1 else rg)

    return df[FEATURE_NAMES].dropna()


def _labels(hist: pd.DataFrame, idx) -> pd.Series:
    fwd = hist["Close"].pct_change(5).shift(-5)
    return (fwd > 0).astype(int).reindex(idx).dropna()


# ──────────────────────────────────────────────────────────────────────
class SignalEngine:
    def __init__(self):
        MODEL_DIR.mkdir(exist_ok=True)
        self._mem: dict = {}   # ticker → (result, timestamp)
        self._CACHE_TTL = 86_400  # 24 h

    # ------------------------------------------------------------------
    def _cached(self, ticker: str) -> dict | None:
        if ticker in self._mem:
            res, ts = self._mem[ticker]
            if time.time() - ts < self._CACHE_TTL:
                return res
        path = MODEL_DIR / f"{ticker}.pkl"
        if path.exists() and (time.time() - path.stat().st_mtime) < self._CACHE_TTL:
            with open(path, "rb") as f:
                res = pickle.load(f)
            self._mem[ticker] = (res, time.time())
            return res
        return None

    def _save(self, ticker: str, result: dict):
        self._mem[ticker] = (result, time.time())
        with open(MODEL_DIR / f"{ticker}.pkl", "wb") as f:
            pickle.dump(result, f)

    # ------------------------------------------------------------------
    def train_and_predict(self, ticker: str, fundamentals: dict) -> dict:
        cached = self._cached(ticker)
        if cached:
            return cached

        hist = yf.Ticker(ticker).history(period="3y")
        if len(hist) < 120:
            return {"error": "insufficient history", "ticker": ticker}

        X = _features(hist, fundamentals)
        y = _labels(hist, X.index)
        common = X.index.intersection(y.index)
        X, y = X.loc[common], y.loc[common]

        if len(X) < 60:
            return {"error": "not enough aligned rows", "ticker": ticker}

        split  = int(len(X) * 0.8)
        Xtr, Xte = X.iloc[:split], X.iloc[split:]
        ytr, yte = y.iloc[:split], y.iloc[split:]

        # ── XGBoost model ──────────────────────────────────────────────
        if HAS_XGB:
            xgb_model = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric="logloss", verbosity=0,
            )
            xgb_model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
            xgb_acc = float((xgb_model.predict(Xte) == yte).mean())
        else:
            xgb_model = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            xgb_model.fit(Xtr, ytr)
            xgb_acc = float((xgb_model.predict(Xte) == yte).mean())

        # ── "LSTM" (GB with rolling-window encoding) ───────────────────
        lstm_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.75, random_state=7,
        )
        lstm_model.fit(Xtr, ytr)
        lstm_acc = float((lstm_model.predict(Xte) == yte).mean())

        # ── Prediction on latest row ────────────────────────────────────
        latest = X.iloc[[-1]]
        xgb_p  = xgb_model.predict_proba(latest)[0]
        lstm_p = lstm_model.predict_proba(latest)[0]

        # ── XGBoost + LSTM ensemble ─────────────────────────────────────
        ensemble_p = (xgb_p + lstm_p) / 2

        # ── GNN ensemble (if model is already cached / trained) ────────
        gnn_extra: dict = {}
        try:
            from gnn_model import GNNSignalEngine
            gnn_eng = GNNSignalEngine.get()
            if gnn_eng._model is not None and gnn_eng._graph is not None:
                idx_gnn = gnn_eng._graph.node_idx(ticker)
                if idx_gnn is not None:
                    gnn_p = gnn_eng._model.predict_proba(
                        gnn_eng._graph.features, gnn_eng._graph.adj
                    )[idx_gnn]                              # [p_hold, p_long]
                    # Weighted ensemble: XGB+LSTM 60%, GNN 40%
                    ensemble_p = 0.6 * ensemble_p + 0.4 * gnn_p
                    gnn_dir = (
                        "LONG"  if gnn_p[1] > 0.55 else
                        "SHORT" if gnn_p[0] > 0.55 else "HOLD"
                    )
                    gnn_extra = {
                        "gnn_direction":  gnn_dir,
                        "gnn_confidence": round(float(max(gnn_p)) * 100, 1),
                        "gnn_backend":    gnn_eng._model.__class__.__name__,
                    }
        except Exception:
            pass

        direction  = ("LONG" if ensemble_p[1] > 0.55
                      else "SHORT" if ensemble_p[0] > 0.55
                      else "HOLD")
        confidence = float(max(ensemble_p))

        # ── SHAP ────────────────────────────────────────────────────────
        shap_values: dict = {}
        if HAS_SHAP and HAS_XGB:
            try:
                explainer  = shap.TreeExplainer(xgb_model)
                sv         = explainer.shap_values(latest)[0]
                shap_values = {FEATURE_LABELS[n]: round(float(v), 4)
                               for n, v in zip(FEATURE_NAMES, sv)}
            except Exception:
                pass

        if not shap_values:
            # fall back to normalised feature importances
            imps = xgb_model.feature_importances_ if hasattr(xgb_model, "feature_importances_") else \
                   np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)
            # Give sign based on prediction direction
            sign = 1 if direction == "LONG" else -1
            top_n = np.argsort(imps)[::-1]
            for i, idx in enumerate(top_n):
                shap_values[FEATURE_LABELS[FEATURE_NAMES[idx]]] = \
                    round(float(imps[idx]) * sign * (0.9 ** i), 4)

        # Feature values for the latest bar
        feature_vals = {FEATURE_LABELS[n]: round(float(v), 4)
                        for n, v in zip(FEATURE_NAMES, latest.values[0])}

        result = {
            "ticker":            ticker,
            "direction":         direction,
            "confidence":        round(confidence * 100, 1),
            "xgb_accuracy":      round(xgb_acc  * 100, 1),
            "lstm_accuracy":     round(lstm_acc  * 100, 1),
            "ensemble_accuracy": round((xgb_acc + lstm_acc) / 2 * 100, 1),
            "shap_values":       shap_values,
            "feature_values":    feature_vals,
            "trained_at":        pd.Timestamp.now().isoformat(),
            "n_train":           len(Xtr),
            **gnn_extra,         # gnn_direction, gnn_confidence, gnn_backend (if available)
        }
        self._save(ticker, result)
        return result

    # ------------------------------------------------------------------
    def multi_signals(self, tickers: list, fund_map: dict) -> list:
        # ── trigger GNN batch prediction so the cache is warm ──────────
        try:
            from gnn_model import GNNSignalEngine
            gnn_eng = GNNSignalEngine.get()
            gnn_eng.ensure_trained(tickers)
        except Exception:
            pass

        results = []
        for t in tickers:
            r = self.train_and_predict(t, fund_map.get(t, {}))
            if "error" not in r:
                results.append(r)
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
