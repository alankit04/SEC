"""gnn_model.py — GraphSAGE signal engine for RAPHI.

Graph topology
--------------
  • Sector edges    : companies sharing the same 2-digit SIC code (from SEC sub.txt).
  • Correlation edges: |ρ(1-year daily returns)| ≥ CORR_THRESHOLD.
  • Self-loops      : every node includes its own features in aggregation.

Node features (12) — identical to ml_model.FEATURE_NAMES
  rsi_14, macd_hist, bb_pct, mom_5d, mom_20d, mom_50d,
  vol_ratio, ret_5d, ret_20d, volatility, pe_norm, rev_growth

Training target
  Binary: 1 if the ticker's FWD_DAYS forward return exceeds the median forward
  return of its graph-neighbors over the same window (relative outperformance).
  Falls back to absolute sign when a node has no neighbours.

Backends
  • PyG  (torch + torch-geometric) : full 2-layer SAGEConv, trained end-to-end.
  • Fallback (numpy + sklearn)      : neighbourhood aggregation → MLPClassifier.
    This IS genuine 1-layer GraphSAGE: the MLP learns W₁/W₂ after the mean/max
    aggregation step, so graph structure is fully incorporated.
  Both backends expose an identical dict response compatible with
  SignalEngine.train_and_predict() so ensembling is a one-liner.

Caching
  Graph data + trained weights are pickled in MODEL_CACHE_DIR with a 24-h TTL.
  Re-trains automatically when the cache is stale or the ticker set changes.
"""

from __future__ import annotations

import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── optional PyTorch-Geometric ──────────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv  # type: ignore
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from paths import MODEL_CACHE_DIR
except ImportError:
    from backend.paths import MODEL_CACHE_DIR

# ── keep in sync with ml_model.py ──────────────────────────────────────
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

N_FEATURES      = len(FEATURE_NAMES)
CORR_THRESHOLD  = 0.65          # |ρ| threshold for correlation edges
GNN_CACHE_TTL   = 86_400        # 24 h
HIDDEN_DIM      = 64
MAX_GRAPH_SIZE  = 300           # cap on nodes (memory safety)
FWD_DAYS        = 5             # forward-return horizon (match ml_model)
MAX_NEIGHBORS   = 5             # top-N neighbors to report for influence
_TICKER_RE      = re.compile(r"^[A-Z]{1,5}$")


# ══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GraphData:
    """Fully computed graph ready for training and inference."""
    tickers:       list[str]              # node index → ticker symbol
    features:      np.ndarray             # (N, F) latest feature snapshot
    feat_history:  np.ndarray             # (N, T, F) rolling historical snapshots
    label_history: np.ndarray             # (N, T) relative-outperformance labels
    edge_src:      np.ndarray             # (E,) source node indices
    edge_dst:      np.ndarray             # (E,) destination node indices
    adj:           dict[int, list[int]]   # adjacency list (no self-loops)
    sic_map:       dict[str, str]         # ticker → 4-digit SIC string
    built_at:      float = field(default_factory=time.time)

    def node_idx(self, ticker: str) -> Optional[int]:
        try:
            return self.tickers.index(ticker.upper())
        except ValueError:
            return None


# ══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION  (mirrors ml_model._features exactly)
# ══════════════════════════════════════════════════════════════════════

def _compute_features(hist: pd.DataFrame,
                      pe: float = 25.0,
                      rev_growth: float = 0.0) -> pd.DataFrame:
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
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    # Bollinger % position
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["bb_pct"] = (df["Close"] - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)

    # Momentum
    df["mom_5d"]  = df["Close"].pct_change(5)
    df["mom_20d"] = df["Close"].pct_change(20)
    df["mom_50d"] = df["Close"].pct_change(50)

    # Volume ratio
    df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    # Returns & volatility
    df["ret_5d"]    = df["Close"].pct_change(5)
    df["ret_20d"]   = df["Close"].pct_change(20)
    df["volatility"] = df["Close"].pct_change().rolling(20).std()

    # Fundamentals (scalar broadcast — same default treatment as ml_model)
    df["pe_norm"]    = float((pe - 25) / 15)
    df["rev_growth"] = float(rev_growth / 100 if abs(rev_growth) > 1 else rev_growth)

    return df[FEATURE_NAMES].dropna()


def _relative_labels(
    own_close: pd.Series,
    own_idx:   pd.Index,
    peer_close: Optional[pd.DataFrame],
) -> pd.Series:
    """
    For each date in own_idx, label = 1 if the ticker's FWD_DAYS forward
    return exceeds the median forward return of its peer nodes over the same
    window. Falls back to absolute sign when there are no peers.
    """
    fwd = own_close.pct_change(FWD_DAYS).shift(-FWD_DAYS).reindex(own_idx)
    if peer_close is not None and not peer_close.empty:
        peer_fwd = peer_close.pct_change(FWD_DAYS).shift(-FWD_DAYS)
        peer_med = peer_fwd.median(axis=1).reindex(own_idx)
        return (fwd > peer_med).astype(int).dropna()
    return (fwd > 0).astype(int).dropna()


# ══════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════

class GraphBuilder:
    """
    Constructs a GraphData from:
      1. SIC sector codes (read from the local SEC sub.txt files via SECData).
      2. 1-year price history fetched from yfinance.
      3. Pairwise return correlation (threshold = CORR_THRESHOLD).
    """

    def __init__(self, sec_data=None):
        self._sec = sec_data

    # ------------------------------------------------------------------
    def _sic_for_ticker(self, ticker: str) -> str:
        """Return 4-digit SIC string, or '0000' if not found."""
        if self._sec is None:
            return "0000"
        cik = self._sec.cik_for_ticker(ticker)
        if not cik:
            return "0000"
        try:
            from sec_data import QUARTERS
        except ImportError:
            from backend.sec_data import QUARTERS
        for q in reversed(QUARTERS):
            df = self._sec._load_sub(q)
            if df.empty or "cik" not in df.columns or "sic" not in df.columns:
                continue
            mask = df["cik"].apply(
                lambda x: str(int(x)) == cik if pd.notna(x) else False
            )
            rows = df[mask]
            if not rows.empty:
                raw = str(rows.iloc[0].get("sic", "0")).split(".")[0]
                return raw.zfill(4)
        return "0000"

    # ------------------------------------------------------------------
    def build(self, tickers: list[str]) -> GraphData:
        tickers = [t.upper() for t in tickers[:MAX_GRAPH_SIZE]]
        logger.info("GNN GraphBuilder: fetching history for %d tickers", len(tickers))

        # ── 1. Fetch 1-year price history ────────────────────────────
        hist_map: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                h = yf.Ticker(t).history(period="1y")
                if len(h) >= 60:
                    hist_map[t] = h
            except Exception:
                pass

        valid = [t for t in tickers if t in hist_map]
        if len(valid) < 2:
            raise RuntimeError(
                "GNN needs ≥ 2 tickers with sufficient price history. "
                f"Got {len(valid)} from {len(tickers)} requested."
            )

        # ── 2. SIC codes from SEC data ───────────────────────────────
        sic_map = {t: self._sic_for_ticker(t) for t in valid}

        # ── 3. Align all histories to common trading dates ───────────
        common_dates: Optional[pd.Index] = None
        for t in valid:
            feats = _compute_features(hist_map[t])
            common_dates = feats.index if common_dates is None else \
                           common_dates.intersection(feats.index)

        if common_dates is None or len(common_dates) < 30:
            raise RuntimeError(
                "GNN: fewer than 30 aligned trading days across tickers."
            )

        # ── 4. Build feature history (N, T, F) ───────────────────────
        feat_arrays: list[np.ndarray] = []
        for t in valid:
            feats = _compute_features(hist_map[t]).reindex(common_dates)
            feats = feats.ffill().fillna(0.0)
            feat_arrays.append(feats.values)                # (T, F)

        feat_history = np.stack(feat_arrays, axis=0).astype(np.float32)  # (N, T, F)
        latest_feats = feat_history[:, -1, :]                             # (N, F)

        # ── 5. Aligned close prices for label + correlation ──────────
        close_df = pd.DataFrame(
            {t: hist_map[t]["Close"].reindex(common_dates) for t in valid}
        ).ffill().bfill()

        # ── 6. Edges: sector + correlation ──────────────────────────
        ret_df   = close_df.pct_change().fillna(0.0)
        corr_mat = ret_df.corr().values  # (N, N)

        undirected: set[tuple[int, int]] = set()
        for i, ti in enumerate(valid):
            for j, tj in enumerate(valid):
                if j <= i:
                    continue
                # Sector edge
                if sic_map[ti][:2] == sic_map[tj][:2] != "00":
                    undirected.add((i, j))
                # Correlation edge
                elif abs(corr_mat[i, j]) >= CORR_THRESHOLD:
                    undirected.add((i, j))

        # Build directed edge arrays (both directions) + self-loops
        edge_src_list: list[int] = []
        edge_dst_list: list[int] = []
        for i, j in undirected:
            edge_src_list += [i, j]
            edge_dst_list += [j, i]
        for i in range(len(valid)):   # self-loops for SAGEConv
            edge_src_list.append(i)
            edge_dst_list.append(i)

        edge_src = np.array(edge_src_list, dtype=np.int64)
        edge_dst = np.array(edge_dst_list, dtype=np.int64)

        # ── 7. Adjacency list (no self-loops, for aggregation) ───────
        adj: dict[int, list[int]] = {i: [] for i in range(len(valid))}
        for i, j in undirected:
            adj[i].append(j)
            adj[j].append(i)

        # ── 8. Relative-outperformance labels (N, T) ─────────────────
        T = len(common_dates)
        label_history = np.zeros((len(valid), T), dtype=np.float32)
        for n, t in enumerate(valid):
            peers = adj.get(n, [])
            peer_close = close_df.iloc[:, peers] if peers else None
            labels = _relative_labels(close_df[t], common_dates, peer_close)
            label_history[n] = labels.reindex(common_dates).fillna(0).values

        logger.info(
            "GNN graph built: %d nodes, %d undirected edges",
            len(valid), len(undirected),
        )
        return GraphData(
            tickers      = valid,
            features     = latest_feats,
            feat_history = feat_history,
            label_history= label_history,
            edge_src     = edge_src,
            edge_dst     = edge_dst,
            adj          = adj,
            sic_map      = sic_map,
        )


# ══════════════════════════════════════════════════════════════════════
# BACKEND: NUMPY / SKLEARN  (always available)
# ══════════════════════════════════════════════════════════════════════

class _NumpySAGE:
    """
    1-layer GraphSAGE implemented with numpy aggregation + sklearn MLP.

    For each (node, time-step):
        agg_mean[n, t] = mean( feat[neighbors(n), t], axis=0 )
        agg_max [n, t] = max ( feat[neighbors(n), t], axis=0 )
        z[n, t]        = concat( feat[n,t], agg_mean[n,t], agg_max[n,t] )  → (3·F,)

    An MLPClassifier is then trained on z with labels = relative-outperformance.
    This is mathematically equivalent to training the W₁/W₂ matrices of GraphSAGE
    (the MLP *is* the learned combination after aggregation).
    """

    def __init__(self, hidden: tuple = (128, 64), random_state: int = 42):
        self.mlp    = MLPClassifier(
            hidden_layer_sizes = hidden,
            max_iter           = 500,
            random_state       = random_state,
            early_stopping     = True,
            validation_fraction= 0.15,
            n_iter_no_change   = 15,
        )
        self.scaler  = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def _aggregate(self, feat_mat: np.ndarray,
                   adj: dict[int, list[int]]) -> np.ndarray:
        """
        feat_mat: (N, F)  →  returns (N, 3F)  [own | mean_nbr | max_nbr]
        Nodes with no neighbours fall back to their own features for agg.
        """
        N, F     = feat_mat.shape
        mean_agg = np.empty_like(feat_mat)
        max_agg  = np.empty_like(feat_mat)

        for n in range(N):
            nbrs = adj.get(n, [])
            if nbrs:
                nbr_f       = feat_mat[nbrs]        # (|N(n)|, F)
                mean_agg[n] = nbr_f.mean(axis=0)
                max_agg[n]  = nbr_f.max(axis=0)
            else:
                mean_agg[n] = feat_mat[n]
                max_agg[n]  = feat_mat[n]

        return np.concatenate([feat_mat, mean_agg, max_agg], axis=1)  # (N, 3F)

    # ------------------------------------------------------------------
    def fit(self, graph: GraphData) -> "_NumpySAGE":
        N, T, F = graph.feat_history.shape
        usable_T = T - FWD_DAYS - 1     # drop tail where labels are NaN

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        for t in range(usable_T):
            feat_t  = graph.feat_history[:, t, :]   # (N, F)
            z_t     = self._aggregate(feat_t, graph.adj)  # (N, 3F)
            label_t = graph.label_history[:, t]           # (N,)
            X_parts.append(z_t)
            y_parts.append(label_t)

        X = np.vstack(X_parts)           # (N * usable_T, 3F)
        y = np.concatenate(y_parts)      # (N * usable_T,)

        # Drop rows with non-finite values or NaN labels
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[valid_mask], y[valid_mask]

        if len(np.unique(y)) < 2:
            raise RuntimeError(
                "GNN training labels contain only one class — "
                "check that forward-return data spans multiple periods."
            )

        X_scaled = self.scaler.fit_transform(X)
        self.mlp.fit(X_scaled, y.astype(int))
        self._fitted = True
        logger.info(
            "NumpySAGE trained on %d samples (%d nodes × %d time steps), "
            "val-accuracy ~ %.1f%%",
            len(X), N, usable_T,
            self.mlp.validation_scores_[-1] * 100 if hasattr(self.mlp, "validation_scores_") else float("nan"),
        )
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, feat_mat: np.ndarray,
                      adj: dict[int, list[int]]) -> np.ndarray:
        """feat_mat: (N, F)  →  (N, 2)  class probabilities."""
        z     = self._aggregate(feat_mat, adj)     # (N, 3F)
        z_sc  = self.scaler.transform(z)
        proba = self.mlp.predict_proba(z_sc)       # (N, 2) or (N, 1)
        if proba.shape[1] == 1:
            # Degenerate case: only one class seen during training
            p = proba[:, 0:1]
            proba = np.concatenate([1 - p, p], axis=1)
        return proba                               # (N, 2)

    # ------------------------------------------------------------------
    def neighbor_influence(self, node_idx: int,
                           feat_mat: np.ndarray,
                           adj: dict[int, list[int]]) -> dict[int, float]:
        """
        Ablation-based influence score: how much does each neighbour's
        feature vector contribute to this node's LONG probability?
        A positive score means the neighbour pushes the prediction toward LONG.
        """
        if not self._fitted:
            return {}

        base_p = self.predict_proba(feat_mat, adj)[node_idx, 1]
        scores: dict[int, float] = {}

        for nbr in adj.get(node_idx, []):
            ablated          = feat_mat.copy()
            ablated[nbr]     = 0.0          # zero out neighbour's features
            ablated_p        = self.predict_proba(ablated, adj)[node_idx, 1]
            scores[nbr]      = float(base_p - ablated_p)

        return scores


# ══════════════════════════════════════════════════════════════════════
# BACKEND: PYTORCH-GEOMETRIC  (optional, faster + deeper)
# ══════════════════════════════════════════════════════════════════════

if HAS_PYG:
    class _TorchSAGENet(torch.nn.Module):   # type: ignore[misc]
        """2-layer SAGEConv with BatchNorm and dropout."""

        def __init__(self, in_ch: int = N_FEATURES,
                     hid_ch: int = HIDDEN_DIM,
                     out_ch: int = 2):
            super().__init__()
            self.conv1 = SAGEConv(in_ch,  hid_ch)
            self.conv2 = SAGEConv(hid_ch, hid_ch)
            self.bn1   = torch.nn.BatchNorm1d(hid_ch)
            self.bn2   = torch.nn.BatchNorm1d(hid_ch)
            self.head  = torch.nn.Linear(hid_ch, out_ch)

        def forward(self, x, edge_index):
            x = F.relu(self.bn1(self.conv1(x, edge_index)))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.bn2(self.conv2(x, edge_index)))
            return self.head(x)

    class _TorchSAGE:
        """Trains and wraps a _TorchSAGENet on the full historical graph."""

        def __init__(self):
            self.net    = _TorchSAGENet()
            self.scaler = StandardScaler()
            self._fitted = False

        def fit(self, graph: GraphData) -> "_TorchSAGE":
            N, T, F = graph.feat_history.shape
            usable_T = T - FWD_DAYS - 1

            # Flatten history for scaler fit
            flat = graph.feat_history[:, :usable_T, :].reshape(-1, F)
            self.scaler.fit(flat)

            edge_index = torch.tensor(
                np.stack([graph.edge_src, graph.edge_dst], axis=0),
                dtype=torch.long,
            )

            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3,
                                         weight_decay=1e-4)
            criterion = torch.nn.CrossEntropyLoss()

            self.net.train()
            for epoch in range(200):
                total_loss = 0.0
                for t in range(usable_T):
                    feat_t  = graph.feat_history[:, t, :]       # (N, F)
                    feat_sc = self.scaler.transform(feat_t)
                    x       = torch.tensor(feat_sc, dtype=torch.float32)
                    y       = torch.tensor(
                        graph.label_history[:, t], dtype=torch.long
                    )
                    optimizer.zero_grad()
                    logits = self.net(x, edge_index)
                    loss   = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 50 == 0:
                    logger.debug("TorchSAGE epoch %d  loss=%.4f", epoch,
                                 total_loss / usable_T)

            self._fitted = True
            return self

        def predict_proba(self, feat_mat: np.ndarray,
                          adj: dict[int, list[int]]) -> np.ndarray:
            # Build edge_index from adj (no self-loops needed — SAGEConv adds them)
            srcs, dsts = [], []
            for src, nbrs in adj.items():
                for dst in nbrs:
                    srcs.append(src)
                    dsts.append(dst)
            if srcs:
                edge_index = torch.tensor(
                    [srcs, dsts], dtype=torch.long
                )
            else:
                N = feat_mat.shape[0]
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            feat_sc = self.scaler.transform(feat_mat)
            x       = torch.tensor(feat_sc, dtype=torch.float32)
            self.net.eval()
            with torch.no_grad():
                logits = self.net(x, edge_index)
                return F.softmax(logits, dim=-1).numpy()   # (N, 2)

        def neighbor_influence(self, node_idx: int,
                               feat_mat: np.ndarray,
                               adj: dict[int, list[int]]) -> dict[int, float]:
            """Gradient-based influence (∂p_LONG/∂feat_nbr · feat_nbr)."""
            if not self._fitted:
                return {}

            srcs, dsts = [], []
            for src, nbrs in adj.items():
                for dst in nbrs:
                    srcs.append(src)
                    dsts.append(dst)
            edge_index = (torch.tensor([srcs, dsts], dtype=torch.long)
                          if srcs else torch.zeros((2, 0), dtype=torch.long))

            feat_sc = self.scaler.transform(feat_mat)
            x       = torch.tensor(feat_sc, dtype=torch.float32,
                                   requires_grad=True)
            self.net.eval()
            logits = self.net(x, edge_index)
            p_long = F.softmax(logits, dim=-1)[node_idx, 1]
            p_long.backward()

            grads = x.grad.detach().numpy()         # (N, F)
            scores: dict[int, float] = {}
            for nbr in adj.get(node_idx, []):
                # Hadamard magnitude of gradient × feature value
                scores[nbr] = float(np.abs(grads[nbr] * feat_sc[nbr]).sum())
            return scores


# ══════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ══════════════════════════════════════════════════════════════════════

class GNNSignalEngine:
    """
    Singleton GNN engine.  Call GNNSignalEngine.get(sec_data) to obtain it.

    Key methods
    -----------
    ensure_trained(tickers)          Build graph + train if cache is stale.
    predict(ticker, tickers)         Single-ticker result dict (same shape as
                                     SignalEngine.train_and_predict output).
    predict_batch(tickers)           One graph forward pass → dict[ticker, result].
    invalidate()                     Force re-train on next call.
    """

    _instance: Optional["GNNSignalEngine"] = None

    def __init__(self, sec_data=None):
        MODEL_CACHE_DIR.mkdir(exist_ok=True)
        self._builder    = GraphBuilder(sec_data)
        self._graph:     Optional[GraphData]           = None
        self._model:     Optional[_NumpySAGE]          = None  # or _TorchSAGE
        self._built_at:  float                         = 0.0
        self._cache_path = MODEL_CACHE_DIR / "gnn_state.pkl"
        self._load_cache()

    # ------------------------------------------------------------------
    @classmethod
    def get(cls, sec_data=None) -> "GNNSignalEngine":
        if cls._instance is None:
            cls._instance = cls(sec_data)
        elif sec_data is not None and cls._instance._builder._sec is None:
            cls._instance._builder._sec = sec_data
        return cls._instance

    # ------------------------------------------------------------------
    def _stale(self) -> bool:
        return (time.time() - self._built_at) > GNN_CACHE_TTL

    def _normalize_tickers(self, tickers: list[str]) -> list[str]:
        """Uppercase, validate, dedupe, and cap tickers for graph construction."""
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in tickers:
            ticker = str(raw).strip().upper()
            if not ticker:
                continue
            if not _TICKER_RE.match(ticker):
                raise ValueError(
                    f"Invalid ticker '{ticker}'. Must be 1-5 uppercase letters."
                )
            if ticker not in seen:
                normalized.append(ticker)
                seen.add(ticker)
            if len(normalized) >= MAX_GRAPH_SIZE:
                break
        return normalized

    def _covers_tickers(self, tickers: list[str]) -> bool:
        if self._graph is None:
            return False
        graph_tickers = set(self._graph.tickers)
        return set(tickers).issubset(graph_tickers)

    def _load_cache(self):
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "rb") as fh:
                    state          = pickle.load(fh)
                self._graph        = state["graph"]
                self._model        = state["model"]
                self._built_at     = state["built_at"]
                age_h = (time.time() - self._built_at) / 3600
                logger.info("GNN: loaded cached model (%.1f h old)", age_h)
            except Exception as exc:
                logger.warning("GNN cache load failed (%s) — will retrain.", exc)

    def _save_cache(self):
        try:
            with open(self._cache_path, "wb") as fh:
                pickle.dump({
                    "graph":    self._graph,
                    "model":    self._model,
                    "built_at": self._built_at,
                }, fh)
            logger.info("GNN: cache saved to %s", self._cache_path)
        except Exception as exc:
            logger.warning("GNN cache save failed: %s", exc)

    # ------------------------------------------------------------------
    def ensure_trained(self, tickers: list[str], force: bool = False):
        """Build graph and train model if the cache is missing or stale."""
        tickers = self._normalize_tickers(tickers)
        if len(tickers) < 2:
            raise ValueError("GNN training needs at least 2 valid tickers.")

        if (
            not force
            and not self._stale()
            and self._model is not None
            and self._covers_tickers(tickers)
        ):
            return

        logger.info("GNN: building graph for %d tickers …", len(tickers))
        self._graph    = self._builder.build(tickers)
        ModelCls       = _TorchSAGE if HAS_PYG else _NumpySAGE  # type: ignore[assignment]
        self._model    = ModelCls().fit(self._graph)
        self._built_at = time.time()
        self._save_cache()
        logger.info(
            "GNN: ready  backend=%s  nodes=%d  edges=%d",
            "torch-geometric" if HAS_PYG else "numpy-sage",
            len(self._graph.tickers),
            int((len(self._graph.edge_src) - len(self._graph.tickers)) / 2),
        )

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """Public status payload used by API and MCP surfaces."""
        if self._graph is None or self._model is None:
            return {
                "trained": False,
                "backend": None,
                "graph_nodes": 0,
                "graph_edges": 0,
                "tickers": [],
                "cache_age_h": None,
                "stale": True,
            }

        age_h = (time.time() - self._built_at) / 3600 if self._built_at else None
        return {
            "trained": True,
            "backend": "torch-geometric" if HAS_PYG else "numpy-sage",
            "graph_nodes": len(self._graph.tickers),
            "graph_edges": int(
                (len(self._graph.edge_src) - len(self._graph.tickers)) / 2
            ),
            "tickers": self._graph.tickers,
            "cache_age_h": round(age_h, 2) if age_h is not None else None,
            "stale": self._stale(),
            "trained_at": pd.Timestamp.fromtimestamp(
                self._built_at
            ).isoformat() if self._built_at else None,
        }

    # ------------------------------------------------------------------
    def predict(self, ticker: str, tickers: list[str]) -> dict:
        """
        Return a result dict for *ticker* using the GNN.

        The dict is shaped identically to SignalEngine.train_and_predict() so
        it can be merged / ensembled directly.
        """
        try:
            ticker = ticker.strip().upper()
            tickers = self._normalize_tickers([ticker, *tickers])
            self.ensure_trained(tickers)
        except Exception as exc:
            logger.error("GNN.predict training error: %s", exc)
            return {"error": str(exc), "ticker": ticker}

        if self._graph is None or self._model is None:
            return {"error": "GNN not trained", "ticker": ticker}

        idx = self._graph.node_idx(ticker)
        if idx is None:
            return {"error": f"{ticker} not in GNN graph", "ticker": ticker}

        feat_mat = self._graph.features       # (N, F)
        adj      = self._graph.adj
        probas   = self._model.predict_proba(feat_mat, adj)   # (N, 2)
        p        = probas[idx]                                  # [p_hold, p_long]

        direction  = (
            "LONG"  if p[1] > 0.55  else
            "SHORT" if p[0] > 0.55  else "HOLD"
        )
        confidence = round(float(max(p)) * 100, 1)

        # Neighbor influence (top-MAX_NEIGHBORS by absolute influence)
        raw_infl = self._model.neighbor_influence(idx, feat_mat, adj)
        neighbors = []
        for nbr_idx, infl in sorted(
            raw_infl.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:MAX_NEIGHBORS]:
            nbr_t = self._graph.tickers[nbr_idx]
            neighbors.append({
                "ticker":    nbr_t,
                "sic":       self._graph.sic_map.get(nbr_t, ""),
                "influence": round(infl, 4),
            })

        return {
            "ticker":      ticker,
            "direction":   direction,
            "confidence":  confidence,
            "gnn_probas":  {
                "long":       round(float(p[1]), 4),
                "hold_short": round(float(p[0]), 4),
            },
            "backend":     "torch-geometric" if HAS_PYG else "numpy-sage",
            "neighbors":   neighbors,
            "sic":         self._graph.sic_map.get(ticker, ""),
            "graph_nodes": len(self._graph.tickers),
            "graph_edges": int(
                (len(self._graph.edge_src) - len(self._graph.tickers)) / 2
            ),
            "trained_at":  pd.Timestamp.fromtimestamp(
                self._built_at
            ).isoformat() if self._built_at else None,
        }

    # ------------------------------------------------------------------
    def predict_batch(self, tickers: list[str]) -> dict[str, dict]:
        """
        Single graph forward pass → returns {ticker: result_dict} for all
        tickers. Far more efficient than calling predict() in a loop.
        """
        try:
            tickers = self._normalize_tickers(tickers)
            self.ensure_trained(tickers)
        except Exception as exc:
            logger.error("GNN.predict_batch training error: %s", exc)
            return {t: {"error": str(exc), "ticker": t} for t in tickers}

        if self._graph is None or self._model is None:
            return {t: {"error": "GNN not trained", "ticker": t} for t in tickers}

        feat_mat = self._graph.features
        adj      = self._graph.adj
        probas   = self._model.predict_proba(feat_mat, adj)   # (N, 2)

        trained_ts = (
            pd.Timestamp.fromtimestamp(self._built_at).isoformat()
            if self._built_at else None
        )
        results: dict[str, dict] = {}
        for t in tickers:
            t_up = t.upper()
            idx  = self._graph.node_idx(t_up)
            if idx is None:
                results[t_up] = {"error": f"{t_up} not in GNN graph", "ticker": t_up}
                continue
            p = probas[idx]
            results[t_up] = {
                "ticker":    t_up,
                "direction": (
                    "LONG"  if p[1] > 0.55  else
                    "SHORT" if p[0] > 0.55  else "HOLD"
                ),
                "confidence":  round(float(max(p)) * 100, 1),
                "gnn_probas":  {
                    "long":       round(float(p[1]), 4),
                    "hold_short": round(float(p[0]), 4),
                },
                "backend":     "torch-geometric" if HAS_PYG else "numpy-sage",
                "sic":         self._graph.sic_map.get(t_up, ""),
                "graph_nodes": len(self._graph.tickers),
                "graph_edges": int(
                    (len(self._graph.edge_src) - len(self._graph.tickers)) / 2
                ),
                "trained_at":  trained_ts,
            }
        return results

    # ------------------------------------------------------------------
    def invalidate(self):
        """Force graph rebuild + model retrain on the next call."""
        self._built_at = 0.0
        if self._cache_path.exists():
            try:
                self._cache_path.unlink()
            except OSError:
                pass
        logger.info("GNN: cache invalidated — will retrain on next call.")
