import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import gnn_model as gm


class RecordingGraphBuilder:
    def __init__(self):
        self.calls = []

    def build(self, tickers):
        tickers = [t.upper() for t in tickers]
        self.calls.append(tickers)
        n = len(tickers)
        t_steps = gm.FWD_DAYS + 12
        features = np.ones((n, gm.N_FEATURES), dtype=np.float32)
        feat_history = np.ones((n, t_steps, gm.N_FEATURES), dtype=np.float32)
        label_history = np.tile(np.arange(t_steps) % 2, (n, 1)).astype(np.float32)

        edge_src = list(range(n))
        edge_dst = list(range(n))
        adj = {i: [] for i in range(n)}
        if n > 1:
            for i in range(n - 1):
                adj[i].append(i + 1)
                adj[i + 1].append(i)
                edge_src.extend([i, i + 1])
                edge_dst.extend([i + 1, i])

        return gm.GraphData(
            tickers=tickers,
            features=features,
            feat_history=feat_history,
            label_history=label_history,
            edge_src=np.array(edge_src, dtype=np.int64),
            edge_dst=np.array(edge_dst, dtype=np.int64),
            adj=adj,
            sic_map={ticker: "3571" for ticker in tickers},
            built_at=time.time(),
        )


class DeterministicSAGE:
    def fit(self, graph):
        self.graph = graph
        self._fitted = True
        return self

    def predict_proba(self, feat_mat, adj):
        return np.tile(np.array([[0.3, 0.7]], dtype=np.float32), (feat_mat.shape[0], 1))

    def neighbor_influence(self, node_idx, feat_mat, adj):
        return {nbr: 0.12 for nbr in adj.get(node_idx, [])}


def make_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(gm, "MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(gm, "HAS_PYG", False)
    monkeypatch.setattr(gm, "_NumpySAGE", DeterministicSAGE)
    engine = gm.GNNSignalEngine()
    builder = RecordingGraphBuilder()
    engine._builder = builder
    return engine, builder


def test_ensure_trained_reuses_cache_only_when_requested_tickers_are_covered(monkeypatch, tmp_path):
    engine, builder = make_engine(monkeypatch, tmp_path)

    engine.ensure_trained(["aaa", "bbb"])
    engine.ensure_trained(["BBB", "AAA"])
    engine.ensure_trained(["CCC", "AAA"])

    assert builder.calls == [["AAA", "BBB"], ["CCC", "AAA"]]


def test_predict_includes_requested_ticker_and_reports_public_status(monkeypatch, tmp_path):
    engine, builder = make_engine(monkeypatch, tmp_path)

    result = engine.predict("ccc", ["AAA", "BBB"])
    status = engine.status()

    assert builder.calls[-1] == ["CCC", "AAA", "BBB"]
    assert result["ticker"] == "CCC"
    assert result["direction"] == "LONG"
    assert result["confidence"] == 70.0
    assert status["trained"] is True
    assert status["backend"] == "numpy-sage"
    assert status["tickers"] == ["CCC", "AAA", "BBB"]


def test_invalid_ticker_returns_error_without_building_graph(monkeypatch, tmp_path):
    engine, builder = make_engine(monkeypatch, tmp_path)

    result = engine.predict("BRK.B", ["AAA", "BBB"])

    assert "Invalid ticker" in result["error"]
    assert builder.calls == []
