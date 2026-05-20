"""
model_optimization.py — local RL, distillation, and quantization utilities.

These features operate on RAPHI's local signal stack. They do not fine-tune,
quantize, or distill hosted LLMs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / ".model_cache"
OPT_DIR = MODEL_DIR / "optimization"
RL_POLICY_FILE = OPT_DIR / "rl_policy.json"


def _ensure_dirs() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OPT_DIR.mkdir(parents=True, exist_ok=True)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def _action_from_direction(direction: str) -> str:
    direction = str(direction or "").upper()
    if direction == "LONG":
        return "LONG"
    if direction == "SHORT":
        return "SHORT"
    return "HOLD"


class DistilledStudent:
    """Small logistic student trained on ensemble soft labels."""

    def __init__(self, learning_rate: float = 0.08, epochs: int = 350, l2: float = 0.002):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X, teacher_prob) -> dict:
        x = np.asarray(X, dtype=np.float64)
        y = np.asarray(teacher_prob, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or len(x) < 20:
            raise ValueError("distillation requires at least 20 feature rows")
        if len(y) != len(x):
            raise ValueError("teacher probabilities must align with X")

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        z = (x - mean) / std

        weights = np.zeros(z.shape[1], dtype=np.float64)
        bias = 0.0
        for _ in range(self.epochs):
            pred = _sigmoid(z @ weights + bias)
            err = pred - y
            grad_w = (z.T @ err) / len(z) + self.l2 * weights
            grad_b = float(err.mean())
            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b

        fitted = {
            "weights": weights.astype(np.float64),
            "bias": float(bias),
            "mean": mean.astype(np.float64),
            "std": std.astype(np.float64),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
        }
        return fitted

    @staticmethod
    def predict_proba(student: dict, X) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        z = (x - np.asarray(student["mean"])) / np.asarray(student["std"])
        p_long = _sigmoid(z @ np.asarray(student["weights"]) + float(student["bias"]))
        return np.column_stack([1.0 - p_long, p_long])


def quantize_student(student: dict, bits: int = 8) -> dict:
    """Symmetric int8 quantization for the distilled student's weights."""
    if bits != 8:
        raise ValueError("only int8 quantization is currently supported")
    weights = np.asarray(student["weights"], dtype=np.float64)
    max_abs = float(np.max(np.abs(weights))) if weights.size else 0.0
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    weights_q = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
    payload = {
        "bits": 8,
        "weight_scale": scale,
        "weights_q": weights_q.tolist(),
        "bias": float(student["bias"]),
        "mean": np.asarray(student["mean"], dtype=np.float32).tolist(),
        "std": np.asarray(student["std"], dtype=np.float32).tolist(),
        "size_bytes": int(weights_q.nbytes + np.asarray(student["mean"], dtype=np.float32).nbytes + np.asarray(student["std"], dtype=np.float32).nbytes + 4),
    }
    return payload


def predict_quantized_student(qstudent: dict, X) -> np.ndarray:
    weights = np.asarray(qstudent["weights_q"], dtype=np.float64) * float(qstudent["weight_scale"])
    student = {
        "weights": weights,
        "bias": float(qstudent["bias"]),
        "mean": np.asarray(qstudent["mean"], dtype=np.float64),
        "std": np.asarray(qstudent["std"], dtype=np.float64),
    }
    return DistilledStudent.predict_proba(student, X)


def save_student_artifacts(ticker: str, student: dict, qstudent: dict) -> dict:
    _ensure_dirs()
    ticker = ticker.upper()
    student_path = OPT_DIR / f"{ticker}_student.pkl"
    quantized_path = OPT_DIR / f"{ticker}_student_int8.json"
    with open(student_path, "wb") as f:
        pickle.dump(student, f)
    with open(quantized_path, "w", encoding="utf-8") as f:
        json.dump(qstudent, f)
    return {
        "student_path": str(student_path.relative_to(BASE_DIR)),
        "quantized_path": str(quantized_path.relative_to(BASE_DIR)),
    }


class ReinforcementPolicy:
    """Contextual bandit-style Q policy learned from resolved convictions."""

    ACTIONS = ("LONG", "SHORT", "HOLD")

    def __init__(self, path: Path = RL_POLICY_FILE, alpha: float = 0.25):
        self.path = path
        self.alpha = alpha
        self.state = {
            "version": 1,
            "updates": 0,
            "q": {},
            "seen_resolutions": [],
        }
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.state.update(loaded)
            except Exception:
                pass

    def save(self) -> None:
        _ensure_dirs()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, sort_keys=True)

    def _bucket(self, ticker: str) -> dict:
        ticker = ticker.upper()
        q = self.state.setdefault("q", {})
        if ticker not in q:
            q[ticker] = {action: 0.0 for action in self.ACTIONS}
        return q[ticker]

    @staticmethod
    def reward_from_resolution(resolution: dict) -> float:
        result = str(resolution.get("ml_result", "")).upper()
        if result == "CONFIRMED":
            base = 1.0
        elif result == "CONTRADICTED":
            base = -1.0
        else:
            base = 0.0
        vs_entry = resolution.get("vs_entry_pct")
        try:
            magnitude = min(abs(float(vs_entry)) / 10.0, 1.0)
        except (TypeError, ValueError):
            magnitude = 0.0
        return round(base * (1.0 + 0.25 * magnitude), 4)

    def update(self, ticker: str, action: str, reward: float, resolution_id: str) -> bool:
        seen = set(self.state.get("seen_resolutions", []))
        if resolution_id in seen:
            return False
        bucket = self._bucket(ticker)
        action = _action_from_direction(action)
        old_q = float(bucket.get(action, 0.0))
        bucket[action] = round(old_q + self.alpha * (float(reward) - old_q), 6)
        self.state["updates"] = int(self.state.get("updates", 0)) + 1
        self.state.setdefault("seen_resolutions", []).append(resolution_id)
        self.state["seen_resolutions"] = self.state["seen_resolutions"][-5000:]
        return True

    def update_from_records(self, convictions: Iterable[dict], resolutions: Iterable[dict]) -> dict:
        convictions_by_id = {c.get("id"): c for c in convictions if c.get("id")}
        applied = 0
        for res in resolutions:
            cid = res.get("conviction_id")
            conv = convictions_by_id.get(cid)
            if not conv:
                continue
            action = conv.get("ml", {}).get("direction", "HOLD")
            ticker = conv.get("ticker", "")
            resolution_id = f"{cid}:{res.get('lookback', '')}"
            reward = self.reward_from_resolution(res)
            if ticker and self.update(ticker, action, reward, resolution_id):
                applied += 1
        if applied:
            self.save()
        return {
            "applied_updates": applied,
            "total_updates": self.state.get("updates", 0),
            "policy_path": _display_path(self.path),
        }

    def q_values(self, ticker: str) -> dict:
        return dict(self._bucket(ticker))

    def adjust_probabilities(self, ticker: str, probs) -> dict:
        raw = np.asarray(probs, dtype=np.float64).reshape(-1)
        if raw.size != 2 or not np.isfinite(raw).all():
            return {"probabilities": probs, "applied": False, "q_values": self.q_values(ticker)}
        p_long = float(raw[1])
        q = self.q_values(ticker)
        bias = 0.08 * (float(q.get("LONG", 0.0)) - float(q.get("SHORT", 0.0)))
        adjusted_long = float(np.clip(p_long + bias, 0.02, 0.98))
        adjusted = np.array([1.0 - adjusted_long, adjusted_long], dtype=np.float64)
        return {
            "probabilities": adjusted,
            "applied": bool(self.state.get("updates", 0) > 0),
            "bias": round(bias, 5),
            "q_values": q,
            "updates": int(self.state.get("updates", 0)),
        }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def optimize_from_conviction_ledger(convictions_file: Path, resolutions_file: Path) -> dict:
    policy = ReinforcementPolicy()
    return policy.update_from_records(load_jsonl(convictions_file), load_jsonl(resolutions_file))


def optimization_status() -> dict:
    policy = ReinforcementPolicy()
    _ensure_dirs()
    quantized = sorted(p.name for p in OPT_DIR.glob("*_student_int8.json"))
    students = sorted(p.name for p in OPT_DIR.glob("*_student.pkl"))
    return {
        "rl_policy": {
            "path": str(RL_POLICY_FILE.relative_to(BASE_DIR)),
            "updates": int(policy.state.get("updates", 0)),
            "tickers": sorted(policy.state.get("q", {}).keys()),
        },
        "distillation": {
            "student_artifacts": students,
            "count": len(students),
        },
        "quantization": {
            "quantized_artifacts": quantized,
            "bits": 8,
            "count": len(quantized),
        },
    }
