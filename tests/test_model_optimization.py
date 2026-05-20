import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from model_optimization import (
    DistilledStudent,
    ReinforcementPolicy,
    predict_quantized_student,
    quantize_student,
)


def test_distilled_student_and_int8_quantized_student_predict_probabilities():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 5))
    teacher = 1.0 / (1.0 + np.exp(-(X[:, 0] - 0.6 * X[:, 1] + 0.25 * X[:, 2])))

    student = DistilledStudent(epochs=120, learning_rate=0.1).fit(X, teacher)
    probs = DistilledStudent.predict_proba(student, X[:4])

    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    qstudent = quantize_student(student)
    qprobs = predict_quantized_student(qstudent, X[:4])

    assert qstudent["bits"] == 8
    assert qstudent["size_bytes"] > 0
    assert qprobs.shape == (4, 2)
    assert np.allclose(qprobs.sum(axis=1), 1.0)
    assert np.mean(np.abs(qprobs[:, 1] - probs[:, 1])) < 0.02


def test_reinforcement_policy_updates_from_conviction_resolutions(tmp_path):
    policy = ReinforcementPolicy(path=tmp_path / "rl_policy.json", alpha=0.5)
    convictions = [
        {"id": "cvx-1", "ticker": "NVDA", "ml": {"direction": "LONG"}},
        {"id": "cvx-2", "ticker": "ASST", "ml": {"direction": "SHORT"}},
    ]
    resolutions = [
        {"conviction_id": "cvx-1", "lookback": "30d", "ml_result": "CONFIRMED", "vs_entry_pct": 8},
        {"conviction_id": "cvx-2", "lookback": "30d", "ml_result": "CONTRADICTED", "vs_entry_pct": -6},
    ]

    result = policy.update_from_records(convictions, resolutions)

    assert result["applied_updates"] == 2
    assert policy.q_values("NVDA")["LONG"] > 0
    assert policy.q_values("ASST")["SHORT"] < 0

    adjusted = policy.adjust_probabilities("NVDA", np.array([0.5, 0.5]))

    assert adjusted["applied"] is True
    assert adjusted["probabilities"][1] > 0.5


def test_reinforcement_policy_is_idempotent_for_same_resolution(tmp_path):
    policy = ReinforcementPolicy(path=tmp_path / "rl_policy.json", alpha=0.5)
    convictions = [{"id": "cvx-1", "ticker": "MSFT", "ml": {"direction": "LONG"}}]
    resolutions = [{"conviction_id": "cvx-1", "lookback": "30d", "ml_result": "CONFIRMED"}]

    first = policy.update_from_records(convictions, resolutions)
    second = policy.update_from_records(convictions, resolutions)

    assert first["applied_updates"] == 1
    assert second["applied_updates"] == 0
    assert policy.state["updates"] == 1
