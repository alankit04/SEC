import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import release_gates  # noqa: E402


def test_evaluate_release_passes_for_strong_metrics():
    records = [
        {
            "eval": {"overall_score": 0.9},
            "quality": {
                "unsupported_claim_ratio": 0.01,
                "citation_precision": 0.95,
                "trace_completeness": 1.0,
                "routing_accuracy": 0.98,
            },
        }
        for _ in range(5)
    ]
    result = release_gates.evaluate_release(records)
    assert result["pass"] is True
    assert result["failures"] == []


def test_evaluate_release_fails_for_low_eval_score():
    records = [
        {
            "eval": {"overall_score": 0.2},
            "quality": {
                "unsupported_claim_ratio": 0.10,
                "citation_precision": 0.2,
                "trace_completeness": 0.2,
                "routing_accuracy": 0.2,
            },
        }
    ]
    result = release_gates.evaluate_release(records)
    assert result["pass"] is False
    assert "eval_score_below_threshold" in result["failures"]


def test_load_run_records_ignores_bad_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        json.dumps({"eval": {"overall_score": 1.0}}) + "\n" + "not-json\n",
        encoding="utf-8",
    )
    rows = release_gates.load_run_records(path)
    assert len(rows) == 1
