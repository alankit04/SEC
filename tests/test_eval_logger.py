import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import eval_logger  # noqa: E402


def test_build_run_record_has_required_fields():
    record = eval_logger.build_run_record(
        prompt="Analyze NVDA",
        final_response="HOLD with risk framing",
        expected_tools=["market", "sec"],
        observed_tools=["market"],
        citations=[{"url": "https://www.sec.gov/Archives/x"}],
        ticker="NVDA",
    )

    assert record["run_id"].startswith("run_")
    assert record["prompt"] == "Analyze NVDA"
    assert isinstance(record["expected_tools"], list)
    assert isinstance(record["observed_tools"], list)
    assert "timestamp" in record


def test_log_eval_run_writes_json_and_jsonl(tmp_path, monkeypatch):
    run_dir = tmp_path / "eval_runs"
    run_jsonl = tmp_path / "eval_runs.jsonl"
    ledger_jsonl = tmp_path / "immutable_run_ledger.jsonl"

    monkeypatch.setattr(eval_logger, "EVAL_RUN_DIR", run_dir)
    monkeypatch.setattr(eval_logger, "EVAL_RUN_JSONL", run_jsonl)
    monkeypatch.setattr(eval_logger, "IMMUTABLE_LEDGER_JSONL", ledger_jsonl)

    record = eval_logger.build_run_record(
        run_id="run_test123",
        prompt="Memo for NVDA",
        final_response="### Recommendation\nHOLD",
    )
    stored = eval_logger.log_eval_run(record)

    run_file = run_dir / "run_test123.json"
    assert run_file.exists()
    assert run_jsonl.exists()

    payload = json.loads(run_file.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_test123"
    assert payload["prompt"] == "Memo for NVDA"

    lines = [line for line in run_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["run_id"] == "run_test123"
    assert stored["run_id"] == "run_test123"
    assert "immutable_ledger" in stored

    verification = eval_logger.verify_immutable_ledger(ledger_jsonl)
    assert verification["ok"] is True
