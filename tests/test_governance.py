import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import governance  # noqa: E402


def test_assess_output_detects_prohibited_certainty():
    report = governance.assess_output("This is guaranteed and risk-free. Buy now.", output_kind="memo")
    assert report["high_risk"] is True
    assert "prohibited_certainty_language" in report["findings"]


def test_review_queue_roundtrip(tmp_path, monkeypatch):
    queue_path = tmp_path / "review_queue.json"
    monkeypatch.setattr(governance, "_QUEUE_PATH", queue_path)

    assessment = governance.assess_output("BUY with confidence.", output_kind="memo")
    item = governance.enqueue_review(
        "run_abc",
        kind="memo",
        user_id="u1",
        role="analyst",
        summary="memo body",
        assessment=assessment,
    )
    assert item["status"] == "pending"

    pending = governance.list_reviews(status="pending")
    assert len(pending) == 1
    assert pending[0]["run_id"] == "run_abc"

    updated = governance.decide_review("run_abc", decision="approved", reviewer="reviewer1", note="ok")
    assert updated is not None
    assert updated["status"] == "approved"
