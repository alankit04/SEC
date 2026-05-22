from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_QUEUE_PATH = _REPO_ROOT / "data" / "review_queue.json"
_LOCK = threading.Lock()

_PROHIBITED_CERTAINTY = [
    "guaranteed",
    "zero risk",
    "can not lose",
    "certain to",
    "risk-free",
]

_UNCERTAINTY_TERMS = [
    "risk",
    "uncertain",
    "uncertainty",
    "confidence",
    "probability",
    "may",
    "could",
]


def _load_queue() -> dict[str, Any]:
    if not _QUEUE_PATH.exists():
        return {"items": []}
    try:
        return json.loads(_QUEUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"items": []}


def _save_queue(data: dict[str, Any]) -> None:
    _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _QUEUE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, _QUEUE_PATH)


def assess_output(text: str, *, output_kind: str = "chat") -> dict[str, Any]:
    lower = (text or "").lower()
    certainty_hits = [phrase for phrase in _PROHIBITED_CERTAINTY if phrase in lower]
    has_uncertainty = any(term in lower for term in _UNCERTAINTY_TERMS)
    has_recommendation = bool(re.search(r"\b(buy|sell|hold|overweight|underweight)\b", lower))

    findings: list[str] = []
    if certainty_hits:
        findings.append("prohibited_certainty_language")
    if has_recommendation and not has_uncertainty:
        findings.append("missing_uncertainty_framing")
    if output_kind == "memo" and has_recommendation and "not investment advice" not in lower:
        findings.append("missing_suitability_disclaimer")

    high_risk = bool(certainty_hits) or (output_kind == "memo" and has_recommendation)
    return {
        "high_risk": high_risk,
        "has_recommendation": has_recommendation,
        "certainty_hits": certainty_hits,
        "has_uncertainty_framing": has_uncertainty,
        "findings": findings,
    }


def enqueue_review(run_id: str, *, kind: str, user_id: str, role: str, summary: str, assessment: dict[str, Any]) -> dict[str, Any]:
    item = {
        "run_id": run_id,
        "kind": kind,
        "user_id": user_id,
        "role": role,
        "summary": summary[:1200],
        "assessment": assessment,
        "status": "pending",
        "created_at": time.time(),
        "updated_at": time.time(),
        "reviewer": "",
        "decision_note": "",
    }
    with _LOCK:
        data = _load_queue()
        data.setdefault("items", []).append(item)
        _save_queue(data)
    return item


def list_reviews(status: str | None = None) -> list[dict[str, Any]]:
    with _LOCK:
        items = list(_load_queue().get("items", []))
    if status:
        items = [i for i in items if i.get("status") == status]
    return items


def decide_review(run_id: str, *, decision: str, reviewer: str, note: str = "") -> dict[str, Any] | None:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    with _LOCK:
        data = _load_queue()
        for item in data.get("items", []):
            if item.get("run_id") == run_id:
                item["status"] = decision
                item["reviewer"] = reviewer
                item["decision_note"] = note[:500]
                item["updated_at"] = time.time()
                _save_queue(data)
                return item
    return None
