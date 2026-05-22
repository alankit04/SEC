from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "min_eval_score": 0.75,
    "max_unsupported_claim_ratio": 0.03,
    "min_citation_precision": 0.85,
    "min_trace_completeness": 0.95,
    "min_routing_accuracy": 0.90,
}


def evaluate_release(records: list[dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)

    if not records:
        return {
            "pass": False,
            "reason": "no_records",
            "thresholds": cfg,
            "metrics": {},
            "failures": ["No evaluation records available"],
        }

    n = len(records)
    eval_scores = [
        float((r.get("eval") or r.get("eval_result") or {}).get("overall_score", 0.0) or 0.0)
        for r in records
    ]
    unsupported = [float(r.get("quality", {}).get("unsupported_claim_ratio", 0.0) or 0.0) for r in records]
    citation_prec = [float(r.get("quality", {}).get("citation_precision", 0.0) or 0.0) for r in records]
    trace_comp = [float(r.get("quality", {}).get("trace_completeness", 0.0) or 0.0) for r in records]
    route_acc = [float(r.get("quality", {}).get("routing_accuracy", 0.0) or 0.0) for r in records]

    metrics = {
        "count": n,
        "avg_eval_score": sum(eval_scores) / n,
        "avg_unsupported_claim_ratio": sum(unsupported) / n,
        "avg_citation_precision": sum(citation_prec) / n,
        "avg_trace_completeness": sum(trace_comp) / n,
        "avg_routing_accuracy": sum(route_acc) / n,
    }

    failures: list[str] = []
    if metrics["avg_eval_score"] < cfg["min_eval_score"]:
        failures.append("eval_score_below_threshold")
    if metrics["avg_unsupported_claim_ratio"] > cfg["max_unsupported_claim_ratio"]:
        failures.append("unsupported_claim_ratio_above_threshold")
    if metrics["avg_citation_precision"] < cfg["min_citation_precision"]:
        failures.append("citation_precision_below_threshold")
    if metrics["avg_trace_completeness"] < cfg["min_trace_completeness"]:
        failures.append("trace_completeness_below_threshold")
    if metrics["avg_routing_accuracy"] < cfg["min_routing_accuracy"]:
        failures.append("routing_accuracy_below_threshold")

    return {
        "pass": not failures,
        "thresholds": cfg,
        "metrics": metrics,
        "failures": failures,
    }


def load_run_records(path: str | Path) -> list[dict[str, Any]]:
    run_path = Path(path)
    if not run_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in run_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records
