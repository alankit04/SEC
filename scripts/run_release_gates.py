#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure backend import from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.release_gates import evaluate_release, load_run_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAPHI release gates against eval runs")
    parser.add_argument("--runs", default=str(REPO_ROOT / "eval_runs.jsonl"), help="Path to eval runs jsonl")
    parser.add_argument("--limit", type=int, default=200, help="Use most recent N records")
    parser.add_argument("--thresholds", default="", help="Optional JSON string for threshold overrides")
    args = parser.parse_args()

    records = load_run_records(args.runs)
    if args.limit > 0:
        records = records[-args.limit:]

    overrides = json.loads(args.thresholds) if args.thresholds else None
    result = evaluate_release(records, thresholds=overrides)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
