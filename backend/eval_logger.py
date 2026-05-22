from __future__ import annotations

import json
import os
import hashlib
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


BASE_DIR = Path(__file__).resolve().parents[1]
EVAL_RUN_DIR = BASE_DIR / "data" / "eval_runs"
EVAL_RUN_JSONL = BASE_DIR / "eval_runs.jsonl"
IMMUTABLE_LEDGER_JSONL = BASE_DIR / "data" / "audit" / "immutable_run_ledger.jsonl"
_lock = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex}"


def build_run_record(**kwargs) -> dict:
    run_id = kwargs.get("run_id") or new_run_id()
    record = {
        "run_id": run_id,
        "timestamp": kwargs.get("timestamp") or utc_now_iso(),
        "prompt": kwargs.get("prompt", ""),
        "final_response": kwargs.get("final_response", ""),
        "expected_tools": list(kwargs.get("expected_tools", [])),
        "observed_tools": list(kwargs.get("observed_tools", [])),
        "tool_trace": list(kwargs.get("tool_trace", [])),
        "citations": list(kwargs.get("citations", [])),
        "ticker": kwargs.get("ticker", ""),
        "allowed_tickers": list(kwargs.get("allowed_tickers", [])),
        "memo_schema": bool(kwargs.get("memo_schema", False)),
        "guardrail_repairs": list(kwargs.get("guardrail_repairs", [])),
        "guardrail_warnings": list(kwargs.get("guardrail_warnings", [])),
        "guardrail_missing_sections": list(kwargs.get("guardrail_missing_sections", [])),
        "guardrail_unknown_tickers": list(kwargs.get("guardrail_unknown_tickers", [])),
        "evidence_enforcement": dict(kwargs.get("evidence_enforcement", {})),
        "thread_id": kwargs.get("thread_id"),
        "session_id": kwargs.get("session_id"),
        "user_id": kwargs.get("user_id", "anonymous"),
        "user_role": kwargs.get("user_role", "analyst"),
        "model_path": kwargs.get("model_path", ""),
        "provider_path": kwargs.get("provider_path", "anthropic/claude-agent-sdk"),
        "latency_ms": int(kwargs.get("latency_ms", 0) or 0),
        "eval_result": kwargs.get("eval_result"),
        "governance": dict(kwargs.get("governance", {})),
        "review": dict(kwargs.get("review", {})),
        "quality": dict(kwargs.get("quality", {})),
    }
    return record


def _append_jsonl(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _last_ledger_hash(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        last = ""
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if not last:
            return ""
        obj = json.loads(last)
        return str(obj.get("hash", ""))
    except Exception:
        return ""


def _compute_ledger_hash(entry_without_hash: dict) -> str:
    payload = json.dumps(entry_without_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_immutable_ledger(run: dict) -> dict:
    prev_hash = _last_ledger_hash(IMMUTABLE_LEDGER_JSONL)
    entry = {
        "run_id": run.get("run_id", ""),
        "timestamp": run.get("timestamp", utc_now_iso()),
        "prev_hash": prev_hash,
        "record_hash": hashlib.sha256(
            json.dumps(run, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    entry["hash"] = _compute_ledger_hash(entry)
    _append_jsonl(IMMUTABLE_LEDGER_JSONL, json.dumps(entry, sort_keys=True) + "\n")
    return entry


def verify_immutable_ledger(path: Path | None = None) -> dict:
    ledger_path = path or IMMUTABLE_LEDGER_JSONL
    if not ledger_path.exists():
        return {"ok": True, "entries": 0, "reason": "missing_ledger"}

    entries = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))

    prev = ""
    for idx, entry in enumerate(entries):
        expected_prev = prev
        if str(entry.get("prev_hash", "")) != expected_prev:
            return {
                "ok": False,
                "entries": len(entries),
                "index": idx,
                "reason": "prev_hash_mismatch",
            }
        hashed = dict(entry)
        got_hash = str(hashed.pop("hash", ""))
        expected_hash = _compute_ledger_hash(hashed)
        if got_hash != expected_hash:
            return {
                "ok": False,
                "entries": len(entries),
                "index": idx,
                "reason": "hash_mismatch",
            }
        prev = got_hash

    return {"ok": True, "entries": len(entries)}


def log_eval_run(record: dict) -> dict:
    run = dict(record)
    run.setdefault("run_id", new_run_id())
    run.setdefault("timestamp", utc_now_iso())
    EVAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_file = EVAL_RUN_DIR / f"{run['run_id']}.json"

    with _lock:
        run["immutable_ledger"] = _append_immutable_ledger(run)
        run_file.write_text(json.dumps(run, indent=2, sort_keys=True), encoding="utf-8")
        _append_jsonl(EVAL_RUN_JSONL, json.dumps(run, sort_keys=True) + "\n")

    return run
