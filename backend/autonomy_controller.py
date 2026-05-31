from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

BASE_DIR = Path(__file__).resolve().parents[1]
AUTONOMY_DIR = BASE_DIR / "data" / "autonomy"
POLICY_FILE = AUTONOMY_DIR / "policy_state.json"

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("execution", re.compile(r"\b(execute|place order|submit order|open position|close position|trade now)\b", re.I)),
    ("rebalance", re.compile(r"\b(rebalance|rebalancing|allocation shift|weight change|position sizing)\b", re.I)),
    ("memo", re.compile(r"\b(memo|investment thesis|recommendation|trade plan|deep dive|full analysis)\b", re.I)),
    ("monitor", re.compile(r"\b(monitor|watch|alert|track|notify|continuous|for 30 minutes|long horizon)\b", re.I)),
    ("compare", re.compile(r"\b(compare|versus|vs\.?|relative|peer|benchmark|cross-check)\b", re.I)),
    ("explain", re.compile(r"\b(why|explain|reason|drivers|what changed|how come)\b", re.I)),
    ("lookup", re.compile(r"\b(price|quote|latest|current|show|lookup|what is)\b", re.I)),
]

FALSE_BEHAVIOR_TICKERS = {
    "A2A", "AI", "API", "GNN", "LLM", "LOCAL", "MCP", "ML", "RAG", "RAPHI", "SEC",
    "WHAT", "WHO", "HOW", "WHY", "WHEN", "WHERE", "WHICH", "TELL", "ASK", "CAN",
    "YOU", "YOUR", "ME", "MY", "ARE", "THE", "THIS", "THAT", "ABOUT", "STOCK",
    "STOCKS", "LATEST", "CURRENT", "BUY", "SELL", "HOLD", "MEMO", "RISK",
    "PRICE", "MODEL", "TRAIN", "RETRAIN", "AFTER", "BEFORE",
}


@dataclass
class PlannerObjective:
    intent: str
    quality_target: float
    latency_budget_ms: int
    cost_budget: str
    critique_loops: int
    tool_budgets: dict[str, int]
    reasoning_mode: str


class AutonomyController:
    """Persistent policy learner for cross-run routing and objective optimization."""

    def __init__(self, policy_file: Path = POLICY_FILE) -> None:
        self.policy_file = policy_file
        self._lock = threading.Lock()
        self.state: dict[str, Any] = {
            "version": 1,
            "updated_at": time.time(),
            "global": {
                "runs": 0,
                "avg_eval_score": 0.0,
                "avg_latency_ms": 0.0,
                "pass_rate": 0.0,
            },
            "intent_stats": {},
            "user_stats": {},
            "tool_reliability": {},
            "confidence_calibration": {
                "bins": [
                    {"low": 0.0, "high": 0.2, "count": 0, "success": 0},
                    {"low": 0.2, "high": 0.4, "count": 0, "success": 0},
                    {"low": 0.4, "high": 0.6, "count": 0, "success": 0},
                    {"low": 0.6, "high": 0.8, "count": 0, "success": 0},
                    {"low": 0.8, "high": 1.01, "count": 0, "success": 0},
                ],
                "brier_score": 0.0,
                "count": 0,
            },
            "oversight": {
                "default_review_required": os.environ.get("RAPHI_REVIEW_REQUIRED", "1") in {"1", "true", "yes"},
                "risk_escalation_threshold": 0.35,
            },
        }
        self._load()

    def _ensure_dir(self) -> None:
        AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self.policy_file.exists():
            return
        try:
            loaded = json.loads(self.policy_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.state.update(loaded)
        except Exception:
            return

    def _save(self) -> None:
        self._ensure_dir()
        tmp = self.policy_file.with_suffix(".tmp")
        payload = dict(self.state)
        payload["updated_at"] = time.time()
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.policy_file)

    def classify_intent(self, message: str) -> str:
        text = str(message or "").strip()
        for label, pattern in _INTENT_PATTERNS:
            if pattern.search(text):
                return label
        return "explain"

    def _intent_bucket(self, intent: str) -> dict[str, Any]:
        stats = self.state.setdefault("intent_stats", {})
        if intent not in stats:
            stats[intent] = {
                "runs": 0,
                "passes": 0,
                "avg_eval_score": 0.0,
                "avg_latency_ms": 0.0,
                "avg_citation_precision": 0.0,
                "avg_confidence_error": 0.0,
                "recent_failures": 0,
            }
        return stats[intent]

    def _user_bucket(self, user_scope: str) -> dict[str, Any]:
        users = self.state.setdefault("user_stats", {})
        if user_scope not in users:
            users[user_scope] = {
                "runs": 0,
                "passes": 0,
                "recent_failures": 0,
                "risk_class": "standard",
                "intent_success": {},
                "behavior": {
                    "events": 0,
                    "event_counts": {},
                    "intent_counts": {},
                    "ticker_interest": {},
                    "page_views": {},
                    "response_modes": {},
                    "followup_count": 0,
                    "last_events": [],
                    "preferred_response_mode": "compact",
                    "preferred_tickers": [],
                    "preferred_pages": [],
                    "preferred_focus": "balanced",
                },
            }
        return users[user_scope]

    def _behavior_bucket(self, user_scope: str) -> dict[str, Any]:
        bucket = self._user_bucket(user_scope)
        behavior = bucket.setdefault("behavior", {})
        defaults = {
            "events": 0,
            "event_counts": {},
            "intent_counts": {},
            "ticker_interest": {},
            "page_views": {},
            "response_modes": {},
            "followup_count": 0,
            "last_events": [],
            "preferred_response_mode": "compact",
            "preferred_tickers": [],
            "preferred_pages": [],
            "preferred_focus": "balanced",
        }
        for key, value in defaults.items():
            behavior.setdefault(key, value.copy() if isinstance(value, (dict, list)) else value)
        return behavior

    def _tool_bucket(self, tool: str) -> dict[str, Any]:
        rel = self.state.setdefault("tool_reliability", {})
        if tool not in rel:
            rel[tool] = {
                "runs": 0,
                "success": 0,
                "avg_latency_ms": 0.0,
                "failure_streak": 0,
                "last_error": "",
            }
        return rel[tool]

    @staticmethod
    def _ema(old: float, new: float, n: int) -> float:
        if n <= 1:
            return float(new)
        alpha = 2.0 / (n + 1.0)
        return (1.0 - alpha) * float(old) + alpha * float(new)

    def calibrated_confidence(self, raw_confidence: float) -> float:
        conf = max(0.0, min(1.0, float(raw_confidence)))
        bins = ((self.state.get("confidence_calibration") or {}).get("bins") or [])
        for item in bins:
            low = float(item.get("low", 0.0))
            high = float(item.get("high", 1.0))
            if low <= conf < high:
                count = int(item.get("count", 0) or 0)
                if count < 8:
                    return conf
                empirical = float(item.get("success", 0) or 0) / max(count, 1)
                return max(0.0, min(1.0, 0.35 * conf + 0.65 * empirical))
        return conf

    def objective_for(
        self,
        *,
        message: str,
        intent: str,
        user_scope: str,
        user_role: str,
        provider_status: dict[str, Any] | None = None,
    ) -> PlannerObjective:
        intent_stats = self._intent_bucket(intent)
        user_stats = self._user_bucket(user_scope)

        pass_rate = (intent_stats.get("passes", 0) / max(intent_stats.get("runs", 1), 1))
        avg_latency = float(intent_stats.get("avg_latency_ms", 0.0) or 0.0)
        degraded = bool(provider_status and provider_status.get("status") == "degraded")

        quality_target = 0.84
        latency_budget = 4200
        cost_budget = "medium"
        critique_loops = 2
        reasoning_mode = "balanced"
        tool_budgets = {
            "market": 1,
            "sec": 1,
            "citation": 1,
            "ml": 1,
            "gnn": 1,
            "portfolio": 1,
            "memory": 1,
        }

        if intent in {"lookup"}:
            quality_target = 0.76
            latency_budget = 1800
            cost_budget = "low"
            critique_loops = 1
            reasoning_mode = "fast"
            tool_budgets.update({"sec": 0, "gnn": 0, "portfolio": 0})
        elif intent in {"compare", "memo", "rebalance", "execution"}:
            quality_target = 0.9
            latency_budget = 6500
            cost_budget = "high"
            critique_loops = 3
            reasoning_mode = "thorough"
            tool_budgets.update({"sec": 2, "citation": 2, "portfolio": 2})
        elif intent == "monitor":
            quality_target = 0.86
            latency_budget = 3000
            cost_budget = "medium"
            critique_loops = 1
            reasoning_mode = "monitor"
            tool_budgets.update({"market": 2, "sec": 1, "citation": 1, "memory": 2})

        if pass_rate < 0.7:
            quality_target = min(0.95, quality_target + 0.04)
            critique_loops = min(4, critique_loops + 1)
            tool_budgets["citation"] = max(tool_budgets.get("citation", 1), 2)

        if avg_latency > 0 and avg_latency > latency_budget:
            latency_budget = int(avg_latency * 0.9)
            reasoning_mode = "latency_constrained"
            tool_budgets["citation"] = max(1, tool_budgets.get("citation", 1) - 1)

        if degraded:
            latency_budget = int(latency_budget * 0.8)
            tool_budgets["citation"] = max(0, tool_budgets.get("citation", 1) - 1)
            cost_budget = "low"

        if user_role == "viewer":
            quality_target = max(quality_target, 0.88)
            critique_loops = max(critique_loops, 2)

        if int(user_stats.get("recent_failures", 0)) >= 3:
            quality_target = min(0.96, quality_target + 0.03)
            critique_loops = min(4, critique_loops + 1)

        behavior = self._behavior_bucket(user_scope)
        preferred_focus = str(behavior.get("preferred_focus") or "balanced")
        if preferred_focus == "evidence_depth":
            quality_target = min(0.96, quality_target + 0.025)
            critique_loops = min(4, critique_loops + 1)
            tool_budgets["citation"] = max(tool_budgets.get("citation", 1), 2)
            reasoning_mode = "evidence_depth" if reasoning_mode == "balanced" else reasoning_mode
        elif preferred_focus == "speed" and intent in {"lookup", "explain"}:
            latency_budget = int(latency_budget * 0.85)
            critique_loops = max(1, critique_loops - 1)
            cost_budget = "low"
            reasoning_mode = "fast"

        return PlannerObjective(
            intent=intent,
            quality_target=round(quality_target, 3),
            latency_budget_ms=max(1000, int(latency_budget)),
            cost_budget=cost_budget,
            critique_loops=critique_loops,
            tool_budgets=tool_budgets,
            reasoning_mode=reasoning_mode,
        )

    def recommend_tool_fallbacks(self, required_tools: list[str]) -> list[str]:
        scored: list[tuple[float, str]] = []
        for tool in required_tools:
            bucket = self._tool_bucket(tool)
            runs = max(int(bucket.get("runs", 0) or 0), 1)
            success = int(bucket.get("success", 0) or 0)
            success_rate = success / runs
            streak = int(bucket.get("failure_streak", 0) or 0)
            score = (1.0 - success_rate) + min(0.4, 0.1 * streak)
            scored.append((score, tool))
        scored.sort(reverse=True)
        return [tool for _, tool in scored]

    def should_require_human_review(self, user_scope: str, role: str) -> bool:
        bucket = self._user_bucket(user_scope)
        default_required = bool((self.state.get("oversight") or {}).get("default_review_required", True))
        if role in {"viewer"}:
            return True
        recent_failures = int(bucket.get("recent_failures", 0) or 0)
        if recent_failures >= 2:
            return True
        return default_required

    def _update_confidence_calibration(self, predicted: float, outcome: int) -> float:
        predicted = max(0.0, min(1.0, float(predicted)))
        calibration = self.state.setdefault("confidence_calibration", {})
        bins = calibration.setdefault("bins", [])
        for item in bins:
            low = float(item.get("low", 0.0))
            high = float(item.get("high", 1.0))
            if low <= predicted < high:
                item["count"] = int(item.get("count", 0) or 0) + 1
                item["success"] = int(item.get("success", 0) or 0) + int(outcome)
                break
        n = int(calibration.get("count", 0) or 0) + 1
        prev = float(calibration.get("brier_score", 0.0) or 0.0)
        brier = (predicted - float(outcome)) ** 2
        calibration["count"] = n
        calibration["brier_score"] = self._ema(prev, brier, n)
        empirical = self.calibrated_confidence(predicted)
        return abs(empirical - outcome)

    @staticmethod
    def _bump_counter(counter: dict[str, Any], key: str, amount: int = 1) -> None:
        normalized = str(key or "").strip()
        if not normalized:
            return
        counter[normalized] = int(counter.get(normalized, 0) or 0) + int(amount)

    @staticmethod
    def _top_keys(counter: dict[str, Any], limit: int = 5) -> list[str]:
        items = [
            (str(key), int(value or 0))
            for key, value in (counter or {}).items()
            if str(key).strip()
        ]
        items.sort(key=lambda pair: (-pair[1], pair[0]))
        return [key for key, _ in items[:limit]]

    @staticmethod
    def _filtered_ticker_counter(counter: dict[str, Any]) -> dict[str, int]:
        filtered: dict[str, int] = {}
        for raw, count in (counter or {}).items():
            ticker = str(raw or "").strip().upper()
            if (
                not re.fullmatch(r"[A-Z]{1,5}", ticker)
                or ticker in FALSE_BEHAVIOR_TICKERS
                or int(count or 0) <= 0
            ):
                continue
            filtered[ticker] = int(count or 0)
        return filtered

    @staticmethod
    def _normalize_tickers(raw: Any) -> list[str]:
        values: list[str]
        if isinstance(raw, str):
            values = re.findall(r"\b[A-Z]{1,5}\b", raw.upper())
        elif isinstance(raw, list):
            values = [str(item).upper().strip() for item in raw]
        else:
            values = []
        seen = set()
        out = []
        for ticker in values:
            if not re.fullmatch(r"[A-Z]{1,5}", ticker):
                continue
            if ticker in FALSE_BEHAVIOR_TICKERS:
                continue
            if ticker in seen:
                continue
            seen.add(ticker)
            out.append(ticker)
        return out[:8]

    def learn_from_behavior(
        self,
        *,
        user_scope: str,
        event_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Learn lightweight user preferences from interaction behavior.

        This intentionally stores aggregate metadata, not raw prompt text.
        """
        metadata = dict(metadata or {})
        event_type = str(event_type or "event").strip().lower()[:80] or "event"
        now = time.time()

        with self._lock:
            behavior = self._behavior_bucket(user_scope)
            behavior["events"] = int(behavior.get("events", 0) or 0) + 1
            self._bump_counter(behavior.setdefault("event_counts", {}), event_type)

            intent = str(metadata.get("intent") or "").strip().lower()
            if intent:
                self._bump_counter(behavior.setdefault("intent_counts", {}), intent)

            page = str(metadata.get("page") or "").strip().lower()
            if page:
                self._bump_counter(behavior.setdefault("page_views", {}), page)

            response_mode = str(metadata.get("response_mode") or "").strip().lower()
            if response_mode:
                self._bump_counter(behavior.setdefault("response_modes", {}), response_mode)

            if bool(metadata.get("is_followup")):
                behavior["followup_count"] = int(behavior.get("followup_count", 0) or 0) + 1

            tickers = self._normalize_tickers(metadata.get("tickers") or metadata.get("ticker"))
            for ticker in tickers:
                self._bump_counter(behavior.setdefault("ticker_interest", {}), ticker)

            ticker_interest = self._filtered_ticker_counter(behavior.get("ticker_interest", {}))
            behavior["preferred_tickers"] = self._top_keys(ticker_interest, 8)
            behavior["preferred_pages"] = self._top_keys(behavior.get("page_views", {}), 5)
            response_modes = behavior.get("response_modes", {})
            preferred_modes = self._top_keys(response_modes, 1)
            if preferred_modes:
                behavior["preferred_response_mode"] = preferred_modes[0]

            total_chat_events = int((behavior.get("event_counts") or {}).get("chat_message", 0) or 0)
            followup_ratio = int(behavior.get("followup_count", 0) or 0) / max(total_chat_events, 1)
            intent_counts = behavior.get("intent_counts", {})
            evidence_heavy = sum(int(intent_counts.get(name, 0) or 0) for name in ["memo", "compare", "rebalance", "execution"])
            lookup_heavy = int(intent_counts.get("lookup", 0) or 0)
            if followup_ratio >= 0.45 or evidence_heavy >= max(2, lookup_heavy):
                behavior["preferred_focus"] = "evidence_depth"
            elif behavior.get("preferred_response_mode") in {"fast", "compact", "lite"} and lookup_heavy > evidence_heavy:
                behavior["preferred_focus"] = "speed"
            else:
                behavior["preferred_focus"] = "balanced"

            event_snapshot = {
                "event_type": event_type,
                "timestamp": now,
                "intent": intent,
                "page": page,
                "tickers": tickers,
                "response_mode": response_mode,
                "is_followup": bool(metadata.get("is_followup")),
            }
            last_events = behavior.setdefault("last_events", [])
            last_events.append(event_snapshot)
            behavior["last_events"] = last_events[-30:]

            self._save()
            return self.behavior_profile(user_scope)

    def behavior_profile(self, user_scope: str) -> dict[str, Any]:
        behavior = self._behavior_bucket(user_scope)
        ticker_interest = self._filtered_ticker_counter(behavior.get("ticker_interest", {}))
        return {
            "user_scope": user_scope,
            "events": int(behavior.get("events", 0) or 0),
            "preferred_tickers": self._top_keys(ticker_interest, 8),
            "preferred_pages": list(behavior.get("preferred_pages") or []),
            "preferred_response_mode": behavior.get("preferred_response_mode", "compact"),
            "preferred_focus": behavior.get("preferred_focus", "balanced"),
            "followup_count": int(behavior.get("followup_count", 0) or 0),
            "event_counts": dict(behavior.get("event_counts") or {}),
            "intent_counts": dict(behavior.get("intent_counts") or {}),
            "ticker_interest": ticker_interest,
            "page_views": dict(behavior.get("page_views") or {}),
            "response_modes": dict(behavior.get("response_modes") or {}),
            "last_events": list(behavior.get("last_events") or []),
        }

    def learn_from_run(self, run_record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            prompt = str(run_record.get("prompt") or "")
            user_id = str(run_record.get("user_id") or "anonymous")
            role = str(run_record.get("user_role") or "analyst")
            latency = int(run_record.get("latency_ms", 0) or 0)
            eval_result = run_record.get("eval_result") or {}
            score = float(eval_result.get("overall_score", 0.0) or 0.0)
            passed = bool(eval_result.get("passed", False))
            observed_tools = [str(t) for t in (run_record.get("observed_tools") or []) if str(t)]

            intent = self.classify_intent(prompt)
            intent_bucket = self._intent_bucket(intent)
            user_bucket = self._user_bucket(user_id)

            intent_bucket["runs"] = int(intent_bucket.get("runs", 0) or 0) + 1
            intent_bucket["passes"] = int(intent_bucket.get("passes", 0) or 0) + (1 if passed else 0)
            intent_bucket["avg_eval_score"] = self._ema(float(intent_bucket.get("avg_eval_score", 0.0)), score, intent_bucket["runs"])
            intent_bucket["avg_latency_ms"] = self._ema(float(intent_bucket.get("avg_latency_ms", 0.0)), latency, intent_bucket["runs"])
            citation_precision = float((eval_result.get("metrics") or {}).get("citation_precision", {}).get("score", 0.0) or 0.0)
            intent_bucket["avg_citation_precision"] = self._ema(
                float(intent_bucket.get("avg_citation_precision", 0.0)),
                citation_precision,
                intent_bucket["runs"],
            )
            if passed:
                intent_bucket["recent_failures"] = 0
            else:
                intent_bucket["recent_failures"] = int(intent_bucket.get("recent_failures", 0) or 0) + 1

            user_bucket["runs"] = int(user_bucket.get("runs", 0) or 0) + 1
            user_bucket["passes"] = int(user_bucket.get("passes", 0) or 0) + (1 if passed else 0)
            if passed:
                user_bucket["recent_failures"] = 0
            else:
                user_bucket["recent_failures"] = int(user_bucket.get("recent_failures", 0) or 0) + 1

            intent_success = user_bucket.setdefault("intent_success", {})
            pair = intent_success.setdefault(intent, {"runs": 0, "passes": 0})
            pair["runs"] += 1
            pair["passes"] += 1 if passed else 0

            user_failures = int(user_bucket.get("recent_failures", 0) or 0)
            if role == "viewer" or user_failures >= 3:
                user_bucket["risk_class"] = "elevated"
            elif user_failures >= 1:
                user_bucket["risk_class"] = "guarded"
            else:
                user_bucket["risk_class"] = "standard"

            for tool in observed_tools:
                bucket = self._tool_bucket(tool)
                bucket["runs"] = int(bucket.get("runs", 0) or 0) + 1
                if passed:
                    bucket["success"] = int(bucket.get("success", 0) or 0) + 1
                    bucket["failure_streak"] = 0
                    bucket["last_error"] = ""
                else:
                    bucket["failure_streak"] = int(bucket.get("failure_streak", 0) or 0) + 1
                    failures = (run_record.get("review") or {}).get("retry_failures") or []
                    if failures:
                        bucket["last_error"] = str(failures[-1])[:500]
                bucket["avg_latency_ms"] = self._ema(
                    float(bucket.get("avg_latency_ms", 0.0) or 0.0),
                    latency,
                    bucket["runs"],
                )

            global_bucket = self.state.setdefault("global", {})
            global_bucket["runs"] = int(global_bucket.get("runs", 0) or 0) + 1
            global_bucket["avg_eval_score"] = self._ema(float(global_bucket.get("avg_eval_score", 0.0) or 0.0), score, global_bucket["runs"])
            global_bucket["avg_latency_ms"] = self._ema(float(global_bucket.get("avg_latency_ms", 0.0) or 0.0), latency, global_bucket["runs"])
            global_bucket["pass_rate"] = self._ema(float(global_bucket.get("pass_rate", 0.0) or 0.0), 1.0 if passed else 0.0, global_bucket["runs"])

            raw_confidence = max(0.0, min(1.0, score))
            confidence_error = self._update_confidence_calibration(raw_confidence, 1 if passed else 0)
            intent_bucket["avg_confidence_error"] = self._ema(
                float(intent_bucket.get("avg_confidence_error", 0.0) or 0.0),
                confidence_error,
                intent_bucket["runs"],
            )

            self._save()
            return {
                "intent": intent,
                "user_risk_class": user_bucket.get("risk_class"),
                "pass": passed,
                "score": round(score, 4),
                "calibrated_confidence": round(self.calibrated_confidence(raw_confidence), 4),
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self.state)
            try:
                snapshot["policy_file"] = str(self.policy_file.relative_to(BASE_DIR))
            except ValueError:
                snapshot["policy_file"] = str(self.policy_file)
            return snapshot


class AutonomousMonitorManager:
    """Long-horizon autonomous monitor jobs with persisted state."""

    def __init__(self, state_file: Path = AUTONOMY_DIR / "monitor_jobs.json") -> None:
        self.state_file = state_file
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._load()

    def _ensure_dir(self) -> None:
        AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._jobs = dict(payload.get("jobs") or {})
        except Exception:
            self._jobs = {}

    def _save(self) -> None:
        self._ensure_dir()
        payload = {"jobs": self._jobs, "updated_at": time.time()}
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_file)

    def _append_event(self, job_id: str, event: dict[str, Any]) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        events = job.setdefault("events", [])
        events.append(event)
        job["last_event_at"] = time.time()
        job["events"] = events[-200:]

    def _worker(self, *, job_id: str, signal_provider: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            stop_event = self._stop_events.setdefault(job_id, threading.Event())

        started_at = float(job.get("started_at", time.time()))
        duration_s = int(job.get("duration_s", 1800) or 1800)
        poll_interval_s = int(job.get("poll_interval_s", 60) or 60)
        baseline_price: float | None = None
        baseline_direction = ""

        while not stop_event.is_set():
            now = time.time()
            if now - started_at >= duration_s:
                break

            observation: dict[str, Any]
            try:
                observation = signal_provider(job)
            except Exception as exc:
                observation = {"error": str(exc)}

            direction = str(observation.get("direction") or "")
            price = observation.get("price")
            decisive = False
            reasons: list[str] = []

            if isinstance(price, (int, float)):
                if baseline_price is None:
                    baseline_price = float(price)
                elif baseline_price > 0:
                    change_pct = ((float(price) - baseline_price) / baseline_price) * 100.0
                    if abs(change_pct) >= 2.0:
                        decisive = True
                        reasons.append(f"price_move_{change_pct:+.2f}pct")
            if direction:
                if not baseline_direction:
                    baseline_direction = direction
                elif direction != baseline_direction:
                    decisive = True
                    reasons.append(f"direction_shift_{baseline_direction}_to_{direction}")
                    baseline_direction = direction

            confidence = observation.get("confidence")
            if isinstance(confidence, (int, float)) and float(confidence) >= 80.0:
                decisive = True
                reasons.append("high_confidence_signal")

            event = {
                "timestamp": now,
                "observation": observation,
                "decisive": decisive,
                "reasons": reasons,
            }
            with self._lock:
                self._append_event(job_id, event)
                job = self._jobs.get(job_id, {})
                job["status"] = "running"
                if decisive:
                    job["decisive_signals"] = int(job.get("decisive_signals", 0) or 0) + 1
                    job["last_decisive_signal"] = event
                self._save()

            stop_event.wait(timeout=max(10, poll_interval_s))

        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                if stop_event.is_set():
                    job["status"] = "stopped"
                else:
                    job["status"] = "completed"
                job["finished_at"] = time.time()
                self._save()

    def start_job(
        self,
        *,
        user_scope: str,
        ticker: str,
        intent: str,
        duration_s: int,
        poll_interval_s: int,
        objective: dict[str, Any],
        signal_provider: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        duration_s = max(60, min(int(duration_s), 3600))
        poll_interval_s = max(10, min(int(poll_interval_s), 600))
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        job = {
            "job_id": job_id,
            "user_scope": user_scope,
            "ticker": ticker,
            "intent": intent,
            "objective": objective,
            "duration_s": duration_s,
            "poll_interval_s": poll_interval_s,
            "status": "running",
            "created_at": time.time(),
            "started_at": time.time(),
            "decisive_signals": 0,
            "events": [],
        }
        with self._lock:
            self._jobs[job_id] = job
            self._stop_events[job_id] = threading.Event()
            self._save()
        thread = threading.Thread(
            target=self._worker,
            kwargs={"job_id": job_id, "signal_provider": signal_provider},
            daemon=True,
            name=f"monitor-{job_id}",
        )
        thread.start()
        return job

    def list_jobs(self, *, user_scope: str | None = None, active_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        if user_scope:
            jobs = [job for job in jobs if str(job.get("user_scope")) == str(user_scope)]
        if active_only:
            jobs = [job for job in jobs if job.get("status") == "running"]
        jobs.sort(key=lambda item: float(item.get("created_at", 0.0)), reverse=True)
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def stop_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            event = self._stop_events.get(job_id)
            if event:
                event.set()
            job = self._jobs.get(job_id)
            if not job:
                return None
            job["status"] = "stopping"
            self._save()
            return dict(job)
