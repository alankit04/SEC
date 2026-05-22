from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CircuitBreakerState:
    failures: int = 0
    opened_at: float | None = None
    last_error: str = ""


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout_s: int = 60) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.state = CircuitBreakerState()

    def is_open(self) -> bool:
        if self.state.opened_at is None:
            return False
        if (time.time() - self.state.opened_at) >= self.reset_timeout_s:
            self.state.opened_at = None
            self.state.failures = 0
            self.state.last_error = ""
            return False
        return True

    def allow(self) -> bool:
        return not self.is_open()

    def record_success(self) -> None:
        self.state.failures = 0
        self.state.opened_at = None
        self.state.last_error = ""

    def record_failure(self, error: str) -> None:
        self.state.failures += 1
        self.state.last_error = str(error)[:500]
        if self.state.failures >= self.failure_threshold:
            self.state.opened_at = time.time()

    def status(self) -> dict[str, Any]:
        return {
            "open": self.is_open(),
            "failures": self.state.failures,
            "failure_threshold": self.failure_threshold,
            "reset_timeout_s": self.reset_timeout_s,
            "last_error": self.state.last_error,
            "opened_at": self.state.opened_at,
        }


class ProviderHealthRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, dict[str, Any]] = {}

    def set_provider(self, name: str, *, configured: bool, breaker: CircuitBreaker | None = None, meta: dict[str, Any] | None = None) -> None:
        self._providers[name] = {
            "configured": bool(configured),
            "breaker": breaker,
            "meta": dict(meta or {}),
        }

    def status(self) -> dict[str, Any]:
        providers: dict[str, Any] = {}
        for name, info in self._providers.items():
            breaker = info.get("breaker")
            providers[name] = {
                "configured": info.get("configured", False),
                "circuit_breaker": breaker.status() if breaker else None,
                "meta": info.get("meta", {}),
            }
        all_ok = all(
            p["configured"] and (not p["circuit_breaker"] or not p["circuit_breaker"]["open"])
            for p in providers.values()
        ) if providers else True
        return {
            "status": "ok" if all_ok else "degraded",
            "providers": providers,
            "timestamp": time.time(),
        }
