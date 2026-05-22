import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from provider_controls import CircuitBreaker, ProviderHealthRegistry  # noqa: E402


def test_circuit_breaker_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=3600)
    assert cb.allow() is True
    cb.record_failure("x")
    assert cb.allow() is True
    cb.record_failure("y")
    assert cb.allow() is False


def test_provider_registry_degraded_when_breaker_open():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=3600)
    cb.record_failure("boom")

    registry = ProviderHealthRegistry()
    registry.set_provider("p1", configured=True, breaker=cb)
    status = registry.status()

    assert status["status"] == "degraded"
    assert status["providers"]["p1"]["circuit_breaker"]["open"] is True
