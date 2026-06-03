"""Control Plane 3 — retrieval gate for the agentic loop.

Data returned from external tools (Firecrawl excerpts, EDGAR/SEC markdown, news
titles) is written into ``state.retrieval_results`` and then flows into the
model's context. An adversarial prompt-injection payload buried in any of those
string fields could hijack the agent. ``screen_retrieval_result`` scans tool
output for the SAME injection patterns the input gate uses and redacts any
offending string field before the data is trusted.

Deterministic, no LLM calls, no new dependencies. The injection patterns are
reused from ``backend.security`` (``_INJECTION_RE`` is the pre-compiled form of
``_INJECTION_PATTERNS``) so there is a single source of truth — they are not
duplicated here.
"""

from __future__ import annotations

from backend.security import _INJECTION_RE

REDACTION_NOTICE = "[redacted: injection pattern detected in retrieved content]"


def _contains_injection(text: str) -> bool:
    return any(rx.search(text) for rx in _INJECTION_RE)


def _screen_value(value):
    """Recursively screen a value. String leaves matching an injection pattern
    are replaced with the redaction notice; dicts and lists are screened
    element-wise; all other types (numbers, bools, None) pass through unchanged.

    Recursion is required because real tool payloads nest the threat — e.g.
    stock_news returns ``{"items": [{"title": ...}]}`` and the news title is an
    explicit injection vector.
    """
    if isinstance(value, str):
        return REDACTION_NOTICE if _contains_injection(value) else value
    if isinstance(value, dict):
        return {k: _screen_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_screen_value(item) for item in value]
    return value


def screen_retrieval_result(tool_name: str, result: dict | list) -> dict | list:
    """Screen a tool result for prompt-injection content before it is stored.

    ``tool_name`` is accepted for future tool-specific policy; screening today is
    uniform across tools. Returns a screened copy of the result (lists screened
    item-by-item, dict string fields redacted where an injection pattern matches).
    Non-dict/non-list results are returned unchanged.
    """
    if isinstance(result, (dict, list)):
        return _screen_value(result)
    return result
