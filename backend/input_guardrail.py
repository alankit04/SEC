"""Control Plane 1 — pre-loop input guardrail for the agentic route.

A deterministic, no-LLM domain classifier that routes a query into one of two
buckets BEFORE the agentic loop runs:

  - "finance"  → the query is in RAPHI's domain; let it enter the agentic loop.
  - "general"  → greeting, identity, or off-topic; it must be handled before the
                 loop so it never produces a Perceive→Plan→Execute state dump.

This sits immediately after ``sanitize_user_input()`` (which guards prompt
injection and length) and is purely about *domain*, not safety. No LLM calls,
no new dependencies. Ticker detection is intentionally tighter than the
planner's ``extract_tickers`` — that extractor treats any capitalized word as a
candidate because the loop validates downstream, but this gate has no validation
step, so a capitalized greeting must not read as a ticker. It does reuse the
planner's ``FALSE_TICKER_WORDS`` constant as a single source of truth for the
stop-word list.
"""

from __future__ import annotations

import re
from typing import Literal

from raphi.orchestrators.planner import FALSE_TICKER_WORDS

Bucket = Literal["finance", "general"]

# Finance signal vocabulary, grouped to mirror the spec. Matched as whole words
# (case-insensitive) so substrings like "various" do not trip the "var" signal.
_FINANCE_TERMS = [
    # SEC / filing terms
    "sec", "filing", "filings", "10-k", "10-q", "8-k", "form 4",
    # market / price / stock
    "market", "price", "stock", "stocks",
    # portfolio / risk
    "portfolio", "var", "sharpe",
    # recommendation phrasing
    "buy", "sell", "hold", "should i",
    # model / signal
    "gnn", "signal", "xgboost",
]

# Word-boundary alternation. The negative look-arounds also reject a leading or
# trailing hyphen so the form tokens ("10-k") match cleanly without "var"
# tripping on "various".
_FINANCE_RE = re.compile(
    r"(?<![\w-])(?:" + "|".join(re.escape(t) for t in _FINANCE_TERMS) + r")(?![\w-])",
    re.IGNORECASE,
)

# Explicit ticker syntax is an unambiguous finance signal: $PLTR, ticker:pltr,
# NASDAQ: PLTR. Unlike the planner's extractor, the guardrail does NOT treat
# every capitalized word as a ticker — there is no downstream validation here to
# reject false hits, so a capitalized greeting must not read as finance.
_EXPLICIT_TICKER_RE = re.compile(
    r"\$[A-Za-z]{1,5}(?:\.[A-Za-z])?\b"
    r"|(?:ticker|NASDAQ|NYSE|AMEX|OTC)\s*:\s*[A-Za-z]{1,5}(?:\.[A-Za-z])?\b",
    re.IGNORECASE,
)

# A genuinely all-uppercase token (e.g. "NVDA" in an otherwise mixed-case query)
# is a likely ticker. Capitalized words like "Hello" do not match.
_BARE_UPPER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b")


def _has_ticker_signal(text: str) -> bool:
    if _EXPLICIT_TICKER_RE.search(text):
        return True

    candidates = [
        t for t in _BARE_UPPER_RE.findall(text)
        if t.replace(".", "") not in FALSE_TICKER_WORDS
    ]
    if not candidates:
        return False

    # Shouting guard: if the whole query is uppercase and there are several
    # candidate tokens, it is emphasis ("HELLO THERE"), not a ticker. A lone
    # all-caps token ("NVDA") still counts.
    letters = [ch for ch in text if ch.isalpha()]
    if letters and "".join(letters).isupper() and len(candidates) > 1:
        return False

    return True


def classify_input_bucket(query: str) -> Bucket:
    """Classify a query into "finance" or "general".

    Returns "finance" if the query contains any finance signal term or a ticker
    reference, otherwise "general". Empty/whitespace queries are "general".
    """
    text = (query or "").strip()
    if not text:
        return "general"

    if _FINANCE_RE.search(text):
        return "finance"

    if _has_ticker_signal(text):
        return "finance"

    return "general"
