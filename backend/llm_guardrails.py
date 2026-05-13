"""LLM guardrails for RAPHI chat and memo responses.

These checks are intentionally deterministic and local. They do not replace
model-side safety, but they give the app enforceable post-generation behavior:
investment responses must include risk/uncertainty framing, memo responses must
keep the expected section schema, unsupported ticker references are flagged, and
overconfident language is softened before the text reaches the browser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

REQUIRED_MEMO_SECTIONS = [
    "Recommendation",
    "Key Evidence",
    "GNN / Peer Influence",
    "Risks",
    "Trade Plan",
]

GUARANTEE_PATTERNS = [
    (re.compile(r"\bwill\s+definitely\b", re.IGNORECASE), "may"),
    (re.compile(r"\bguaranteed?\b", re.IGNORECASE), "uncertain"),
    (re.compile(r"\brisk[-\s]?free\b", re.IGNORECASE), "lower-risk"),
    (re.compile(r"\bcannot\s+lose\b", re.IGNORECASE), "can still lose money"),
    (re.compile(r"\bsure\s+thing\b", re.IGNORECASE), "high-conviction but uncertain setup"),
]

COMMON_ACRONYMS = {
    "AI", "API", "AUM", "CAPEX", "CEO", "CFO", "CIK", "CPI", "DCF", "EBIT",
    "EBITDA", "EDGAR", "EPS", "ETF", "FCF", "FOMC", "FX", "FY", "GAAP", "GDP",
    "GNN", "GPU", "HTTP", "IPO", "JSON",
    "LLM", "MCP", "ML", "NASDAQ", "NYSE", "PCE", "PE", "RAG", "ROIC", "SEC",
    "SSE", "TTM", "UI", "USD", "VAR", "VIX", "XBRL",
    "BUY", "SELL", "HOLD", "LONG", "SHORT",
}


@dataclass
class GuardrailContext:
    ticker: str = ""
    allowed_tickers: set[str] = field(default_factory=set)
    source_summary: str = ""
    require_memo_schema: bool = False


@dataclass
class GuardrailReport:
    valid: bool
    repairs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    unknown_tickers: list[str] = field(default_factory=list)


def detect_investment_content(text: str) -> bool:
    return bool(re.search(
        r"\b(buy|sell|hold|long|short|target|stop[- ]loss|allocation|position|portfolio|risk|conviction|upside|downside)\b",
        text,
        re.IGNORECASE,
    ))


def soften_overconfidence(text: str) -> tuple[str, list[str]]:
    repairs = []
    updated = text
    for pattern, replacement in GUARANTEE_PATTERNS:
        if pattern.search(updated):
            updated = pattern.sub(replacement, updated)
            repairs.append(f"softened overconfident phrase: {pattern.pattern}")
    return updated, repairs


def find_missing_memo_sections(text: str) -> list[str]:
    missing = []
    for section in REQUIRED_MEMO_SECTIONS:
        pattern = re.compile(rf"(^|\n)\s*(#+\s*)?{re.escape(section)}\b", re.IGNORECASE)
        if not pattern.search(text):
            missing.append(section)
    return missing


def find_unknown_tickers(text: str, allowed_tickers: Iterable[str]) -> list[str]:
    allowed = {t.upper() for t in allowed_tickers if t}
    tokens = set(re.findall(r"\b[A-Z]{2,5}\b", text))
    unknown = sorted(t for t in tokens if t not in allowed and t not in COMMON_ACRONYMS)
    return unknown[:12]


def has_risk_framing(text: str) -> bool:
    return bool(re.search(r"\b(risk|downside|uncertain|uncertainty|invalidation|stop[- ]loss|scenario|may|could)\b", text, re.IGNORECASE))


def append_guardrail_section(text: str, report: GuardrailReport, context: GuardrailContext) -> str:
    lines = []
    if not has_risk_framing(text):
        lines.append("- This view is uncertain and should be sized with explicit downside limits.")
        report.repairs.append("added risk and uncertainty framing")
    if report.unknown_tickers:
        lines.append("- Unverified ticker references: " + ", ".join(report.unknown_tickers) + ". Confirm before relying on them.")
        report.warnings.append("unknown ticker references detected")
    if context.source_summary:
        lines.append(f"- Data provenance checked: {context.source_summary}")
    if not lines:
        return text
    return text.rstrip() + "\n\n### Guardrail Notes\n" + "\n".join(lines)


def repair_memo_schema(text: str, report: GuardrailReport) -> str:
    if not report.missing_sections:
        return text
    additions = []
    for section in report.missing_sections:
        if section == "Risks":
            body = "- No investment view is complete without downside, sizing, and invalidation criteria."
        elif section == "Trade Plan":
            body = "- Define entry, target, stop-loss, sizing, and review horizon before acting."
        elif section == "GNN / Peer Influence":
            body = "- Peer and graph-neighbor influence was not explicitly cited; treat this as unverified until checked."
        else:
            body = "- Section missing from model output; rerun memo generation for a fully sourced view."
        additions.append(f"### {section}\n{body}")
    report.repairs.append("added missing memo schema sections")
    return text.rstrip() + "\n\n" + "\n\n".join(additions)


def validate_and_repair_response(
    text: str,
    context: Optional[GuardrailContext] = None,
) -> tuple[str, GuardrailReport]:
    context = context or GuardrailContext()
    report = GuardrailReport(valid=True)
    repaired, overconfidence_repairs = soften_overconfidence(text or "")
    report.repairs.extend(overconfidence_repairs)

    allowed = set(context.allowed_tickers)
    if context.ticker:
        allowed.add(context.ticker.upper())
    report.unknown_tickers = find_unknown_tickers(repaired, allowed)

    if context.require_memo_schema:
        report.missing_sections = find_missing_memo_sections(repaired)
        repaired = repair_memo_schema(repaired, report)

    if detect_investment_content(repaired):
        repaired = append_guardrail_section(repaired, report, context)

    report.valid = not report.missing_sections and not report.unknown_tickers and not overconfidence_repairs
    return repaired, report
