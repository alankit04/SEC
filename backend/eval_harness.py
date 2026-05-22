"""Formal eval harness for RAPHI agent responses.

The harness evaluates completed agent runs. It does not call the LLM itself;
instead it scores the observable output of a run: prompt, final answer, tool
trace, retrieved citations, and guardrail report. This makes evals repeatable
and usable in CI or during local regression testing.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:  # pragma: no cover - package/import style differs between tests and app
    from llm_guardrails import (
        REQUIRED_MEMO_SECTIONS,
        GuardrailContext,
        GuardrailReport,
        find_missing_memo_sections,
        validate_and_repair_response,
    )
except ImportError:  # pragma: no cover
    from backend.llm_guardrails import (
        REQUIRED_MEMO_SECTIONS,
        GuardrailContext,
        GuardrailReport,
        find_missing_memo_sections,
        validate_and_repair_response,
    )


SEC_URL_RE = re.compile(r"https://www\.sec\.gov/Archives/[^\s)\]]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
ACCESSION_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
CITATION_MARKER_RE = re.compile(r"(?:\[\d+\]|https?://|accession\s+\d{10}-\d{2}-\d{6})", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

FACTUAL_CUE_RE = re.compile(
    r"\b("
    r"revenue|income|cash|debt|margin|eps|filing|filed|10-k|10-q|8-k|"
    r"market cap|price|shares|volume|liquidity|sec|xbrl|edgar|"
    r"signal|confidence|gnn|portfolio|var|sharpe|exposure|risk|"
    r"news|announced|reported|guidance|quarter|year|accession"
    r")\b",
    re.IGNORECASE,
)

LOW_EVIDENCE_RE = re.compile(
    r"\b("
    r"may|could|might|uncertain|risk|risks|scenario|not guaranteed|"
    r"requires verification|needs review|insufficient evidence|unknown"
    r")\b",
    re.IGNORECASE,
)

MODEL_SIGNAL_CUE_RE = re.compile(r"\b(model|signal|xgboost|lstm|shap|gnn|graph|peer|neighbor)\b", re.IGNORECASE)
ML_BULLISH_RE = re.compile(r"\b(ml|model|signal).{0,60}\b(bullish|buy|positive|long)\b", re.IGNORECASE)
GNN_BEARISH_RE = re.compile(r"\b(gnn|graph|peer|neighbor).{0,60}\b(bearish|sell|negative|short)\b", re.IGNORECASE)

TOOL_KEYWORDS = {
    "market": {"market", "price", "performance", "liquidity", "volume", "news", "technical"},
    "sec": {"sec", "edgar", "filing", "10-k", "10-q", "8-k", "xbrl", "accession", "fundamental"},
    "citation": {"citation", "cite", "source", "evidence", "link", "perplexity", "web"},
    "ml": {"ml", "model", "signal", "confidence", "prediction"},
    "gnn": {"gnn", "graph", "peer", "neighbor", "relationship"},
    "portfolio": {"portfolio", "position", "exposure", "var", "sharpe", "p&l", "allocation"},
    "memory": {"remember", "previous", "earlier", "last time", "thesis", "context"},
}


def evaluate_model_signal_provenance(answer: str, context: dict) -> bool:
    """Check if answer cites model version, date, and provenance."""
    return bool(re.search(r"model.*trained|signal.*model|provenance|version", answer, re.I))


def evaluate_gnn_coverage_honesty(answer: str, context: dict) -> bool:
    """Pass if answer admits when GNN is unavailable or out-of-date."""
    return bool(re.search(r"gnn (unavailable|not available|out[- ]of[- ]date|updated|not trained|not in graph)", answer, re.I))


def evaluate_signal_uncertainty_framing(answer: str, context: dict) -> bool:
    """Pass if answer frames signals as probabilistic, not certain."""
    return bool(re.search(r"uncertainty|probabilistic|confidence|likelihood|not certain|estimate|may|could|might", answer, re.I))


def evaluate_conflict_reasoning(answer: str, context: dict) -> bool:
    """Pass if answer lowers conviction when ML and GNN disagree."""
    return bool(re.search(r"ml.*bullish.*gnn.*bearish|gnn.*bearish.*ml.*bullish|conflict|lowered conviction|disagree|contradict", answer, re.I))


def evaluate_local_optimization_honesty(answer: str, context: dict) -> bool:
    """Pass if answer describes RL/distillation as local, not LLM tuning."""
    return bool(re.search(r"local (rl|distillation|optimization).*not (llm|foundation model|fine[- ]tuning)", answer, re.I))


def _metric_from_bool(name: str, required: bool, passed: bool, *, details: dict[str, Any] | None = None) -> MetricResult:
    if not required:
        return MetricResult(name=name, score=1.0, passed=True, details={"required": False, **(details or {})})
    return MetricResult(name=name, score=1.0 if passed else 0.0, passed=passed, details={"required": True, **(details or {})})


def _requires_ml_context(case: EvalCase) -> bool:
    if not bool(case.metadata.get("enforce_ml_metrics", False)):
        return False
    text = f"{case.prompt}\n{case.response}"
    expected = {_normalize_tool_name(tool) for tool in (case.expected_tools or [])}
    if "ml" in expected or "gnn" in expected:
        return True
    return bool(MODEL_SIGNAL_CUE_RE.search(text))


def evaluate_model_signal_provenance_metric(case: EvalCase) -> MetricResult:
    passed = evaluate_model_signal_provenance(case.response, case.metadata)
    return _metric_from_bool(
        "model_signal_provenance",
        _requires_ml_context(case),
        passed,
        details={"check": "model/version/provenance disclosure"},
    )


def evaluate_gnn_coverage_honesty_metric(case: EvalCase) -> MetricResult:
    required = _requires_ml_context(case)
    passed = evaluate_gnn_coverage_honesty(case.response, case.metadata)
    return _metric_from_bool(
        "gnn_coverage_honesty",
        required,
        passed,
        details={"check": "gnn availability or staleness disclosure"},
    )


def evaluate_signal_uncertainty_framing_metric(case: EvalCase) -> MetricResult:
    passed = evaluate_signal_uncertainty_framing(case.response, case.metadata)
    return _metric_from_bool(
        "signal_uncertainty_framing",
        _requires_ml_context(case),
        passed,
        details={"check": "probabilistic framing"},
    )


def evaluate_conflict_reasoning_metric(case: EvalCase) -> MetricResult:
    required = _requires_ml_context(case) and (bool(ML_BULLISH_RE.search(case.response)) or bool(GNN_BEARISH_RE.search(case.response)))
    passed = evaluate_conflict_reasoning(case.response, case.metadata)
    return _metric_from_bool(
        "conflict_reasoning",
        required,
        passed,
        details={"check": "ML/GNN disagreement handling"},
    )


def evaluate_local_optimization_honesty_metric(case: EvalCase) -> MetricResult:
    required = bool(re.search(r"\b(rl|distillation|quantization|optimization)\b", case.prompt + "\n" + case.response, re.IGNORECASE))
    passed = evaluate_local_optimization_honesty(case.response, case.metadata)
    return _metric_from_bool(
        "local_optimization_honesty",
        required,
        passed,
        details={"check": "local optimization != hosted LLM tuning"},
    )

TOOL_ALIASES = {
    "market": {
        "market",
        "market_overview",
        "stock_detail",
        "stock_news",
        "@market-analyst",
        "mcp__raphi__market_overview",
        "mcp__raphi__stock_detail",
        "mcp__raphi__stock_news",
    },
    "sec": {
        "sec",
        "sec_filings",
        "sec_search",
        "edgar_live_filings",
        "edgar_search_fulltext",
        "@sec-researcher",
        "mcp__raphi__sec_filings",
        "mcp__raphi__sec_search",
        "mcp__raphi__edgar_live_filings",
        "mcp__raphi__edgar_search_fulltext",
    },
    "citation": {
        "citation",
        "web_citations",
        "firecrawl_search",
        "firecrawl_scrape",
        "@web-citation-search",
        "mcp__raphi__web_citations",
        "mcp__raphi__firecrawl_search",
        "mcp__raphi__firecrawl_scrape",
    },
    "ml": {"ml", "ml_signal", "@ml-signals", "mcp__raphi__ml_signal"},
    "gnn": {"gnn", "gnn_signal", "gnn_status", "gnn_train", "@gnn-influence", "mcp__raphi__gnn_signal", "mcp__raphi__gnn_status", "mcp__raphi__gnn_train"},
    "portfolio": {"portfolio", "portfolio_snapshot", "portfolio_alerts", "@portfolio-risk", "mcp__raphi__portfolio_snapshot", "mcp__raphi__portfolio_alerts"},
    "memory": {"memory", "memory_status", "memory_retrieve", "mcp__raphi__memory_status", "mcp__raphi__memory_retrieve"},
}


@dataclass
class ToolCallRecord:
    """Observed tool activity from an agent run."""

    name: str
    ok: bool = True
    latency_ms: float | None = None
    input: dict[str, Any] = field(default_factory=dict)
    output_summary: str = ""
    error: str = ""

    @classmethod
    def from_any(cls, value: Any) -> "ToolCallRecord":
        if isinstance(value, ToolCallRecord):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if isinstance(value, dict):
            return cls(
                name=str(value.get("name") or value.get("tool") or value.get("id") or ""),
                ok=bool(value.get("ok", value.get("success", not value.get("error")))),
                latency_ms=value.get("latency_ms"),
                input=dict(value.get("input") or value.get("arguments") or {}),
                output_summary=str(value.get("output_summary") or value.get("summary") or ""),
                error=str(value.get("error") or ""),
            )
        return cls(name=str(value))


@dataclass
class CitationRecord:
    """Citation evidence retrieved or used by a response."""

    url: str
    title: str = ""
    source_type: str = "web"
    accession: str = ""
    used: bool = True

    @classmethod
    def from_any(cls, value: Any) -> "CitationRecord":
        if isinstance(value, CitationRecord):
            return value
        if isinstance(value, str):
            accession = ACCESSION_RE.search(value)
            return cls(url=value, accession=accession.group(0) if accession else "")
        if isinstance(value, dict):
            url = str(value.get("url") or value.get("sec_url") or "")
            accession = str(value.get("accession") or "")
            if not accession:
                match = ACCESSION_RE.search(json.dumps(value, sort_keys=True))
                accession = match.group(0) if match else ""
            source_type = str(value.get("source_type") or value.get("provider") or value.get("type") or "web")
            return cls(
                url=url,
                title=str(value.get("title") or value.get("source") or ""),
                source_type=source_type,
                accession=accession,
                used=bool(value.get("used", True)),
            )
        return cls(url=str(value))


@dataclass
class EvalCase:
    """Golden-case contract for one completed RAPHI run."""

    id: str
    prompt: str
    response: str
    expected_tools: list[str] = field(default_factory=list)
    observed_tools: list[ToolCallRecord] = field(default_factory=list)
    citations: list[CitationRecord] = field(default_factory=list)
    allowed_tickers: set[str] = field(default_factory=set)
    ticker: str = ""
    require_memo_schema: bool = False
    require_citations: bool = True
    min_overall_score: float = 0.75
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(value["id"]),
            prompt=str(value.get("prompt") or ""),
            response=str(value.get("response") or ""),
            expected_tools=[str(item) for item in value.get("expected_tools", [])],
            observed_tools=[ToolCallRecord.from_any(item) for item in value.get("observed_tools", [])],
            citations=[CitationRecord.from_any(item) for item in value.get("citations", [])],
            allowed_tickers={str(item).upper() for item in value.get("allowed_tickers", [])},
            ticker=str(value.get("ticker") or ""),
            require_memo_schema=bool(value.get("require_memo_schema", False)),
            require_citations=bool(value.get("require_citations", True)),
            min_overall_score=float(value.get("min_overall_score", 0.75)),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class MetricResult:
    name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    case_id: str
    overall_score: float
    passed: bool
    metrics: dict[str, MetricResult]
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "failures": self.failures,
            "metrics": {
                name: {
                    "score": round(metric.score, 4),
                    "passed": metric.passed,
                    "details": metric.details,
                }
                for name, metric in self.metrics.items()
            },
        }


@dataclass
class EvalSuiteResult:
    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_score: float
    metric_averages: dict[str, float]
    results: list[EvalResult]

    @property
    def passed(self) -> bool:
        return self.failed_cases == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "metric_averages": {k: round(v, 4) for k, v in self.metric_averages.items()},
            "results": [result.to_dict() for result in self.results],
        }


def _normalize_tool_name(name: str) -> str:
    raw = str(name or "").strip()
    lower = raw.lower()
    if lower.startswith("mcp__raphi__"):
        lower = lower.replace("mcp__raphi__", "", 1)
    for family, aliases in TOOL_ALIASES.items():
        if raw in aliases or lower in {alias.lower() for alias in aliases}:
            return family
    for family in TOOL_ALIASES:
        if family in lower:
            return family
    return lower


def infer_expected_tools(prompt: str) -> list[str]:
    """Infer expected tool families from the user goal.

    This mirrors the product's routing expectations but stays deterministic for
    evals. Cases can override it with explicit expected_tools.
    """

    text = str(prompt or "").lower()
    expected = {"market"}
    for family, keywords in TOOL_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            expected.add(family)
    if any(term in text for term in ("analyze", "analysis", "investable", "memo", "conviction")):
        expected.update({"sec", "ml", "gnn", "portfolio", "citation"})
    return sorted(expected)


def extract_citations(text: str) -> list[CitationRecord]:
    citations: list[CitationRecord] = []
    seen: set[tuple[str, str]] = set()
    for url in URL_RE.findall(text or ""):
        source_type = "sec" if SEC_URL_RE.match(url) else "web"
        accession = ""
        match = ACCESSION_RE.search(url)
        if match:
            accession = match.group(0)
        key = (url, accession)
        if key not in seen:
            citations.append(CitationRecord(url=url, source_type=source_type, accession=accession))
            seen.add(key)
    for accession in ACCESSION_RE.findall(text or ""):
        key = ("", accession)
        if key not in seen:
            citations.append(CitationRecord(url="", source_type="sec", accession=accession))
            seen.add(key)
    return citations


def _valid_citation(citation: CitationRecord) -> bool:
    if citation.accession and ACCESSION_RE.fullmatch(citation.accession):
        return True
    if citation.url and URL_RE.fullmatch(citation.url):
        return True
    return False


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    pieces = SENTENCE_SPLIT_RE.split(cleaned)
    return [piece.strip() for piece in pieces if piece.strip()]


def _is_factual_claim(sentence: str) -> bool:
    text = sentence.strip()
    if len(text) < 35:
        return False
    if text.startswith("#") or text.startswith("- #"):
        return False
    has_number = bool(re.search(r"[$%]|\b\d+(?:\.\d+)?\b", text))
    has_cue = bool(FACTUAL_CUE_RE.search(text))
    if not has_number and not has_cue:
        return False
    if LOW_EVIDENCE_RE.search(text) and not has_number and not has_cue:
        return False
    return True


def _has_inline_citation(sentence: str, known_citations: Iterable[CitationRecord]) -> bool:
    if CITATION_MARKER_RE.search(sentence):
        return True
    for citation in known_citations:
        if citation.url and citation.url in sentence:
            return True
        if citation.accession and citation.accession in sentence:
            return True
    return False


def evaluate_citation_precision(case: EvalCase) -> MetricResult:
    response_citations = extract_citations(case.response)
    all_citations = response_citations + case.citations
    unique: dict[tuple[str, str], CitationRecord] = {}
    for citation in all_citations:
        unique[(citation.url, citation.accession)] = citation
    citations = list(unique.values())
    used = [citation for citation in citations if citation.used]
    valid = [citation for citation in used if _valid_citation(citation)]

    if not case.require_citations:
        score = 1.0
    elif not used:
        score = 0.0
    else:
        score = len(valid) / len(used)

    has_sec_when_required = True
    if case.require_citations and any(tool in {"sec", "edgar_live_filings", "sec_filings"} for tool in case.expected_tools):
        has_sec_when_required = any(c.source_type == "sec" or "sec.gov" in c.url for c in used)
        if not has_sec_when_required:
            score = min(score, 0.6)

    return MetricResult(
        name="citation_precision",
        score=score,
        passed=score >= 0.8 and has_sec_when_required,
        details={
            "citation_count": len(used),
            "valid_citation_count": len(valid),
            "response_citation_count": len(response_citations),
            "has_sec_citation_when_required": has_sec_when_required,
        },
    )


def evaluate_unsupported_claim_rate(case: EvalCase) -> MetricResult:
    claims = [sentence for sentence in _sentences(case.response) if _is_factual_claim(sentence)]
    unsupported = [
        claim for claim in claims
        if case.require_citations and not _has_inline_citation(claim, case.citations)
    ]
    if not claims:
        score = 1.0
        rate = 0.0
    else:
        rate = len(unsupported) / len(claims)
        score = max(0.0, 1.0 - rate)
    return MetricResult(
        name="unsupported_claim_rate",
        score=score,
        passed=rate <= 0.25,
        details={
            "factual_claim_count": len(claims),
            "unsupported_claim_count": len(unsupported),
            "unsupported_claim_rate": round(rate, 4),
            "unsupported_claims": unsupported[:8],
        },
    )


def evaluate_memo_schema(case: EvalCase) -> MetricResult:
    if not case.require_memo_schema:
        return MetricResult(
            name="memo_schema_compliance",
            score=1.0,
            passed=True,
            details={"required": False},
        )
    missing = find_missing_memo_sections(case.response)
    score = (len(REQUIRED_MEMO_SECTIONS) - len(missing)) / len(REQUIRED_MEMO_SECTIONS)
    return MetricResult(
        name="memo_schema_compliance",
        score=score,
        passed=not missing,
        details={"required": True, "missing_sections": missing},
    )


def evaluate_tool_routing(case: EvalCase) -> MetricResult:
    expected = case.expected_tools or infer_expected_tools(case.prompt)
    expected_norm = {_normalize_tool_name(tool) for tool in expected}
    observed_records = [ToolCallRecord.from_any(tool) for tool in case.observed_tools]
    observed_norm = {
        _normalize_tool_name(tool.name)
        for tool in observed_records
        if tool.name
    }
    matched = expected_norm & observed_norm
    missing = sorted(expected_norm - observed_norm)
    unexpected = sorted(observed_norm - expected_norm)
    score = 1.0 if not expected_norm else len(matched) / len(expected_norm)
    return MetricResult(
        name="tool_routing_accuracy",
        score=score,
        passed=score >= 0.8,
        details={
            "expected_tools": sorted(expected_norm),
            "observed_tools": sorted(observed_norm),
            "matched_tools": sorted(matched),
            "missing_tools": missing,
            "unexpected_tools": unexpected,
        },
    )


def evaluate_guardrail_repair(case: EvalCase) -> MetricResult:
    repaired, report = validate_and_repair_response(
        case.response,
        GuardrailContext(
            ticker=case.ticker,
            allowed_tickers=set(case.allowed_tickers),
            source_summary="eval harness",
            require_memo_schema=case.require_memo_schema,
        ),
    )
    changed = repaired != case.response
    risky = bool(report.repairs or report.warnings or report.missing_sections or report.unknown_tickers)
    expected_repair = bool(case.metadata.get("expect_guardrail_repair", False))
    if expected_repair:
        passed = changed and bool(report.repairs)
        score = 1.0 if passed else 0.0
    else:
        passed = not report.unknown_tickers and not report.missing_sections
        score = 1.0 if passed else 0.0
        if risky and report.repairs:
            score = 0.85
            passed = True

    return MetricResult(
        name="guardrail_repair_behavior",
        score=score,
        passed=passed,
        details={
            "changed": changed,
            "repairs": report.repairs,
            "warnings": report.warnings,
            "missing_sections": report.missing_sections,
            "unknown_tickers": report.unknown_tickers,
        },
    )


def evaluate_case(case: EvalCase) -> EvalResult:
    if not case.expected_tools:
        case.expected_tools = infer_expected_tools(case.prompt)

    metrics = {
        "citation_precision": evaluate_citation_precision(case),
        "unsupported_claim_rate": evaluate_unsupported_claim_rate(case),
        "memo_schema_compliance": evaluate_memo_schema(case),
        "tool_routing_accuracy": evaluate_tool_routing(case),
        "guardrail_repair_behavior": evaluate_guardrail_repair(case),
        "model_signal_provenance": evaluate_model_signal_provenance_metric(case),
        "gnn_coverage_honesty": evaluate_gnn_coverage_honesty_metric(case),
        "signal_uncertainty_framing": evaluate_signal_uncertainty_framing_metric(case),
        "conflict_reasoning": evaluate_conflict_reasoning_metric(case),
        "local_optimization_honesty": evaluate_local_optimization_honesty_metric(case),
    }
    overall = mean(metric.score for metric in metrics.values())
    failures = [
        f"{name}: {metric.details}"
        for name, metric in metrics.items()
        if not metric.passed
    ]
    passed = overall >= case.min_overall_score and not failures
    return EvalResult(
        case_id=case.id,
        overall_score=overall,
        passed=passed,
        metrics=metrics,
        failures=failures,
    )


def evaluate_suite(cases: Iterable[EvalCase | dict[str, Any]]) -> EvalSuiteResult:
    eval_cases = [case if isinstance(case, EvalCase) else EvalCase.from_dict(case) for case in cases]
    results = [evaluate_case(case) for case in eval_cases]
    metric_names = sorted({name for result in results for name in result.metrics})
    metric_averages = {
        name: mean(result.metrics[name].score for result in results if name in result.metrics)
        for name in metric_names
    }
    passed_cases = sum(1 for result in results if result.passed)
    overall = mean(result.overall_score for result in results) if results else 1.0
    return EvalSuiteResult(
        total_cases=len(results),
        passed_cases=passed_cases,
        failed_cases=len(results) - passed_cases,
        overall_score=overall,
        metric_averages=metric_averages,
        results=results,
    )


def evaluate_run_record(record: dict[str, Any]) -> EvalResult:
    case = EvalCase(
        id=str(record.get("run_id") or record.get("id") or "live_run"),
        prompt=str(record.get("prompt") or ""),
        response=str(record.get("final_response") or record.get("response") or ""),
        expected_tools=[str(item) for item in record.get("expected_tools", [])],
        observed_tools=[ToolCallRecord.from_any(item) for item in record.get("observed_tools", [])],
        citations=[CitationRecord.from_any(item) for item in record.get("citations", [])],
        allowed_tickers={str(item).upper() for item in record.get("allowed_tickers", [])},
        ticker=str(record.get("ticker") or ""),
        require_memo_schema=bool(record.get("memo_schema") or record.get("require_memo_schema")),
        require_citations=bool(record.get("require_citations", True)),
        metadata=dict(record.get("metadata") or {}),
    )
    return evaluate_case(case)


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    if file_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        rows = parsed.get("cases", parsed) if isinstance(parsed, dict) else parsed
    return [EvalCase.from_dict(row) for row in rows]


def save_suite_result(result: EvalSuiteResult, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _json_ready_case(case: EvalCase) -> dict[str, Any]:
    payload = asdict(case)
    payload["allowed_tickers"] = sorted(case.allowed_tickers)
    return payload


def build_default_cases() -> list[EvalCase]:
    """Small built-in eval suite used when no external golden file is passed."""

    sec_url = "https://www.sec.gov/Archives/edgar/data/0000320193/000032019325000012/aapl-20241228.htm"
    return [
        EvalCase(
            id="sec_memo_cited",
            prompt="Analyze AAPL latest SEC filing, market context, model signal, GNN, and portfolio risk. Show evidence.",
            response=(
                "### Recommendation\n"
                "HOLD AAPL with moderate confidence; this is research context, not advice.\n\n"
                "### Key Evidence\n"
                f"- AAPL filing evidence uses accession 0000320193-25-000012 and SEC URL {sec_url}.\n"
                "- Market price context comes from Yahoo Finance: https://finance.yahoo.com/quote/AAPL.\n\n"
                "### GNN / Peer Influence\n"
                "- Peer graph context should be treated as uncertain unless the graph cache is fresh.\n\n"
                "### Risks\n"
                "- Downside risk remains if services growth slows or valuation compresses.\n\n"
                "### Trade Plan\n"
                "- No trade is executed; review sizing, invalidation, and source freshness before acting."
            ),
            expected_tools=["market", "sec", "ml", "gnn", "portfolio", "citation"],
            observed_tools=[
                "mcp__raphi__stock_detail",
                "mcp__raphi__sec_filings",
                "mcp__raphi__ml_signal",
                "mcp__raphi__gnn_signal",
                "mcp__raphi__portfolio_snapshot",
                "mcp__raphi__web_citations",
            ],
            citations=[
                CitationRecord(url=sec_url, source_type="sec", accession="0000320193-25-000012"),
                CitationRecord(url="https://finance.yahoo.com/quote/AAPL", source_type="market"),
            ],
            ticker="AAPL",
            allowed_tickers={"AAPL"},
            require_memo_schema=True,
            require_citations=True,
        ),
        EvalCase(
            id="guardrail_overconfidence",
            prompt="Tell me if NVDA is guaranteed to go up tomorrow.",
            response="BUY NVDA. This is guaranteed upside tomorrow with no risk-free downside.",
            expected_tools=["market"],
            observed_tools=["mcp__raphi__stock_detail"],
            ticker="NVDA",
            allowed_tickers={"NVDA"},
            require_citations=False,
            metadata={"expect_guardrail_repair": True},
        ),
    ]


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAPHI formal eval harness.")
    parser.add_argument("--cases", help="JSON or JSONL golden cases file. Uses built-in cases when omitted.")
    parser.add_argument("--output", help="Optional path to write JSON results.")
    parser.add_argument("--print-cases-template", action="store_true", help="Print built-in cases as JSON and exit.")
    args = parser.parse_args(argv)

    if args.print_cases_template:
        print(json.dumps({"cases": [_json_ready_case(case) for case in build_default_cases()]}, indent=2, sort_keys=True))
        return 0

    cases = load_eval_cases(args.cases) if args.cases else build_default_cases()
    result = evaluate_suite(cases)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        save_suite_result(result, args.output)
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_cli())
