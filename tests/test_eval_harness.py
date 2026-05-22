import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from eval_harness import (  # noqa: E402
    evaluate_model_signal_provenance,
    evaluate_gnn_coverage_honesty,
    evaluate_signal_uncertainty_framing,
    evaluate_conflict_reasoning,
    evaluate_local_optimization_honesty,
    CitationRecord,
    EvalCase,
    build_default_cases,
    evaluate_case,
    evaluate_suite,
    extract_citations,
    infer_expected_tools,
    load_eval_cases,
    save_suite_result,
)


def test_infer_expected_tools_for_complex_sec_question():
    tools = infer_expected_tools(
        "Analyze ASST using SEC filings, market performance, ML/GNN signal, portfolio risk, and citations."
    )

    assert {"market", "sec", "ml", "gnn", "portfolio", "citation"} <= set(tools)


def test_extract_citations_finds_sec_urls_and_accessions():
    text = (
        "Evidence: https://www.sec.gov/Archives/edgar/data/123/0000000123-25-000001-index.htm "
        "and accession 0000000123-25-000001."
    )

    citations = extract_citations(text)

    assert any(c.url.startswith("https://www.sec.gov/Archives") for c in citations)
    assert any(c.accession == "0000000123-25-000001" for c in citations)


def test_eval_case_passes_when_tools_citations_schema_and_guardrails_are_good():
    sec_url = "https://www.sec.gov/Archives/edgar/data/1318605/000162828025000001/tsla-20241231.htm"
    case = EvalCase(
        id="tsla_good",
        prompt="Analyze TSLA with SEC filings, market context, ML/GNN signal, portfolio risk, and evidence.",
        response=(
            "### Recommendation\n"
            "HOLD TSLA with uncertainty; no trade is executed.\n\n"
            "### Key Evidence\n"
            f"- TSLA SEC evidence uses accession 0001628280-25-000001 and filing URL {sec_url}.\n"
            "- Market source: https://finance.yahoo.com/quote/TSLA.\n\n"
            "### GNN / Peer Influence\n"
            "- Graph context is supportive but uncertain.\n\n"
            "### Risks\n"
            "- Downside risk remains if margins compress.\n\n"
            "### Trade Plan\n"
            "- Review sizing and invalidation before acting."
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
            CitationRecord(url=sec_url, source_type="sec", accession="0001628280-25-000001"),
            CitationRecord(url="https://finance.yahoo.com/quote/TSLA", source_type="market"),
        ],
        ticker="TSLA",
        allowed_tickers={"TSLA"},
        require_memo_schema=True,
    )

    result = evaluate_case(case)

    assert result.passed, result.to_dict()
    assert result.metrics["tool_routing_accuracy"].score == 1.0
    assert result.metrics["memo_schema_compliance"].score == 1.0
    assert result.metrics["citation_precision"].passed


def test_eval_case_fails_missing_tools_and_unsupported_claims():
    case = EvalCase(
        id="bad_answer",
        prompt="Analyze ASST using SEC filings, market context, ML/GNN, portfolio risk, and citations.",
        response=(
            "### Recommendation\n"
            "BUY ASST. Revenue grew 500 percent and the stock will definitely double tomorrow. "
            "### Risks\n"
            "None."
        ),
        expected_tools=["market", "sec", "ml", "gnn", "portfolio", "citation"],
        observed_tools=["mcp__raphi__stock_detail"],
        ticker="ASST",
        allowed_tickers={"ASST"},
        require_memo_schema=True,
    )

    result = evaluate_case(case)

    assert not result.passed
    assert not result.metrics["tool_routing_accuracy"].passed
    assert not result.metrics["unsupported_claim_rate"].passed
    assert not result.metrics["memo_schema_compliance"].passed

def test_model_signal_provenance():
    answer = "The model signal was trained on 2025 data and uses XGBoost v2.1. Provenance: local ML cache."
    assert evaluate_model_signal_provenance(answer, {})

def test_gnn_coverage_honesty():
    answer = "GNN unavailable for this ticker, so no graph context used."
    assert evaluate_gnn_coverage_honesty(answer, {})

def test_signal_uncertainty_framing():
    answer = "This is a probabilistic signal with 62% confidence. There is significant uncertainty."
    assert evaluate_signal_uncertainty_framing(answer, {})

def test_conflict_reasoning():
    answer = "ML is bullish but GNN is bearish, so conviction is reduced."
    assert evaluate_conflict_reasoning(answer, {})

def test_local_optimization_honesty():
    answer = "Local RL/distillation was used for signal optimization, not LLM fine-tuning."
    assert evaluate_local_optimization_honesty(answer, {})


def test_ml_gnn_strict_good_answer_passes():
    case = EvalCase(
        id="ml_gnn_strict_good",
        prompt="Analyze NVDA with ML/GNN and include provenance.",
        response=(
            "### Recommendation\n"
            "HOLD with uncertainty.\n\n"
            "### Key Evidence\n"
            "- Model signal provenance: trained 2025 using XGBoost signal v2.\n"
            "- GNN updated and available for NVDA.\n"
            "- This is probabilistic and not certain.\n\n"
            "### GNN / Peer Influence\n"
            "- Peer context is supportive but uncertain.\n\n"
            "### Risks\n"
            "- Risk remains if demand weakens.\n\n"
            "### Trade Plan\n"
            "- Size conservatively with invalidation."
        ),
        expected_tools=["ml", "gnn"],
        observed_tools=["mcp__raphi__ml_signal", "mcp__raphi__gnn_signal"],
        ticker="NVDA",
        allowed_tickers={"NVDA"},
        require_memo_schema=True,
        require_citations=False,
        metadata={"enforce_ml_metrics": True},
    )
    result = evaluate_case(case)
    assert result.metrics["model_signal_provenance"].passed
    assert result.metrics["gnn_coverage_honesty"].passed
    assert result.metrics["signal_uncertainty_framing"].passed


def test_ml_gnn_strict_overclaim_fails():
    case = EvalCase(
        id="ml_gnn_overclaim",
        prompt="Analyze NVDA with ML/GNN signal.",
        response="ML guarantees NVDA will go up and GNN proves certainty.",
        expected_tools=["ml", "gnn"],
        observed_tools=["mcp__raphi__ml_signal", "mcp__raphi__gnn_signal"],
        ticker="NVDA",
        allowed_tickers={"NVDA"},
        require_citations=False,
        metadata={"enforce_ml_metrics": True},
    )
    result = evaluate_case(case)
    assert not result.metrics["signal_uncertainty_framing"].passed


def test_ml_gnn_unavailable_honesty_passes():
    case = EvalCase(
        id="ml_gnn_unavailable_honest",
        prompt="Analyze a ticker where GNN may be missing.",
        response="GNN unavailable for this ticker; no graph context used. Signal remains probabilistic.",
        expected_tools=["ml", "gnn"],
        observed_tools=["mcp__raphi__ml_signal"],
        ticker="ASST",
        allowed_tickers={"ASST"},
        require_citations=False,
        metadata={"enforce_ml_metrics": True},
    )
    result = evaluate_case(case)
    assert result.metrics["gnn_coverage_honesty"].passed


def test_ml_gnn_conflict_reasoning_passes_when_disclosed():
    case = EvalCase(
        id="ml_gnn_conflict",
        prompt="Analyze ML/GNN conflict and conviction.",
        response=(
            "ML is bullish while GNN is bearish; signals disagree. "
            "Conviction is lowered and position sizing should be reduced."
        ),
        expected_tools=["ml", "gnn"],
        observed_tools=["mcp__raphi__ml_signal", "mcp__raphi__gnn_signal"],
        ticker="NVDA",
        allowed_tickers={"NVDA"},
        require_citations=False,
        metadata={"enforce_ml_metrics": True},
    )
    result = evaluate_case(case)
    assert result.metrics["conflict_reasoning"].passed


def test_local_optimization_honesty_in_strict_mode_passes():
    case = EvalCase(
        id="ml_local_opt_honesty",
        prompt="Describe local RL and distillation optimization behavior.",
        response="Local RL/distillation optimization is used for signal tuning, not LLM fine-tuning.",
        expected_tools=["ml"],
        observed_tools=["mcp__raphi__ml_signal"],
        ticker="NVDA",
        allowed_tickers={"NVDA"},
        require_citations=False,
        metadata={"enforce_ml_metrics": True},
    )
    result = evaluate_case(case)
    assert result.metrics["local_optimization_honesty"].passed

def test_guardrail_repair_behavior_is_scored_when_expected():
    case = EvalCase(
        id="guardrail",
        prompt="Is NVDA guaranteed to go up?",
        response="BUY NVDA. This is guaranteed upside with no risk-free downside.",
        expected_tools=["market"],
        observed_tools=["mcp__raphi__stock_detail"],
        ticker="NVDA",
        allowed_tickers={"NVDA"},
        require_citations=False,
        metadata={"expect_guardrail_repair": True},
    )

    result = evaluate_case(case)

    assert result.metrics["guardrail_repair_behavior"].passed
    assert result.metrics["guardrail_repair_behavior"].details["repairs"]


def test_suite_load_save_and_default_cases(tmp_path):
    cases = build_default_cases()
    suite = evaluate_suite(cases)

    assert suite.total_cases == 2
    assert suite.passed

    output = tmp_path / "result.json"
    save_suite_result(suite, output)
    saved = json.loads(output.read_text())
    assert saved["passed"] is True

    case_file = tmp_path / "cases.json"
    case_file.write_text(json.dumps({"cases": [cases[0].__dict__]}, default=lambda o: list(o) if isinstance(o, set) else o.__dict__))
    loaded = load_eval_cases(case_file)
    assert loaded[0].id == cases[0].id
