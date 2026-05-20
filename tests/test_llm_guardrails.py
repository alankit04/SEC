import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from llm_guardrails import GuardrailContext, validate_and_repair_response


def test_guardrails_repair_overconfidence_and_add_risk_context():
    text = "BUY NVDA. This is guaranteed upside to $250 with no risk-free downside."

    repaired, report = validate_and_repair_response(
        text,
        GuardrailContext(
            ticker="NVDA",
            allowed_tickers={"NVDA"},
            source_summary="unit-test market context",
        ),
    )

    assert "guaranteed upside" not in repaired.lower()
    assert "Guardrail Notes" in repaired
    assert "Data provenance checked" in repaired
    assert report.repairs


def test_guardrails_do_not_rewrite_not_guaranteed():
    text = "HOLD ASST. Investment outcomes are not guaranteed and downside risk remains."

    repaired, report = validate_and_repair_response(
        text,
        GuardrailContext(ticker="ASST", allowed_tickers={"ASST"}),
    )

    assert "not guaranteed" in repaired
    assert "not uncertain" not in repaired
    assert not report.repairs


def test_guardrails_enforce_memo_sections():
    text = "### Recommendation\nHOLD NVDA with 55% confidence.\n\n### Risks\n- Multiple compression."

    repaired, report = validate_and_repair_response(
        text,
        GuardrailContext(
            ticker="NVDA",
            allowed_tickers={"NVDA"},
            require_memo_schema=True,
        ),
    )

    assert "### Key Evidence" in repaired
    assert "### GNN / Peer Influence" in repaired
    assert "### Trade Plan" in repaired
    assert "added missing memo schema sections" in report.repairs


def test_guardrails_allow_financial_acronyms():
    text = "HOLD NVDA. EPS and FCF are improving, but downside risk remains."

    repaired, report = validate_and_repair_response(
        text,
        GuardrailContext(ticker="NVDA", allowed_tickers={"NVDA"}),
    )

    assert "Unverified ticker references" not in repaired
    assert "EPS" not in report.unknown_tickers
    assert "FCF" not in report.unknown_tickers


def test_guardrails_allow_common_market_references():
    repaired, report = validate_and_repair_response(
        "ASST has Bitcoin treasury exposure, so BTC sentiment and SPY risk matter.",
        GuardrailContext(ticker="ASST", allowed_tickers={"ASST"}),
    )

    assert "Unverified ticker references" not in repaired
    assert "BTC" not in report.unknown_tickers
    assert "SPY" not in report.unknown_tickers


def test_guardrails_allow_raphi_brand_name():
    repaired, report = validate_and_repair_response(
        "RAPHI says HOLD ASST, but downside risk remains.",
        GuardrailContext(ticker="ASST", allowed_tickers={"ASST"}),
    )

    assert "Unverified ticker references" not in repaired
    assert "RAPHI" not in report.unknown_tickers
