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
