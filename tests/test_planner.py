def test_explicit_dollar_syntax():
    p = perceive("Analyze $pltr using SEC filings")
    assert p["detected_tickers"] == ["PLTR"]

def test_explicit_ticker_colon():
    p = perceive("Analyze ticker:crwd using SEC filings")
    assert p["detected_tickers"] == ["CRWD"]

def test_explicit_nasdaq_colon():
    p = perceive("Analyze NASDAQ: NVDA using SEC filings")
    assert p["detected_tickers"] == ["NVDA"]

def test_class_share_ticker():
    p = perceive("Analyze BRK.B using SEC filings")
    assert p["detected_tickers"] == ["BRK.B"]

def test_market_risk_words_not_ticker():
    p = perceive("What are the current market risks?")
    assert p["detected_tickers"] == []
from raphi.orchestrators.planner import perceive, classify_intent

def test_raphi_not_detected_as_ticker():
    p = perceive("What is RAPHI?")
    assert p["detected_tickers"] == []
    assert classify_intent(p) == "casual_chat"

def test_common_words_not_tickers():
    p = perceive("What is AI and how does it work?")
    assert p["detected_tickers"] == []
    assert classify_intent(p) == "casual_chat"

def test_extracts_real_ticker_from_sec_query():
    p = perceive("Analyze PLTR using SEC filings")
    assert p["detected_tickers"] == ["PLTR"]

def test_latest_10q_extracts_nvda():
    p = perceive("Show me the latest 10-Q for NVDA")
    assert p["detected_tickers"] == ["NVDA"]
    assert classify_intent(p) == "latest_filing"

def test_recommendation_extracts_ticker_and_intent():
    p = perceive("Should I buy PLTR?")
    assert p["detected_tickers"] == ["PLTR"]
    assert classify_intent(p) == "recommendation"

def test_provided_tickers_are_preserved():
    p = perceive("analyze this company", user_context={"provided_tickers": ["crwd"]})
    assert p["detected_tickers"] == ["CRWD"]
