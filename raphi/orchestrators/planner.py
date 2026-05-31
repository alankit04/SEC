import re
from typing import List, Dict, Optional
from .state import WorkflowState, ToolPlanStep

FALSE_TICKER_WORDS = set([
    # Single letters
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    # Common finance/general words
    "A", "AN", "AM", "IS", "ARE", "BE", "WAS", "WERE",
    "DO", "DOES", "DID",
    "I", "ME", "MY", "MINE", "YOU", "YOUR", "WE", "OUR",
    "IT", "ITS", "THIS", "THAT", "THESE", "THOSE",
    "CAN", "COULD", "WOULD", "SHOULD", "PLEASE",
    "TELL", "GIVE", "GET", "USE", "USING", "SHOW", "ANALYZE",
    "ABOUT", "WITH", "FROM", "INTO", "TO", "OF", "ON", "IN", "AS", "BY", "AT",
    "OR", "IF", "THEN", "THAN",
    "BUY", "SELL", "HOLD", "LONG", "SHORT",
    "SEC", "API", "AI", "ML", "GNN", "RAG",
    "CEO", "CFO", "EPS", "PE", "ETF", "USD", "USA",
    "THE", "AND", "FOR", "HOW", "WHY", "WHAT", "WHEN", "WHERE", "WHO", "WHICH",
    "LATEST", "CURRENT", "TODAY", "NOW", "RAPHI",
    # Non-ticker finance/general words
    "WORK", "DATA", "RISK", "CASH", "DEBT", "FILE", "FORM", "STOCK", "STOCKS", "MARKET", "PRICE", "PRICES", "FILING", "FILINGS", "NEWS", "COMPANY", "COMPANIES", "SHARE", "SHARES", "REPORT", "REPORTS", "YEAR", "YEARS", "MONTH", "WEEK", "DAY", "DAYS", "TOP", "HOT", "MOVERS", "TRENDING", "CURRENT", "LATEST", "RECENT"
])
# --- Ticker Extraction ---
def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = value.upper().strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out

def extract_tickers(user_query: str, user_context: Optional[Dict] = None) -> List[str]:
    candidates = []
    # 1. Always trust provided_tickers after normalization/filtering
    if user_context and user_context.get("provided_tickers"):
        for t in user_context["provided_tickers"]:
            t = str(t).upper().strip()
            if t and t not in FALSE_TICKER_WORDS:
                candidates.append(t)

    # 2. Preprocess: remove/normalize filing forms
    query_for_tickers = re.sub(r"\b(10-Q|10-K|8-K|S-1|FORM\s*4)\b", " ", user_query, flags=re.I)

    # 3. Explicit ticker syntax: $PLTR, ticker:pltr, NASDAQ: PLTR, NYSE: PLTR
    explicit_patterns = [
        r"\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)",
        r"ticker:([A-Za-z]{1,5}(?:\.[A-Za-z])?)",
        r"(?:NASDAQ|NYSE|AMEX|OTC):\s*([A-Za-z]{1,5}(?:\.[A-Za-z])?)"
    ]
    for pat in explicit_patterns:
        for match in re.findall(pat, query_for_tickers, flags=re.I):
            t = match.upper().strip()
            if t and t not in FALSE_TICKER_WORDS:
                candidates.append(t)

    # 4. Class-share tickers (e.g., BRK.B)
    raw_tokens = re.findall(r"\b[A-Za-z]{1,5}(?:\.[A-Za-z])?\b", query_for_tickers)
    for token in raw_tokens:
        upper = token.upper().strip()
        # Only allow plain lowercase as ticker if from explicit syntax or provided_tickers
        if token.islower() and upper not in candidates:
            continue
        if upper in FALSE_TICKER_WORDS:
            continue
        if len(upper.replace(".", "")) > 5:
            continue
        # Don't add if already added from explicit/provided
        if upper not in candidates:
            candidates.append(upper)
    return _dedupe_preserve_order(candidates)
FRESHNESS_TERMS = [
    "latest", "recent", "today", "current", "now", "this week", "this month", "new", "just filed", "2026", "up to date", "updated"
]

# --- Perceive ---
def perceive(user_query: str, history: Optional[List]=None, user_context: Optional[Dict]=None) -> dict:
    raw_query = user_query
    normalized_query = user_query.strip().upper()
    query_lc = user_query.lower()
    detected_tickers = extract_tickers(user_query, user_context)
    detected_entities = []
    possible_intents = []
    freshness_terms = [term for term in FRESHNESS_TERMS if term.upper() in normalized_query]
    freshness_required = bool(freshness_terms)

    # Data-need flags — authoritative single source of truth used by build_plan()
    sec_terms = ["sec", "filing", "filings", "10-q", "10-k", "8-k", "form 4"]
    market_terms = ["market", "price", "stock", "quote", "data"]
    sec_data_needed = any(term in query_lc for term in sec_terms)
    market_data_needed = any(term in query_lc for term in market_terms)
    model_signal_needed = any(t in normalized_query for t in ["MODEL", "SIGNAL", "GNN", "ML"])
    web_citation_needed = "WEB" in normalized_query or "NEWS" in normalized_query
    memory_needed = "MEMORY" in normalized_query
    portfolio_needed = "PORTFOLIO" in normalized_query
    recommendation_requested = any(w in normalized_query for w in ["BUY", "SELL", "HOLD", "LONG", "SHORT", "SHOULD I"])
    confidence_score_requested = "CONFIDENCE" in normalized_query

    # Market-scope flags for trending queries
    watchlist_terms = ["my watchlist", "watchlist", "my stocks", "tracked stocks", "my tracked stocks", "quick watchlist"]
    trending_terms = ["trending", "top stocks", "movers", "hot stocks"]
    watchlist_scope_requested = any(term in query_lc for term in watchlist_terms)
    if watchlist_scope_requested:
        market_scope = "user_watchlist"
    elif any(term in query_lc for term in trending_terms):
        market_scope = "broad_market"
    else:
        market_scope = None

    return {
        "raw_query": raw_query,
        "normalized_query": normalized_query,
        "detected_tickers": detected_tickers,
        "detected_entities": detected_entities,
        "possible_intents": possible_intents,
        "freshness_terms": freshness_terms,
        "freshness_required": freshness_required,
        "sec_data_needed": sec_data_needed,
        "market_data_needed": market_data_needed,
        "model_signal_needed": model_signal_needed,
        "web_citation_needed": web_citation_needed,
        "memory_needed": memory_needed,
        "portfolio_needed": portfolio_needed,
        "recommendation_requested": recommendation_requested,
        "confidence_score_requested": confidence_score_requested,
        "watchlist_scope_requested": watchlist_scope_requested,
        "market_scope": market_scope,
    }

# --- Intent Classification ---
def classify_intent(perception: dict) -> str:
    # Pure function: reads perception, returns intent string, does NOT mutate.
    # All flags (sec_data_needed, market_scope, etc.) are set once in perceive().
    nq = perception.get("normalized_query") or perception.get("raw_query", "").upper()
    query = perception.get("raw_query", "")
    query_lc = query.lower()
    # 1. Recommendation
    if any(term in nq for term in ["BUY", "SELL", "HOLD", "LONG", "SHORT", "SHOULD I"]):
        return "recommendation"
    # 2. Latest/SEC filings
    if any(term in nq for term in ["LATEST", "RECENT", "JUST FILED", "CURRENT", "NOW", "10-Q", "10-K", "8-K", "FORM 4"]):
        if "10-Q" in nq or "10-K" in nq:
            return "latest_filing"
        return "sec_research"
    # 3. Trending
    if any(term in nq for term in ["TRENDING", "TOP STOCKS", "MOVERS", "HOT STOCKS"]):
        return "trending_stocks"
    # 4. Model signal
    if any(term in nq for term in ["GNN", "ML", "MODEL", "SIGNAL", "PREDICTION", "CONFIDENCE"]):
        return "model_signal"
    # 5. Portfolio
    if any(term in nq for term in ["PORTFOLIO", "VAR", "SHARPE", "EXPOSURE", "POSITION"]):
        return "portfolio_risk"
    # 6. Investment memo
    if any(term in nq for term in ["MEMO", "INVESTMENT THESIS", "CONVICTION"]):
        return "investment_memo"
    # 7. Company factual if tickers present or company keyword present
    company_keywords = ["stock", "price", "market cap", "sec", "filing", "earnings", "dividend", "ipo", "ticker", "about", "with", "from", "into"]
    if len(perception.get("detected_tickers", [])) > 0:
        return "company_factual"
    if any(re.search(rf"\b{re.escape(word.lower())}\b", query_lc) for word in company_keywords):
        return "company_factual"
    # 8. Otherwise, casual_chat
    return "casual_chat"

def classify_risk(intent: str, perception: dict) -> str:
    if intent == "recommendation":
        return "high"
    if intent in ["investment_memo", "portfolio_risk"]:
        return "high"
    if intent in ["model_signal", "trending_stocks", "latest_filing", "market_snapshot"]:
        return "medium"
    if intent in ["casual_chat", "company_factual"]:
        return "low"
    return "medium"

def build_plan(state: WorkflowState) -> List[ToolPlanStep]:
    """
    Intent-first plan builder. Intent determines the canonical tool set for the query.
    Perception flags (sec_data_needed, market_data_needed) only ADD optional steps within
    an intent — they never override or bypass the intent-based canonical plan.
    """
    plan = []
    tid = 0
    def next_id():
        nonlocal tid
        tid += 1
        return f"step{tid}"

    intent = state.intent
    tickers = state.validated_tickers
    p = state.perception or {}

    if intent == "trending_stocks":
        pass  # run_trending_stocks_workflow owns this entirely

    elif intent == "company_factual":
        for ticker in tickers:
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_detail",
                purpose=f"Get live market and company profile for {ticker}",
                required=True, args={"ticker": ticker},
                expected_output="Market quote, price, market cap, sector, fundamentals"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_news",
                purpose=f"Get latest news for {ticker}",
                required=True, args={"ticker": ticker, "limit": 5},
                expected_output="Recent news headlines with sentiment"
            ))
            if p.get("sec_data_needed"):
                plan.append(ToolPlanStep(
                    id=next_id(), tool_name="sec_filings",
                    purpose=f"Get live SEC filings for {ticker}",
                    required=False, args={"ticker": ticker, "limit": 5},
                    expected_output="Recent 10-Q, 10-K, 8-K filings"
                ))

    elif intent in ["sec_research", "latest_filing"]:
        for ticker in tickers:
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="sec_filings",
                purpose=f"Get live SEC filings for {ticker} from EDGAR API",
                required=True, args={"ticker": ticker, "limit": 5},
                expected_output="SEC filings list with filing dates, forms, accession/URL"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="edgar_live_summary",
                purpose=f"Get live 8-K events and insider activity for {ticker}",
                required=False, args={"ticker": ticker, "days": 90},
                expected_output="Recent 8-Ks, Form 4 insider transactions"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_news",
                purpose=f"Get latest news for {ticker}",
                required=True, args={"ticker": ticker, "limit": 5},
                expected_output="Recent news headlines"
            ))

    elif intent == "recommendation":
        for ticker in tickers:
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="sec_filings",
                purpose=f"Get SEC filings for {ticker} — required for recommendation evidence",
                required=True, args={"ticker": ticker, "limit": 5},
                expected_output="SEC filings list"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="edgar_live_summary",
                purpose=f"Get live 8-K events and insider activity for {ticker}",
                required=False, args={"ticker": ticker, "days": 90},
                expected_output="Recent 8-Ks, Form 4s"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_detail",
                purpose=f"Get live market data for {ticker}",
                required=True, args={"ticker": ticker},
                expected_output="Market quote and company fundamentals"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_news",
                purpose=f"Get latest news for {ticker}",
                required=True, args={"ticker": ticker, "limit": 5},
                expected_output="Recent news headlines"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="ml_signal",
                purpose=f"Get model signal for {ticker} — needed for recommendation provenance",
                required=False, args={"ticker": ticker},
                expected_output="Model signal direction and confidence"
            ))

    elif intent == "model_signal":
        for ticker in tickers:
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="ml_signal",
                purpose=f"Generate model signal and confidence provenance for {ticker}",
                required=False, args={"ticker": ticker},
                expected_output="Model signal or explicit unavailable status"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_detail",
                purpose=f"Get live market context for {ticker} signal interpretation",
                required=False, args={"ticker": ticker},
                expected_output="Market quote and fundamentals"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_news",
                purpose=f"Get latest news for {ticker}",
                required=True, args={"ticker": ticker, "limit": 5},
                expected_output="Recent news headlines"
            ))

    elif intent == "investment_memo":
        for ticker in tickers:
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="sec_filings",
                purpose=f"Get SEC filings for {ticker} to support memo",
                required=True, args={"ticker": ticker, "limit": 5},
                expected_output="SEC filings with forms, dates, accession numbers"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="edgar_live_summary",
                purpose=f"Get live 8-K events and insider activity for {ticker}",
                required=False, args={"ticker": ticker, "days": 90},
                expected_output="Recent 8-Ks and Form 4 insider transactions"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_detail",
                purpose=f"Get market and company profile for {ticker}",
                required=True, args={"ticker": ticker},
                expected_output="Market quote, sector, market cap, fundamentals"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_news",
                purpose=f"Get latest news for {ticker}",
                required=True, args={"ticker": ticker, "limit": 5},
                expected_output="Recent news headlines"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="ml_signal",
                purpose=f"Get model signal for {ticker} to include in memo",
                required=False, args={"ticker": ticker},
                expected_output="Model signal direction, confidence, SHAP features"
            ))

    elif intent == "portfolio_risk":
        plan.append(ToolPlanStep(
            id=next_id(), tool_name="portfolio_snapshot",
            purpose="Retrieve current portfolio positions and weights",
            required=True, args={},
            expected_output="Portfolio holdings, weights, P&L, VaR, Sharpe ratio"
        ))
        for ticker in tickers:
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_detail",
                purpose=f"Get live market detail for portfolio position {ticker}",
                required=False, args={"ticker": ticker},
                expected_output="Market quote and price change for risk context"
            ))
            plan.append(ToolPlanStep(
                id=next_id(), tool_name="stock_news",
                purpose=f"Get latest news for {ticker}",
                required=True, args={"ticker": ticker, "limit": 3},
                expected_output="Recent news headlines for risk context"
            ))

    return plan
