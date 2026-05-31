"""
Dynamic evidence provider router for RAPHI.
Selects evidence sources based on query, intent, and tickers.
"""
from typing import List, Dict

FRESHNESS_KEYWORDS = [
    "latest", "current", "recent", "today", "now", "this week", "this month", "updated", "up to date", "new", "just filed", "trending", "movers", "2026", "should I", "buy", "sell", "hold", "long", "short"
]

def select_evidence_sources(query: str, intent: str, tickers: List[str]) -> List[Dict]:
    """
    Returns a list of provider plan entries, e.g.:
    {"source_type": "sec", "provider": "edgar", "required": True, "reason": "latest SEC filing requested"}
    """
    plan = []
    q = query.lower() if query else ""
    requires_freshness = any(kw in q for kw in FRESHNESS_KEYWORDS)

    if intent in ("latest_filing", "sec_research"):
        plan.append({"source_type": "sec", "provider": "edgar", "required": True, "reason": "SEC research required"})
        plan.append({"source_type": "web", "provider": "firecrawl", "required": False, "reason": "web context optional"})
        plan.append({"source_type": "company", "provider": "firecrawl", "required": False, "reason": "company IR optional"})
    elif intent == "market_snapshot":
        plan.append({"source_type": "market", "provider": "market_data", "required": True, "reason": "market data required"})
        plan.append({"source_type": "news", "provider": "firecrawl", "required": False, "reason": "news optional"})
        plan.append({"source_type": "web", "provider": "firecrawl", "required": False, "reason": "web context optional"})
    elif intent == "trending_stocks":
        plan.append({"source_type": "market", "provider": "market_data", "required": True, "reason": "market data required"})
        plan.append({"source_type": "news", "provider": "firecrawl", "required": True, "reason": "news required"})
        plan.append({"source_type": "web", "provider": "firecrawl", "required": True, "reason": "web context required"})
        plan.append({"source_type": "sec", "provider": "edgar", "required": False, "reason": "SEC optional for ranked tickers"})
        plan.append({"source_type": "model", "provider": "ml_model", "required": False, "reason": "model signals optional"})
    elif intent == "recommendation":
        plan.append({"source_type": "sec", "provider": "edgar", "required": True, "reason": "SEC required"})
        plan.append({"source_type": "market", "provider": "market_data", "required": True, "reason": "market required"})
        plan.append({"source_type": "news", "provider": "firecrawl", "required": True, "reason": "news required"})
        plan.append({"source_type": "web", "provider": "firecrawl", "required": True, "reason": "web required"})
        plan.append({"source_type": "model", "provider": "ml_model", "required": True, "reason": "model required"})
        plan.append({"source_type": "portfolio", "provider": "portfolio_manager", "required": False, "reason": "portfolio context optional"})
        plan.append({"source_type": "governance", "provider": "governance", "required": True, "reason": "governance required"})
    elif intent == "company_factual":
        plan.append({"source_type": "market", "provider": "market_data", "required": False, "reason": "market optional"})
        plan.append({"source_type": "company", "provider": "firecrawl", "required": False, "reason": "company IR optional"})
        plan.append({"source_type": "web", "provider": "firecrawl", "required": False, "reason": "web context optional"})
        plan.append({"source_type": "sec", "provider": "edgar", "required": False, "reason": "SEC optional if public ticker"})
    elif intent == "model_signal":
        plan.append({"source_type": "model", "provider": "ml_model", "required": True, "reason": "model required"})
        plan.append({"source_type": "market", "provider": "market_data", "required": False, "reason": "market optional"})
        plan.append({"source_type": "sec", "provider": "edgar", "required": False, "reason": "SEC optional"})
    else:
        # Default: try to get as much as possible
        plan.append({"source_type": "web", "provider": "firecrawl", "required": False, "reason": "web context"})
        plan.append({"source_type": "news", "provider": "firecrawl", "required": False, "reason": "news context"})
        plan.append({"source_type": "market", "provider": "market_data", "required": False, "reason": "market context"})
        plan.append({"source_type": "sec", "provider": "edgar", "required": False, "reason": "SEC context"})
        plan.append({"source_type": "company", "provider": "firecrawl", "required": False, "reason": "company IR"})
        plan.append({"source_type": "model", "provider": "ml_model", "required": False, "reason": "model signals"})
        plan.append({"source_type": "portfolio", "provider": "portfolio_manager", "required": False, "reason": "portfolio context"})
        plan.append({"source_type": "memory", "provider": "graph_memory", "required": False, "reason": "memory context"})
    return plan
