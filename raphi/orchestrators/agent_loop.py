import datetime
import uuid

from raphi.orchestrators.planner import perceive, classify_intent, classify_risk, build_plan
from raphi.orchestrators.state import WorkflowState
from raphi.workflows.ticker_onboarding_workflow import onboard_tickers_for_query
from raphi.evals.citation_freshness import infer_freshness_requirement
from raphi.workflows.research_workflow import run_research_workflow
from raphi.orchestrators.tool_executor import execute_plan
from raphi.orchestrators.reflector import reflect

def run_agentic_query(query: str, history=None, user_context=None) -> WorkflowState:
    perception = perceive(query, history, user_context)
    intent = classify_intent(perception)
    risk_class = classify_risk(intent, perception)
    run_id = str(uuid.uuid4())
    tickers = perception.get("detected_tickers", [])
    state = WorkflowState(
        run_id=run_id,
        user_query=query,
        intent=intent,
        risk_class=risk_class,
        entities=perception.get("detected_entities", []),
        tickers=tickers
    )
    state.perception = perception
    # 1. Ticker onboarding if needed
    if tickers and intent != "casual_chat":
        state = onboard_tickers_for_query(state)
        if len(state.validated_tickers) == 0:
            state.final_answer = "I could not validate this ticker from SEC/company/market sources, so I cannot register or analyze it yet."
            return state
        onboarding_only = (
            bool(user_context and user_context.get("provided_tickers"))
            and any(term in query.lower() for term in ["register", "onboard", "track", "add"])
            and not any(term in query.lower() for term in ["analyze", "should", "sec", "filing", "price", "market data"])
        )
        if onboarding_only:
            return state
    # 2. Casual chat: lightweight answer, no onboarding/tools
    if intent == "casual_chat":
        state.final_answer = "This is a general question. No tickers detected. No analysis required."
        return state
    # 3. Recommendation: always run research_workflow, but downgrade if no model/governance
    if intent == "recommendation":
        state = run_research_workflow(state)
        return state
    # 4. Trending stocks: run trending_stocks_workflow
    if intent == "trending_stocks":
        from raphi.workflows.trending_stocks_workflow import run_trending_stocks_workflow
        state = run_trending_stocks_workflow(query)
        return state
    # 5. Latest/SEC/Company/model factual: run research_workflow
    if intent in ["company_factual", "sec_research", "latest_filing", "model_signal", "portfolio_risk", "investment_memo"]:
        state = run_research_workflow(state)
        return state
    # 6. Fallback: all other intents go through research workflow
    state = run_research_workflow(state)
    return state
