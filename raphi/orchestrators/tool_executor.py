import concurrent.futures
from datetime import datetime, timezone

from raphi.orchestrators.state import WorkflowState, ToolPlanStep, ToolTrace
from backend.retrieval_guardrail import screen_retrieval_result
import time


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_TOOL_TIMEOUT = {
    "sec_filings":        15,
    "edgar_live_summary": 15,
    "stock_detail":       10,
    "stock_news":         10,
    "ml_signal":          30,
    "portfolio_snapshot": 5,
}
_DEFAULT_TIMEOUT = 15


def _firecrawl_fallback(tool_name: str, kwargs: dict) -> dict | list | None:
    """
    Attempt to retrieve data via Firecrawl when the primary source times out.
    Returns result dict/list on success, None if Firecrawl is unavailable or also fails.
    """
    try:
        from backend.firecrawl_client import search_web, is_available
        if not is_available():
            return None

        ticker = kwargs.get("ticker", "")
        now = _now_iso()

        if tool_name in ("sec_filings", "edgar_live_summary"):
            query = f"{ticker} SEC 10-Q 10-K 8-K filing 2026"
            results = search_web(query, limit=5, scrape_results=True, max_chars_per_result=3000)
            hits = [r for r in results if r.get("success") and r.get("markdown")]
            if not hits:
                return None
            return [{
                "ticker":       ticker.upper(),
                "form":         "web-fallback",
                "filed":        now[:10],
                "accession":    "",
                "sec_url":      h.get("url", ""),
                "documents_url": h.get("url", ""),
                "source_type":  "sec",
                "provider":     "Firecrawl fallback",
                "retrieved_at": now,
                "raw_excerpt":  h.get("markdown", "")[:1500],
                "title":        h.get("title", ""),
            } for h in hits[:3]]

        if tool_name == "stock_detail":
            query = f"{ticker} stock price market cap fundamentals today"
            results = search_web(query, limit=3, scrape_results=True, max_chars_per_result=2000)
            hits = [r for r in results if r.get("success") and r.get("markdown")]
            if not hits:
                return None
            return {
                "ticker":       ticker.upper(),
                "source_type":  "market",
                "provider":     "Firecrawl fallback",
                "retrieved_at": now,
                "url":          hits[0].get("url", ""),
                "raw_excerpt":  hits[0].get("markdown", "")[:1500],
                "title":        hits[0].get("title", ""),
            }

        if tool_name == "stock_news":
            query = f"{ticker} stock news today"
            results = search_web(query, limit=5, scrape_results=False, max_chars_per_result=500)
            hits = [r for r in results if r.get("success")]
            if not hits:
                return None
            return {
                "ticker":       ticker.upper(),
                "source_type":  "news",
                "provider":     "Firecrawl fallback",
                "retrieved_at": now,
                "items": [{
                    "title":  h.get("title", ""),
                    "url":    h.get("url", ""),
                    "source": "Firecrawl web search",
                } for h in hits],
            }

    except Exception:
        pass
    return None


def execute_plan(state: WorkflowState) -> WorkflowState:

    def call_tool(tool_name, **kwargs):
        try:
            if tool_name == "sec_filings":
                from backend.edgar_live import get_recent_filings
                ticker = kwargs["ticker"]
                limit = kwargs.get("limit", 10)
                results = get_recent_filings(
                    ticker,
                    forms=["10-K", "10-Q", "8-K"],
                    days=180,
                    limit=limit,
                )
                if results:
                    return results
                from backend.sec_data import SECData
                return SECData(base_path=None).ticker_filings(ticker, limit=limit)

            elif tool_name == "edgar_live_summary":
                from backend.edgar_live import get_ticker_live_summary
                ticker = kwargs["ticker"]
                result = get_ticker_live_summary(ticker, days=kwargs.get("days", 90))
                if isinstance(result, dict):
                    result.setdefault("ticker", ticker)
                    result.setdefault("source_type", "sec")
                    result.setdefault("provider", "EDGAR live API")
                    result.setdefault("retrieved_at", _now_iso())
                return result

            elif tool_name == "stock_detail":
                from backend.market_data import MarketData
                result = MarketData().stock_detail(kwargs["ticker"])
                if isinstance(result, dict):
                    result.setdefault("ticker", kwargs["ticker"])
                    result.setdefault("source_type", "market")
                    result.setdefault("provider", "MarketData")
                    result.setdefault("retrieved_at", _now_iso())
                    result.setdefault("timestamp", result.get("retrieved_at"))
                return result

            elif tool_name == "stock_news":
                from backend.market_data import MarketData
                ticker = kwargs["ticker"]
                results = MarketData().stock_news(ticker, limit=kwargs.get("limit", 5))
                return {
                    "ticker":       ticker,
                    "source_type":  "news",
                    "provider":     "Yahoo Finance news via yfinance",
                    "retrieved_at": _now_iso(),
                    "items":        results or [],
                }

            elif tool_name == "portfolio_snapshot":
                from backend.portfolio_manager import PortfolioManager
                result = PortfolioManager().snapshot()
                if isinstance(result, dict):
                    result.setdefault("source_type", "portfolio")
                    result.setdefault("provider", "PortfolioManager")
                    result.setdefault("retrieved_at", _now_iso())
                return result

            elif tool_name == "ml_signal":
                from backend.market_data import MarketData
                from backend.ml_model import SignalEngine
                from backend.filing_classifier import FilingClassifier
                ticker = kwargs["ticker"]
                detail = MarketData().stock_detail(ticker)
                funds = {
                    "pe_ratio":       detail.get("pe_ratio")       if isinstance(detail, dict) else None,
                    "revenue_growth": detail.get("revenue_growth") if isinstance(detail, dict) else None,
                }
                result = SignalEngine().train_and_predict(ticker, funds)
                if isinstance(result, dict):
                    result.setdefault("ticker", ticker)
                    result.setdefault("source_type", "model")
                    result.setdefault("provider", "SignalEngine")
                    result.setdefault("retrieved_at", _now_iso())
                    result.setdefault("timestamp", result.get("retrieved_at"))
                # Filing text classifier — text-based signal from SEC filings
                filing_result = FilingClassifier.get().classify(ticker, financials=funds)
                if isinstance(result, dict) and isinstance(filing_result, dict):
                    result["filing_signal"]     = filing_result.get("signal")
                    result["filing_confidence"] = filing_result.get("confidence")
                    result["filing_reason"]     = filing_result.get("reason")
                    result["filing_source"]     = filing_result.get("source")
                return result

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"error": f"Tool {tool_name} unavailable: {e}"}

    for step in state.tool_plan:
        # Skip steps that already have a successful result from a previous attempt
        existing = state.retrieval_results.get(step.id)
        if existing is not None and not (isinstance(existing, dict) and existing.get("error")):
            continue

        start = time.time()
        ok = True
        error = None
        result = None
        timeout = _TOOL_TIMEOUT.get(step.tool_name, _DEFAULT_TIMEOUT)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(call_tool, step.tool_name, **step.args)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            pool.shutdown(wait=False)
            if step.required:
                # Primary source timed out — try Firecrawl before giving up
                fallback = _firecrawl_fallback(step.tool_name, step.args)
                if fallback:
                    result = fallback
                    error = None
                else:
                    # Firecrawl also failed — tell user clearly
                    ticker = step.args.get("ticker", "")
                    state.final_answer = (
                        f"Could not retrieve {step.tool_name}"
                        f"{f' for {ticker}' if ticker else ''} — "
                        f"primary source timed out after {timeout}s and the "
                        f"web fallback also returned no data. Please try again."
                    )
                    state.add_error({"tool": step.tool_name, "error": f"timed out after {timeout}s, fallback failed"})
                    return state
            else:
                # Optional tool — skip silently
                result = {"error": f"Tool {step.tool_name} timed out after {timeout}s (optional, skipped)"}
                ok = False
                error = result["error"]
        else:
            pool.shutdown(wait=False)

        if result is None:
            result = {"error": f"Tool {step.tool_name} returned no data"}

        if isinstance(result, dict) and result.get("error") and error is None:
            ok = False
            error = result["error"]

        end = time.time()
        # Control Plane 3 — retrieval gate. Screen external tool output for
        # prompt-injection payloads before it is trusted as context.
        result = screen_retrieval_result(step.tool_name, result)
        state.retrieval_results[step.id] = result
        state.add_trace(ToolTrace(
            tool_name=step.tool_name,
            args=step.args,
            started_at=datetime.fromtimestamp(start, tz=timezone.utc).isoformat(),
            ended_at=datetime.fromtimestamp(end, tz=timezone.utc).isoformat(),
            latency_ms=int((end - start) * 1000),
            ok=ok,
            error=error,
            output_summary=str(result)[:500],
            status="success" if ok else "failed",
        ))
        if not ok and step.required:
            state.add_error({"tool": step.tool_name, "error": error})

    return state
