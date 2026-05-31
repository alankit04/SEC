from raphi.orchestrators.state import WorkflowState
from raphi.memory.ticker_registry import validate_ticker, register_ticker_interest, register_gnn_candidate

def onboard_tickers_for_query(state: WorkflowState) -> WorkflowState:
    from raphi.memory import ticker_registry
    for ticker in state.tickers:
        if ticker in state.validated_tickers or ticker in state.invalid_tickers:
            continue
        validation = ticker_registry.validate_ticker(ticker)
        if not validation.get("valid"):
            state.invalid_tickers.append(ticker)
            state.ticker_registration_status[ticker] = {
                "ticker": ticker,
                "valid": False,
                "company_name": validation.get("company_name", ""),
                "cik": validation.get("cik", ""),
                "source_used": validation.get("source_used", "unavailable"),
                "already_tracked": False,
                "registered_to_memory": False,
                "registered_to_watchlist": False,
                "gnn_candidate_added": False,
                "gnn_signal_available": False,
                "errors": validation.get("errors", ["ticker_validation_failed"]),
            }
            state.add_uncertainty_flag(f"Ticker {ticker} could not be validated.")
            continue
        # Only for valid tickers:
        state.validated_tickers.append(ticker)
        reg_result = ticker_registry.register_ticker_interest(ticker, state.user_query)
        state.ticker_registration_status[ticker] = reg_result
        if reg_result.get("error"):
            state.add_error({"ticker": ticker, "error": reg_result["error"]})
        gnn_result = ticker_registry.register_gnn_candidate(ticker, state.universe)
        state.gnn_registration_status[ticker] = gnn_result
        if gnn_result.get("gnn_candidate_added"):
            state.newly_registered_tickers.append(ticker)
        state.memory_updates.append({
            "ticker": ticker,
            "registered": reg_result.get("registered_to_memory", False),
            "gnn_candidate_added": gnn_result.get("gnn_candidate_added", False)
        })
    return state
