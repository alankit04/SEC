import json
import os
from typing import Optional, Dict

_LOCAL_TICKER_INTERESTS: dict[str, dict] = {}


def validate_ticker(ticker: str) -> Dict:
    ticker = str(ticker or "").upper().strip()
    # 1. Check company_tickers.json
    tickers_path = os.path.join(os.path.dirname(__file__), '../../company_tickers.json')
    try:
        with open(tickers_path, 'r') as f:
            tickers = json.load(f)
        entry = None
        if ticker in tickers and isinstance(tickers[ticker], dict):
            entry = tickers[ticker]
        else:
            for candidate in tickers.values() if isinstance(tickers, dict) else []:
                if isinstance(candidate, dict) and str(candidate.get("ticker", "")).upper() == ticker:
                    entry = candidate
                    break
        if entry:
            return {
                "ticker": ticker,
                "valid": True,
                "company_name": entry.get("title") or entry.get("name"),
                "cik": entry.get("cik_str") or entry.get("cik"),
                "source_used": "company_tickers",
                "errors": []
            }
    except Exception:
        pass
    # 2. Check SECData cik_for_ticker
    try:
        from backend.sec_data import SECData
        sec = SECData(base_path=None)
        cik = sec.cik_for_ticker(ticker)
        if cik:
            return {
                "ticker": ticker.upper(),
                "valid": True,
                "company_name": None,
                "cik": cik,
                "source_used": "SECData",
                "errors": []
            }
    except Exception:
        pass
    # 3. Check market/yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        if info and info.get("shortName"):
            return {
                "ticker": ticker.upper(),
                "valid": True,
                "company_name": info.get("shortName"),
                "cik": info.get("cik", None),
                "source_used": "yfinance",
                "errors": []
            }
    except Exception:
        pass
    # 4. If all fail
    return {
        "ticker": ticker.upper(),
        "valid": False,
        "company_name": None,
        "cik": None,
        "source_used": None,
        "errors": ["Ticker not found in company_tickers.json, SECData, or yfinance"]
    }

def register_ticker_interest(ticker: str, user_query: str, user_id: str = "anonymous", source: str = "user_query") -> Dict:
    ticker = str(ticker or "").upper().strip()
    # Register in graph memory
    memory_id = None
    registered_to_memory = False
    error = None
    try:
        from backend.graph_memory import get_graph_memory
        gm = get_graph_memory()
        memory_id = gm.remember_interaction(
            user_text=f"User asked about ticker {ticker}: {user_query}",
            source=source,
            metadata={"ticker": ticker, "query": user_query},
            user_id=user_id,
            importance=0.55,
        ).get("memory_id")
        registered_to_memory = True
    except Exception as exc:
        error = f"graph_memory error: {exc}"
        local_key = f"{user_id}:{ticker}"
        existing = _LOCAL_TICKER_INTERESTS.get(local_key)
        if existing:
            memory_id = existing["memory_id"]
        else:
            memory_id = f"local-{ticker}-{len(_LOCAL_TICKER_INTERESTS) + 1}"
            _LOCAL_TICKER_INTERESTS[local_key] = {
                "ticker": ticker,
                "user_id": user_id,
                "memory_id": memory_id,
                "query": user_query,
                "source": source,
            }
        registered_to_memory = True
    # Register in user_data_store watchlist
    registered_to_watchlist = False
    try:
        from backend.user_data_store import settings_path, load_json, save_json
        path = settings_path(user_id)
        settings = load_json(path, default={})
        watchlist = settings.get("watchlist", [])
        if ticker not in watchlist:
            watchlist.append(ticker)
            settings["watchlist"] = watchlist
            save_json(path, settings)
        registered_to_watchlist = True
    except Exception as exc:
        error = (error or "") + f" user_data_store error: {exc}"
    return {
        "ticker": ticker,
        "registered_to_memory": registered_to_memory,
        "registered_to_watchlist": registered_to_watchlist,
        "memory_id": memory_id,
        "error": error
    }

def register_gnn_candidate(ticker: str, universe: list) -> Dict:
    # Check if ticker already in GNN universe
    try:
        from backend.gnn_model import GNNSignalEngine
        gnn = GNNSignalEngine.get()
        if gnn._covers_tickers([ticker]):
            return {
                "ticker": ticker,
                "already_in_gnn": True,
                "gnn_candidate_added": False,
                "gnn_signal_available": True,
                "requires_refresh": False,
                "error": None
            }
        # Add as candidate (if enough data exists)
        # (Assume candidate is added to a queue for retraining, not forced immediately)
        # Here, just return status; actual retraining is handled elsewhere
        return {
            "ticker": ticker,
            "already_in_gnn": False,
            "gnn_candidate_added": True,
            "gnn_signal_available": False,
            "requires_refresh": True,
            "error": None
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "already_in_gnn": False,
            "gnn_candidate_added": False,
            "gnn_signal_available": False,
            "requires_refresh": False,
            "error": f"gnn_model error: {exc}"
        }
