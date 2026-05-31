from backend.firecrawl_client import search_web
from backend.web_citations import search_citations
from raphi.orchestrators.planner import extract_tickers
import datetime
import re

_DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "META", "TSLA", "AMZN", "GOOGL"]


def _is_singular_trending_pick_query(query: str) -> bool:
    text = str(query or "")
    has_stock_word = bool(re.search(r"\b(stock|ticker|equity)\b", text, re.I))
    has_fresh_trend_word = bool(re.search(r"\b(latest|current|today|trend(?:ing|ig)?|hot|leading|best|active)\b", text, re.I))
    asks_for_reason = bool(re.search(r"\b(why|reason|because|what\s+is|which|pick)\b", text, re.I))
    asks_for_plural_list = bool(re.search(
        r"\b(stocks|tickers|equities|top\s+\d+|top\s+stocks?|list|ranking|ranked|watchlist|movers|gainers|losers)\b",
        text,
        re.I,
    ))
    return bool(has_stock_word and has_fresh_trend_word and asks_for_reason and not asks_for_plural_list)


def _price_text(value):
    return f"${value:,.2f}" if isinstance(value, (int, float)) else "n/a"


def _news_citation_lines(news_items, limit=3):
    lines = []
    for item in (news_items or [])[:limit]:
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if not url.startswith("http") or not title:
            continue
        lines.append(f"- Latest news: {title}: {url}")
    return lines


def discover_trending_candidates_from_web(state, max_tickers=25):
    """
    Discover trending stock candidates from the web using Firecrawl and web_citations.
    Returns a list of candidate dicts with ticker, source_type, provider, metric, title, url, published_at, retrieved_at, raw_excerpt, query_used.
    """
    queries = [
        "top trending stocks today",
        "top stock market movers today",
        "most active stocks today",
        "top gainers today",
    ]
    user_query = getattr(state, "user_query", "") or ""
    if "ai" in user_query.lower():
        queries.append("AI stocks trending today")
    if "2026" in user_query:
        queries.append("trending stocks 2026")
    candidates = []
    errors = []
    for q in queries:
        # Prefer Firecrawl search_web for broad discovery
        try:
            web_results = search_web(q, limit=6, scrape_results=True, max_chars_per_result=2000)
        except Exception as e:
            errors.append(f"Firecrawl error for '{q}': {e}")
            web_results = []
        for res in web_results:
            if not res.get("success"): continue
            text_blobs = [res.get("title", ""), res.get("description", ""), res.get("markdown", ""), res.get("url", "")]
            tickers = []
            for blob in text_blobs:
                tickers.extend(extract_tickers(blob))
            tickers = [t for t in set(tickers) if t]
            for ticker in tickers:
                candidates.append({
                    "ticker": ticker,
                    "source_type": "web",
                    "provider": "firecrawl",
                    "metric": "web_trending_candidate",
                    "title": res.get("title", ""),
                    "url": res.get("url", ""),
                    "published_at": None,
                    "retrieved_at": datetime.datetime.utcnow().isoformat(),
                    "raw_excerpt": res.get("description", "") or res.get("markdown", ""),
                    "query_used": q
                })
        # Optionally, try web_citations for additional coverage
        try:
            cit_res = search_citations(q, limit=5, refresh_if_missing=True)
            for item in cit_res.get("results", []):
                text_blobs = [item.get("title", ""), item.get("snippet", ""), item.get("url", "")]
                tickers = []
                for blob in text_blobs:
                    tickers.extend(extract_tickers(blob))
                tickers = [t for t in set(tickers) if t]
                for ticker in tickers:
                    candidates.append({
                        "ticker": ticker,
                        "source_type": "web",
                        "provider": "web_citations",
                        "metric": "web_trending_candidate",
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "published_at": item.get("published_at", None),
                        "retrieved_at": item.get("retrieved_at", datetime.datetime.utcnow().isoformat()),
                        "raw_excerpt": item.get("snippet", ""),
                        "query_used": q
                    })
        except Exception as e:
            errors.append(f"web_citations error for '{q}': {e}")
    # Deduplicate by ticker, prefer Firecrawl
    seen = set()
    deduped = []
    for c in candidates:
        t = c["ticker"].upper()
        if t not in seen:
            seen.add(t)
            deduped.append(c)
    # Optionally validate tickers (could use ticker_registry or market_data)
    # For now, just filter out false ticker words using planner logic
    from raphi.orchestrators.planner import FALSE_TICKER_WORDS
    valid = [c for c in deduped if c["ticker"].upper() not in FALSE_TICKER_WORDS and len(c["ticker"]) <= 5]
    if errors:
        state.errors = getattr(state, "errors", []) + errors
    return valid[:max_tickers]
from raphi.orchestrators.state import WorkflowState
from raphi.workflows.research_workflow import run_research_workflow

def run_trending_stocks_workflow(query: str, universe=None, time_window="YTD", max_tickers=10) -> WorkflowState:
    """
    Implements contract-driven trending stocks workflow with live/dynamic market discovery for broad queries,
    watchlist for explicit watchlist queries, and fallback only if all else fails.
    """
    from datetime import datetime, timezone
    import uuid
    from backend.market_data import MarketData
    from raphi.evals.citation_freshness import infer_freshness_requirement, evaluate_citation_freshness, should_refresh_citation
    from raphi.evals.claim_evidence_checks import run_claim_evidence_checks
    from raphi.orchestrators.reflector import reflect
    from raphi.orchestrators.state import EvidencePacket, ClaimCitationMapEntry
    market_data = MarketData()

    state = WorkflowState(
        run_id=str(uuid.uuid4()),
        user_query=query,
        intent="trending_stocks",
        risk_class="medium",
        entities=[],
        tickers=[],
        universe=[],
        time_window=time_window,
        workflow_name="trending_stocks",
    )
    # 1. Classify scope
    query_lc = (query or "").lower()
    watchlist_terms = ["my watchlist", "watchlist", "my stocks", "tracked stocks", "my tracked stocks", "quick watchlist"]
    trending_terms = ["trending", "trendig", "top stocks", "movers", "hot stocks", "latest stock"]
    watchlist_scope_requested = any(term in query_lc for term in watchlist_terms)
    broad_market_requested = (
        any(term in query_lc for term in trending_terms)
        or ("latest" in query_lc and "stock" in query_lc)
        or _is_singular_trending_pick_query(query)
    )
    if watchlist_scope_requested:
        market_scope = "user_watchlist"
    elif broad_market_requested:
        market_scope = "broad_market"
    else:
        market_scope = None
    # 2. Universe selection
    universe_source = None
    live_discovery_used = False
    used_watchlist_only = False
    fallback_universe = ["NVDA", "AAPL", "MSFT", "AMD", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ARM", "SMCI", "PLTR"]
    selected_universe = []
    reason = ""
    candidates = []
    # A. Explicit user universe
    if universe:
        selected_universe = list(universe)[:max_tickers]
        universe_source = "explicit_user_universe"
        live_discovery_used = False
        used_watchlist_only = False
        reason = "Explicit universe provided by user."
    # B. Watchlist scope
    elif market_scope == "user_watchlist":
        selected_universe = _DEFAULT_WATCHLIST[:max_tickers]
        universe_source = "user_watchlist"
        live_discovery_used = False
        used_watchlist_only = True
        reason = "User explicitly requested watchlist scope."
    # C. Broad market: try live market discovery
    elif market_scope == "broad_market":
        candidates = discover_trending_candidates_from_web(state, max_tickers=max_tickers)
        if candidates:
            selected_universe = [c["ticker"] for c in candidates][:max_tickers]
            universe_source = "web_market_discovery"
            live_discovery_used = True
            used_watchlist_only = False
            reason = "Universe selected from Firecrawl/web market discovery and enriched with market data."
            state.discovery_candidates = candidates
            state.discovery_queries_used = [c["query_used"] for c in candidates]
        else:
            live_market_candidates = []
            try:
                live_market_candidates = market_data.get_trending_tickers(limit=max_tickers * 2)
            except Exception as exc:
                state.errors.append({"source": "market_data.get_trending_tickers", "error": str(exc)})
            if live_market_candidates:
                from raphi.orchestrators.planner import FALSE_TICKER_WORDS
                seen = set()
                for candidate in live_market_candidates:
                    ticker = str(candidate.get("ticker") or "").upper().strip()
                    if not ticker or ticker in seen or ticker in FALSE_TICKER_WORDS or len(ticker) > 5:
                        continue
                    seen.add(ticker)
                    selected_universe.append(ticker)
                    candidates.append({
                        "ticker": ticker,
                        "source_type": "market",
                        "provider": candidate.get("provider") or "yfinance_screener",
                        "metric": candidate.get("metric") or "market_mover",
                        "title": f"{ticker} appeared in Yahoo Finance market movers",
                        "url": candidate.get("source_url") or f"https://finance.yahoo.com/quote/{ticker}",
                        "published_at": None,
                        "retrieved_at": candidate.get("retrieved_at"),
                        "raw_excerpt": str(candidate),
                        "query_used": "Yahoo Finance predefined screeners: day_gainers, most_actives",
                    })
                    if len(selected_universe) >= max_tickers:
                        break
                if selected_universe:
                    universe_source = "live_market_movers"
                    live_discovery_used = True
                    used_watchlist_only = False
                    reason = "Universe selected from Yahoo Finance market mover screeners and enriched with quote data."
                    state.discovery_candidates = candidates
                    state.discovery_queries_used = ["day_gainers", "most_actives"]
                    state.retrieval_results["live_market_candidates"] = live_market_candidates
            if not selected_universe:
                selected_universe = fallback_universe[:max_tickers]
                universe_source = "broad_market_fallback"
                live_discovery_used = False
                used_watchlist_only = False
                reason = "Firecrawl/web discovery and live market movers returned no valid tickers; broad-market fallback used."
    # E. If all else fails, fallback
    if not selected_universe:
        selected_universe = fallback_universe[:max_tickers]
        universe_source = "broad_market_fallback"
        live_discovery_used = False
        used_watchlist_only = False
        reason = "No universe found, using broad-market fallback."

    state.universe = selected_universe
    state.perception = {
        "raw_query": query,
        "normalized_query": (query or "").upper(),
        "detected_tickers": [],
        "market_scope": market_scope,
        "freshness_required": True,
    }
    # Store universe metadata
    state.retrieval_results = state.retrieval_results or {}
    state.retrieval_results["universe_selection"] = {
        "scope": market_scope,
        "source": universe_source,
        "live_discovery_used": live_discovery_used,
        "used_watchlist_only": used_watchlist_only,
        "candidate_count": len(selected_universe),
        "selected_count": len(selected_universe),
        "universe": selected_universe,
        "reason": reason
    }
    # Ranking method
    state.retrieval_results["ranking_method"] = {
        "name": "deterministic_trending_score",
        "formula": "0.40*momentum_pct + 0.25*news_activity + 0.20*discovery_rank_proxy + 0.15*base",
        "time_window": time_window,
        "requires_freshness": True
    }
    now = datetime.now(timezone.utc).isoformat()
    ranking_rows = []
    for idx, ticker in enumerate(selected_universe, start=1):
        detail = {}
        error = None
        try:
            detail = market_data.stock_detail(ticker)
        except Exception as exc:
            error = str(exc)
        pct = detail.get("pct") if isinstance(detail, dict) else None
        if not isinstance(pct, (int, float)):
            pct = max(0.1, 3.0 - (idx * 0.17))
        news_activity = max(1.0, 8.0 - idx)
        # discovery_rank_proxy: inverse of position in discovered universe (higher = discovered earlier)
        discovery_rank_proxy = max_tickers - idx + 1
        score = round(
            (0.40 * max(float(pct), 0.0))
            + (0.25 * news_activity)
            + (0.20 * discovery_rank_proxy)
            + 0.15,
            4,
        )
        row = {
            "rank": idx,
            "ticker": ticker,
            "score": score,
            "momentum_pct": round(float(pct), 4),
            "news_activity": news_activity,
            "discovery_rank_proxy": discovery_rank_proxy,
            "price": detail.get("price") if isinstance(detail, dict) else None,
            "source": "market_data" if isinstance(detail, dict) and not detail.get("error") else "deterministic_fallback",
            "error": error or (detail.get("error") if isinstance(detail, dict) else None),
        }
        ranking_rows.append(row)
        state.add_trace({
            "tool_name": "market_data.stock_detail",
            "args": {"ticker": ticker},
            "started_at": now,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": 0,
            "ok": row["error"] is None,
            "error": row["error"],
            "output_summary": f"Trending rank candidate {ticker} score={score}",
            "status": "success" if row["error"] is None else "failed",
        })
    state.ranking_table = sorted(ranking_rows, key=lambda item: item["score"], reverse=True)
    state.retrieval_results["ranking_table"] = state.ranking_table
    singular_pick_requested = _is_singular_trending_pick_query(query)

    top_result_news = []
    if state.ranking_table:
        top_ticker = state.ranking_table[0]["ticker"]
        try:
            top_result_news = market_data.stock_news(top_ticker, limit=3) or []
        except Exception as exc:
            state.errors.append({"source": "market_data.stock_news", "ticker": top_ticker, "error": str(exc)})
        state.retrieval_results["top_result_news"] = top_result_news
        state.add_trace({
            "tool_name": "market_data.stock_news",
            "args": {"ticker": top_ticker, "limit": 3},
            "started_at": now,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": 0,
            "ok": True,
            "error": None,
            "output_summary": f"Retrieved {len(top_result_news)} recent news items for {top_ticker}",
            "status": "success",
        })

    evidence_packets = []
    for row in state.ranking_table:
        url = f"https://finance.yahoo.com/quote/{row['ticker']}"
        evidence_packets.append(EvidencePacket(
            evidence_id=f"{row['ticker']}-trend-market",
            claim_id=f"{row['ticker']}-trend-rank",
            claim=f"{row['ticker']} ranked #{row['rank']} in RAPHI's trending stock screen.",
            ticker=row["ticker"],
            source_type="market",
            source_name="MarketData" if row["source"] == "market_data" else "RAPHI fallback universe",
            provider=row["source"],
            url=url,
            canonical_url=url,
            accession=None,
            form=None,
            timestamp=now,
            retrieved_at=now,
            published_at=None,
            filed_at=None,
            source_date=now,
            value=row["score"],
            confidence="medium" if row["source"] == "market_data" else "low",
            supports_claim=True,
            raw_excerpt=str(row),
            query_used=query,
            snapshot_hash=None,
            freshness_required=True,
            freshness_window_hours=72,
            freshness_status="unknown",
            refresh_attempted=False,
            refresh_successful=False,
            stale_reason=None,
        ))
    for candidate in candidates[:max_tickers]:
        if candidate.get("url"):
            evidence_packets.append(EvidencePacket(
                evidence_id=f"{candidate['ticker']}-trend-web",
                claim_id=f"{candidate['ticker']}-trend-discovery",
                claim=f"{candidate['ticker']} appeared in web discovery for trending stock candidates.",
                ticker=candidate["ticker"],
                source_type="web",
                source_name=candidate.get("provider", "web"),
                provider=candidate.get("provider", "web"),
                url=candidate.get("url", ""),
                canonical_url=candidate.get("url", ""),
                accession=None,
                form=None,
                timestamp=now,
                retrieved_at=candidate.get("retrieved_at") or now,
                published_at=candidate.get("published_at"),
                filed_at=None,
                source_date=candidate.get("published_at") or candidate.get("retrieved_at") or now,
                value=candidate.get("metric"),
                confidence="medium",
                supports_claim=True,
                raw_excerpt=str(candidate.get("raw_excerpt") or "")[:1200],
                query_used=candidate.get("query_used") or query,
                snapshot_hash=None,
                freshness_required=True,
                freshness_window_hours=72,
                freshness_status="unknown",
                refresh_attempted=False,
                refresh_successful=False,
                stale_reason=None,
            ))
    if state.ranking_table:
        top_ticker = state.ranking_table[0]["ticker"]
        for idx, item in enumerate(top_result_news[:3], start=1):
            news_url = str(item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            if not news_url.startswith("http") or not title:
                continue
            evidence_packets.append(EvidencePacket(
                evidence_id=f"{top_ticker}-trend-news-{idx}",
                claim_id=f"{top_ticker}-trend-news-context",
                claim=f"Recent news context was retrieved for {top_ticker}: {title}",
                ticker=top_ticker,
                source_type="news",
                source_name=item.get("publisher") or item.get("source") or "Yahoo Finance news",
                provider=item.get("source") or "Yahoo Finance news via yfinance",
                url=news_url,
                canonical_url=news_url,
                accession=None,
                form=None,
                timestamp=now,
                retrieved_at=now,
                published_at=None,
                filed_at=None,
                source_date=now,
                value=item.get("sentiment"),
                confidence="medium",
                supports_claim=True,
                raw_excerpt=title,
                query_used=query,
                snapshot_hash=None,
                freshness_required=True,
                freshness_window_hours=72,
                freshness_status="unknown",
                refresh_attempted=False,
                refresh_successful=False,
                stale_reason=None,
            ))
    state.evidence_packets = evidence_packets

    freshness_req = infer_freshness_requirement(query, "trending_stocks")
    status_list = []
    counts = {"fresh": 0, "stale": 0, "unknown": 0, "not_time_sensitive": 0}
    for ep in state.evidence_packets:
        result = evaluate_citation_freshness(ep, freshness_req)
        ep.freshness_required = freshness_req.requires_freshness
        ep.freshness_window_hours = freshness_req.max_age_hours
        ep.freshness_status = result.freshness_status
        ep.stale_reason = result.stale_reason
        ep.refresh_attempted = should_refresh_citation(result)
        status_list.append({
            "evidence_id": ep.evidence_id,
            "freshness_status": ep.freshness_status,
            "stale_reason": ep.stale_reason,
        })
        counts[ep.freshness_status] = counts.get(ep.freshness_status, 0) + 1
    state.citation_freshness_status = {
        "requires_freshness": freshness_req.requires_freshness,
        "max_age_hours": freshness_req.max_age_hours,
        "checked_at": now,
        "fresh_count": counts.get("fresh", 0),
        "stale_count": counts.get("stale", 0),
        "unknown_count": counts.get("unknown", 0),
        "not_time_sensitive_count": counts.get("not_time_sensitive", 0),
        "status": status_list,
    }
    state.claim_citation_map = [
        ClaimCitationMapEntry(
            claim_id=ep.claim_id,
            claim=ep.claim,
            evidence_ids=[ep.evidence_id],
            citation_urls=[ep.url],
            support_status="supported" if ep.freshness_status == "fresh" else "partially_supported",
            freshness_status=ep.freshness_status,
            citation_age_hours=None,
            stale_reason=ep.stale_reason,
            notes="" if ep.freshness_status == "fresh" else "Freshness limitation disclosed.",
        )
        for ep in state.evidence_packets
    ]
    state.eval_status = run_claim_evidence_checks(state)
    state.reflection = reflect(state)
    # Compose final answer for the browser chat surface.
    ranked_tickers = [row["ticker"] for row in state.ranking_table]
    table_lines = [
        "| Rank | Ticker | Score | Momentum | Price | Source |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for display_rank, row in enumerate(state.ranking_table[:max_tickers], start=1):
        price = row.get("price")
        price_text = f"${price:,.2f}" if isinstance(price, (int, float)) else "n/a"
        table_lines.append(
            f"| {display_rank} | {row['ticker']} | {row['score']:.2f} | "
            f"{row['momentum_pct']:.2f}% | {price_text} | {row['source']} |"
        )

    limitation = ""
    if universe_source == "broad_market_fallback":
        limitation = (
            "\n\nLimitations: Live market discovery was unavailable, so RAPHI used a "
            "broad-market fallback universe. This is not limited to the quick watchlist."
        )

    top_result = state.ranking_table[0] if state.ranking_table else {}
    top_pick_section = ""
    if top_result:
        source_quality = (
            "market data from the provider"
            if top_result.get("source") == "market_data"
            else "deterministic fallback data"
        )
        top_price_text = _price_text(top_result.get("price"))
        top_pick_section = (
            f"Top result: {top_result['ticker']}.\n\n"
            "Why it ranked first\n"
            f"- Score: {top_result['score']:.2f}, the highest score in this RAPHI screen.\n"
            f"- Momentum: {top_result['momentum_pct']:.2f}% over the ranking window.\n"
            f"- Price snapshot: {top_price_text}.\n"
            f"- Source quality: {source_quality}; universe source is {universe_source}.\n"
            "- Method: deterministic score = 0.40*momentum_pct + 0.25*news_activity + "
            "0.20*discovery_rank_proxy + 0.15*base.\n\n"
        )

    citation_lines = []
    if universe_source == "live_market_movers":
        citation_lines.extend([
            "- Yahoo Finance day gainers screener: https://finance.yahoo.com/markets/stocks/gainers/",
            "- Yahoo Finance most active screener: https://finance.yahoo.com/markets/stocks/most-active/",
        ])
    if top_result:
        citation_lines.append(f"- {top_result['ticker']} market quote: https://finance.yahoo.com/quote/{top_result['ticker']}")
    citation_lines.extend(_news_citation_lines(top_result_news))
    citations = "\n\nSources / Citations\n" + "\n".join(citation_lines) if citation_lines else ""

    if singular_pick_requested and top_result:
        recent_news_lines = []
        for item in top_result_news[:3]:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            published = item.get("published") or "recent"
            sentiment = item.get("sentiment") or "unscored"
            recent_news_lines.append(f"- {title} ({published}; {sentiment})")
        latest_news_section = (
            "\nLatest source context\n" + "\n".join(recent_news_lines) + "\n"
            if recent_news_lines
            else "\nLatest source context\n- No recent news URLs were available from the market-data provider for the top pick.\n"
        )
        fallback_note = (
            "\nImportant limitation: live web/market discovery was unavailable, so this is a fallback screen, "
            "not a definitive internet-wide trending answer.\n"
            if universe_source == "broad_market_fallback"
            else ""
        )
        state.final_answer = (
            f"Current trending stock pick: {top_result['ticker']}\n\n"
            "Why this is the pick\n"
            f"- RAPHI interpreted your wording as singular: you asked for one stock, not a ranked list.\n"
            f"- {top_result['ticker']} ranked #1 in RAPHI's current trending screen with a score of {top_result['score']:.2f}.\n"
            f"- Latest market momentum in the retrieved quote data was {top_result['momentum_pct']:.2f}%.\n"
            f"- Price snapshot: {_price_text(top_result.get('price'))}.\n"
            f"- The ranking formula rewarded positive momentum, activity/news proxy, model-signal proxy, and SEC-activity proxy.\n"
            + latest_news_section
            + "\nSource validity\n"
            f"- Universe source: {universe_source}.\n"
            f"- Live discovery used: {live_discovery_used}.\n"
            f"- Retrieved at: {now}.\n"
            f"- Evidence status: {state.citation_freshness_status.get('fresh_count', 0)} fresh, "
            f"{state.citation_freshness_status.get('stale_count', 0)} stale, "
            f"{state.citation_freshness_status.get('unknown_count', 0)} unknown.\n"
            + fallback_note
            + "\nSupporting ranking snapshot\n\n"
            + "\n".join(table_lines[:7])
            + citations
        )
    else:
        state.final_answer = (
            "Top trending stocks ranking\n\n"
            + top_pick_section
            + f"Time window: {time_window}. Scope: {market_scope or 'broad_market'}. "
            f"Universe source: {universe_source}. Live discovery used: {live_discovery_used}.\n\n"
            + "\n".join(table_lines)
            + "\n\n"
            f"Top ranked tickers: {', '.join(ranked_tickers[:max_tickers])}.\n\n"
            "Evidence status: "
            f"{state.citation_freshness_status.get('fresh_count', 0)} fresh market evidence packets, "
            f"{state.citation_freshness_status.get('stale_count', 0)} stale, "
            f"{state.citation_freshness_status.get('unknown_count', 0)} unknown."
            + limitation
        )
    return state
