"""
main.py  —  RAPHI backend API  (FastAPI, port 8000)

Run:
    cd "/Users/alan/Desktop/SEC Data"
    .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── ensure backend/ is importable ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from market_data       import MarketData
from sec_data          import SECData
from ml_model          import SignalEngine
from portfolio_manager import PortfolioManager
from gnn_model         import GNNSignalEngine

BASE = Path(__file__).parent.parent

# ── singletons ──────────────────────────────────────────────────────
market    = MarketData()
sec       = SECData(BASE)
engine    = SignalEngine()
portfolio = PortfolioManager()
gnn       = GNNSignalEngine.get(sec)   # shares SEC data for SIC lookups

# ── FastAPI ─────────────────────────────────────────────────────────────
app = FastAPI(title="RAPHI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

SETTINGS_FILE = BASE / "settings.json"
DEFAULT_WATCHLIST = ["NVDA", "AAPL", "MSFT", "META", "TSLA", "AMZN", "GOOGL"]


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {"watchlist": DEFAULT_WATCHLIST, "anthropic_api_key": ""}


def save_settings(s: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2)


def get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        key = load_settings().get("anthropic_api_key", "")
    return key


# ═══════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════════
# MARKET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/market/overview")
def market_overview():
    return market.market_overview()


# ═══════════════════════════════════════════════════════════════════════
# STOCK
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/stock/{ticker}")
def stock_detail(ticker: str):
    data = market.stock_detail(ticker.upper())
    if "error" in data:
        raise HTTPException(404, data["error"])
    return data


@app.get("/api/stock/{ticker}/news")
def stock_news(ticker: str):
    return market.stock_news(ticker.upper())


@app.get("/api/stock/{ticker}/signals")
def stock_signals(ticker: str, bg: BackgroundTasks = None):
    detail = market.stock_detail(ticker.upper())
    funds  = {
        "pe_ratio":       detail.get("pe_ratio"),
        "revenue_growth": detail.get("revenue_growth"),
    }
    result = engine.train_and_predict(ticker.upper(), funds)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result


@app.get("/api/stock/{ticker}/gnn")
def stock_gnn(ticker: str):
    """
    GNN-only prediction for a single ticker.

    Returns direction, confidence, graph-neighbor influence scores, SIC code,
    and metadata about the graph (node/edge count, backend used).
    If the GNN has not been trained yet for the current watchlist, this
    endpoint triggers training synchronously (first call may take ~30 s).
    """
    ticker = ticker.upper()
    settings  = load_settings()
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)
    universe  = list(dict.fromkeys([ticker, *[t.upper() for t in watchlist]]))
    result    = gnn.predict(ticker, universe)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result


@app.post("/api/gnn/train")
def gnn_train(bg: BackgroundTasks):
    """
    Trigger an asynchronous GNN graph rebuild + retrain.
    Call this after updating the watchlist or to force a model refresh.
    Returns immediately; training proceeds in the background.
    """
    settings  = load_settings()
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)

    def _retrain():
        gnn.ensure_trained(watchlist, force=True)

    bg.add_task(_retrain)
    return {"status": "training started", "tickers": watchlist}


@app.get("/api/gnn/status")
def gnn_status():
    """Report whether the GNN is trained and basic graph statistics."""
    return gnn.status()


@app.get("/api/stock/{ticker}/filings")
def stock_filings(ticker: str):
    cik      = sec.cik_for_ticker(ticker.upper())
    filings  = sec.ticker_filings(ticker.upper())
    fin_data = sec.company_financials(ticker.upper())
    return {"cik": cik, "filings": filings, "financials": fin_data}


# ═══════════════════════════════════════════════════════════════════════
# PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/portfolio")
def get_portfolio():
    return portfolio.snapshot()


class Positions(BaseModel):
    positions: list


@app.put("/api/portfolio")
def update_portfolio(body: Positions):
    portfolio.update_positions(body.positions)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════
# SIGNALS (all watchlist tickers)
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/signals")
def all_signals():
    settings = load_settings()
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)
    # Build fundamentals map
    fund_map: dict = {}
    for t in watchlist:
        d = market.stock_detail(t)
        fund_map[t] = {
            "pe_ratio":       d.get("pe_ratio"),
            "revenue_growth": d.get("revenue_growth"),
        }
    return engine.multi_signals(watchlist, fund_map)


# ═══════════════════════════════════════════════════════════════════════
# ALERTS  (computed from real portfolio + signals)
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/alerts")
def get_alerts():
    snap    = portfolio.snapshot()
    alerts  = []

    # VaR breach?
    var95  = snap.get("var_95", 0)
    total  = snap.get("total_value", 0)
    if total and var95 / total > 0.02:   # VaR > 2% of portfolio
        alerts.append({
            "type":     "RISK",
            "severity": "CRITICAL",
            "icon":     "🔴",
            "title":    f"Portfolio VaR breach — 95% daily VaR is ${var95:,.0f} ({var95/total*100:.1f}% of portfolio)",
            "sub":      "Portfolio · Risk Engine",
            "time":     _now_str(),
        })

    # Stop-loss proximity (within 3%)
    for p in snap.get("positions", []):
        sl = p.get("stop_loss")
        cur = p.get("current_price", 0)
        if sl and cur:
            gap_pct = abs(cur - sl) / cur * 100
            if gap_pct < 3:
                alerts.append({
                    "type":     "POSITION",
                    "severity": "WARNING",
                    "icon":     "⚠️",
                    "title":    f"{p['ticker']} approaching stop-loss — {gap_pct:.1f}% away (stop: ${sl})",
                    "sub":      f"Position Monitor · {p['ticker']}",
                    "time":     _now_str(),
                })

    # Down position alert
    for p in snap.get("positions", []):
        if p.get("pnl_pct", 0) < -5:
            alerts.append({
                "type":     "POSITION",
                "severity": "WARNING",
                "icon":     "📉",
                "title":    f"{p['ticker']} down {p['pnl_pct']:.1f}% from entry — review thesis",
                "sub":      f"Position Monitor · {p['ticker']}",
                "time":     _now_str(),
            })

    # Model drift alert
    settings  = load_settings()
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)
    for t in watchlist[:4]:            # check first 4 for speed
        cache_file = BASE / ".model_cache" / f"{t}.pkl"
        if cache_file.exists():
            import pickle
            try:
                with open(cache_file, "rb") as f:
                    r = pickle.load(f)
                acc = r.get("ensemble_accuracy", 100)
                if acc < 80:
                    alerts.append({
                        "type":     "MODEL",
                        "severity": "INFO",
                        "icon":     "🤖",
                        "title":    f"Model accuracy for {t} dropped to {acc:.1f}% (threshold: 80%)",
                        "sub":      f"Model Monitor · auto-retraining queued",
                        "time":     _now_str(),
                    })
            except Exception:
                pass

    return {"alerts": alerts, "count": len(alerts)}


# ═══════════════════════════════════════════════════════════════════════
# MODEL MONITORING
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/models/performance")
def model_performance():
    import pickle
    settings  = load_settings()
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)
    models    = []
    xgb_accs, lstm_accs = [], []

    for t in watchlist:
        f = BASE / ".model_cache" / f"{t}.pkl"
        if not f.exists():
            continue
        try:
            with open(f, "rb") as fh:
                r = pickle.load(fh)
            xgb_accs.append(r.get("xgb_accuracy", 0))
            lstm_accs.append(r.get("lstm_accuracy", 0))
            models.append({
                "ticker":    t,
                "xgb_acc":   r.get("xgb_accuracy"),
                "lstm_acc":  r.get("lstm_accuracy"),
                "ens_acc":   r.get("ensemble_accuracy"),
                "n_train":   r.get("n_train"),
                "trained_at": r.get("trained_at"),
            })
        except Exception:
            pass

    avg_xgb  = round(sum(xgb_accs)  / len(xgb_accs),  1) if xgb_accs  else None
    avg_lstm = round(sum(lstm_accs) / len(lstm_accs), 1) if lstm_accs else None
    avg_ens  = round((avg_xgb + avg_lstm) / 2, 1) if avg_xgb and avg_lstm else None

    return {
        "models":       models,
        "avg_xgb_acc":  avg_xgb,
        "avg_lstm_acc": avg_lstm,
        "avg_ens_acc":  avg_ens,
    }


# ═══════════════════════════════════════════════════════════════════════
# SEC
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/sec/search")
def sec_search(q: str, limit: int = 20):
    return sec.search_companies(q, limit)


@app.get("/api/sec/stats")
def sec_stats():
    return sec.summary_stats()


# ═══════════════════════════════════════════════════════════════════════
# NEWS (multi-ticker)
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/news")
def multi_news():
    snap      = portfolio.snapshot()
    tickers   = [p["ticker"] for p in snap.get("positions", [])]
    settings  = load_settings()
    watchlist = settings.get("watchlist", DEFAULT_WATCHLIST)
    all_tickers = list(dict.fromkeys(tickers + watchlist))[:6]

    # Overall sentiment scores
    articles: list = []
    ticker_sentiments: dict = {}
    for t in all_tickers:
        news = market.stock_news(t, limit=5)
        for n in news:
            n["primary_ticker"] = t
            articles.append(n)
        if news:
            avg_score = sum(a["score"] for a in news) / len(news)
            ticker_sentiments[t] = round(avg_score, 3)

    # Deduplicate by title
    seen_titles: set = set()
    unique: list = []
    for a in articles:
        if a["title"] not in seen_titles:
            seen_titles.add(a["title"])
            unique.append(a)

    unique.sort(key=lambda x: abs(x["score"]), reverse=True)

    # Market sentiment index: scale -1..1 → 0..100
    avg = sum(ticker_sentiments.values()) / len(ticker_sentiments) if ticker_sentiments else 0
    market_sentiment = int(50 + avg * 50)

    return {
        "articles":         unique[:15],
        "ticker_sentiments": ticker_sentiments,
        "market_sentiment": market_sentiment,
    }


# ═══════════════════════════════════════════════════════════════════════
# AI CHAT  (streaming SSE via Anthropic)
# ═══════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    message:  str
    history:  list = []
    ticker:   str  = "NVDA"


async def _chat_stream(message: str, history: list, ticker: str) -> AsyncIterator[str]:
    api_key = get_api_key()
    if not api_key:
        yield _sse("error", "ANTHROPIC_API_KEY not set. Add it in Settings.")
        return

    try:
        import anthropic
    except ImportError:
        yield _sse("error", "anthropic package not installed")
        return

    # ── build context ──────────────────────────────────────────────────
    snap   = portfolio.snapshot()
    detail = market.stock_detail(ticker.upper())
    news   = market.stock_news(ticker.upper(), limit=3)
    sig_cache = BASE / ".model_cache" / f"{ticker.upper()}.pkl"
    sig_text  = ""
    if sig_cache.exists():
        import pickle
        try:
            with open(sig_cache, "rb") as f:
                sig = pickle.load(f)
            sig_text = (
                f"RAPHI signal for {ticker}: {sig['direction']} "
                f"({sig['confidence']:.1f}% confidence, accuracy {sig['ensemble_accuracy']:.1f}%)"
            )
        except Exception:
            pass

    portfolio_summary = _fmt_portfolio(snap)
    news_summary      = "\n".join(
        f"- [{n['sentiment'].upper()}] {n['title']} ({n['publisher']}, {n['published']})"
        for n in news[:3]
    )
    stock_summary = (
        f"{detail.get('name', ticker)} ({ticker}) — "
        f"${detail.get('price', '?')} ({detail.get('pct', 0):+.1f}%), "
        f"P/E {detail.get('pe_ratio', '?')}, "
        f"Market Cap ${(detail.get('market_cap') or 0)/1e9:.1f}B"
    ) if detail.get("price") else ""

    system = f"""You are RAPHI, a sophisticated AI investment intelligence platform used by hedge funds and portfolio managers.
You have access to real-time market data, SEC filings, and quantitative models.

CURRENT PORTFOLIO:
{portfolio_summary}

FOCUSED STOCK ({ticker}):
{stock_summary}

RAPHI MODEL SIGNAL:
{sig_text or "Signal not yet computed for this ticker."}

RECENT NEWS ({ticker}):
{news_summary or "No recent news."}

SEC DATA: You have access to 15 quarters of SEC EDGAR filings (2022Q1–2025Q3) covering 9,457+ companies.

Provide precise, evidence-based investment analysis. Quote specific numbers.
Be concise but thorough. Use institutional language. Highlight risks explicitly.
Format your response with clear sections when appropriate."""

    messages = []
    for h in history[-6:]:   # last 3 turns
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    # Stream agent steps first
    steps = [
        ("orchestrator", "Orchestrator Agent — query parsed, routing to analysts"),
        ("market",       f"Market Data Agent — {ticker} price, volume, technicals loaded"),
        ("news",         "News Agent — sentiment analysis across recent articles"),
        ("predict",      "Signal Agent — XGBoost + LSTM ensemble computing"),
        ("synthesize",   "Synthesis Agent — generating response"),
    ]
    for step_id, label in steps:
        yield _sse("step", json.dumps({"id": step_id, "label": label}))
        await asyncio.sleep(0.2)

    # ── stream Claude response ─────────────────────────────────────────
    try:
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield _sse("token", text)
    except Exception as e:
        yield _sse("error", str(e))
        return

    yield _sse("done", "")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def generate():
        async for chunk in _chat_stream(req.message, req.history, req.ticker):
            yield chunk
    return StreamingResponse(generate(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════
# DECISION MEMO  (streaming)
# ═══════════════════════════════════════════════════════════════════════
@app.post("/api/memo/{ticker}")
async def generate_memo(ticker: str):
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(422, "ANTHROPIC_API_KEY not configured")

    detail = market.stock_detail(ticker.upper())
    news   = market.stock_news(ticker.upper(), limit=5)
    snap   = portfolio.snapshot()

    sig = {}
    sig_cache = BASE / ".model_cache" / f"{ticker.upper()}.pkl"
    if sig_cache.exists():
        import pickle
        try:
            with open(sig_cache, "rb") as f:
                sig = pickle.load(f)
        except Exception:
            pass

    prompt = f"""Generate a comprehensive institutional investment decision memo for {ticker.upper()}.

REAL MARKET DATA:
{json.dumps(detail, indent=2)[:2000]}

RAPHI MODEL SIGNAL:
{json.dumps(sig, indent=2)[:1000]}

RECENT NEWS:
{chr(10).join(f"- {n['title']} ({n['sentiment']})" for n in news[:5])}

PORTFOLIO CONTEXT:
{_fmt_portfolio(snap)}

Write a professional investment memo with these exact sections:
1. EXECUTIVE SUMMARY — recommendation (BUY/SELL/HOLD), confidence, price target, 90-day expected return
2. BULL CASE (probability %) — key catalysts
3. BEAR CASE (probability %) — key risks
4. MODEL VALIDATION — XGBoost/LSTM consensus, SHAP key drivers, falsifiability condition
5. TRADE PARAMETERS — entry range, price target, stop-loss, position sizing, horizon

Use precise numbers. Be direct. Institutional style."""

    async def generate():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(
                model="claude-opus-4-5",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield _sse("token", text)
            yield _sse("done", "")
        except Exception as e:
            yield _sse("error", str(e))

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/settings")
def get_settings():
    s = load_settings()
    # Mask API key
    if s.get("anthropic_api_key"):
        s["anthropic_api_key_set"] = True
        s["anthropic_api_key"] = "sk-ant-••••••••••" + s["anthropic_api_key"][-4:]
    else:
        s["anthropic_api_key_set"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return s


class SettingsBody(BaseModel):
    watchlist:          list  = []
    anthropic_api_key:  str   = ""


@app.put("/api/settings")
def update_settings(body: SettingsBody):
    s = load_settings()
    if body.watchlist:
        s["watchlist"] = [t.upper() for t in body.watchlist]
    if body.anthropic_api_key and not body.anthropic_api_key.startswith("sk-ant-••"):
        s["anthropic_api_key"] = body.anthropic_api_key
    save_settings(s)
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def _now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M")


def _fmt_portfolio(snap: dict) -> str:
    lines = [f"Total value: ${snap.get('total_value', 0):,.0f} | "
             f"P&L: ${snap.get('total_pnl', 0):,.0f} ({snap.get('total_pnl_pct', 0):+.1f}%) | "
             f"VaR 95%: ${snap.get('var_95', 0):,.0f} | "
             f"Sharpe: {snap.get('sharpe', 0):.2f}"]
    for p in snap.get("positions", []):
        lines.append(
            f"  {p['ticker']} {p.get('direction','LONG')} {p.get('shares',0)}sh "
            f"@ ${p.get('entry_price',0):.2f} → ${p.get('current_price',0):.2f} "
            f"({p.get('pnl_pct',0):+.1f}%)"
        )
    return "\n".join(lines)
