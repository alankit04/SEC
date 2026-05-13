import { TrendingUp, FileText, Brain, Shield, BarChart3, MessageSquare, Zap, Database } from "lucide-react";

export default function Features() {
  return (
    <section id="features" className="relative py-28 px-5 overflow-hidden">
      <div className="aurora-orb w-[600px] h-[400px] top-[10%] right-[-10%] bg-violet-600/10" />

      <div className="relative max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-14">
          <p className="label mb-3">Platform Capabilities</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-white tracking-tight max-w-xl">
            Eight systems.<br />
            <span className="text-gradient-brand">One platform.</span>
          </h2>
        </div>

        {/* ── Bento grid ── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 auto-rows-auto">

          {/* Hero card — AI Agent Swarm */}
          <div className="md:row-span-2 glass glass-hover rounded-2xl p-7 flex flex-col
                          relative overflow-hidden group">
            <div className="aurora-orb w-[300px] h-[200px] bottom-[-50px] left-[-50px]
                            bg-violet-600/15 group-hover:bg-violet-600/25 transition-all duration-700" />
            <div className="relative z-10 flex-1 flex flex-col">
              <div className="w-10 h-10 rounded-xl bg-violet-500/15 border border-violet-500/25
                              flex items-center justify-center mb-5">
                <MessageSquare className="w-5 h-5 text-violet-400" />
              </div>
              <p className="text-[11px] font-semibold tracking-widest uppercase text-violet-400/70 mb-2">
                Core Engine
              </p>
              <h3 className="text-xl font-bold text-white mb-3">AI Agent Swarm</h3>
              <p className="text-sm text-white/45 leading-relaxed mb-6">
                6 specialized Claude agents orchestrated by memo-synthesizer. Market analyst, SEC researcher,
                ML signals, and portfolio risk run in parallel via the Task tool — returning in seconds.
              </p>
              {/* Agent pills */}
              <div className="mt-auto space-y-2">
                {[
                  { n: "market-analyst",  c: "bg-blue-500/10 text-blue-300 border-blue-500/20"   },
                  { n: "sec-researcher",  c: "bg-violet-500/10 text-violet-300 border-violet-500/20" },
                  { n: "ml-signals",      c: "bg-indigo-500/10 text-indigo-300 border-indigo-500/20" },
                  { n: "portfolio-risk",  c: "bg-cyan-500/10 text-cyan-300 border-cyan-500/20"    },
                ].map(({ n, c }) => (
                  <div key={n} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg
                                           border text-xs font-mono ${c}`}>
                    <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                    @{n}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Real-Time Market Data */}
          <div className="glass glass-hover rounded-2xl p-6 relative overflow-hidden group">
            <div className="aurora-orb w-[200px] h-[150px] bottom-[-30px] right-[-30px]
                            bg-blue-600/10 group-hover:bg-blue-600/20 transition-all duration-700" />
            <div className="relative z-10">
              <div className="w-9 h-9 rounded-xl bg-blue-500/15 border border-blue-500/25
                              flex items-center justify-center mb-4">
                <TrendingUp className="w-4 h-4 text-blue-400" />
              </div>
              <h3 className="text-base font-bold text-white mb-2">Real-Time Market Data</h3>
              <p className="text-sm text-white/40 leading-relaxed">
                Live prices, fundamentals, P/E, market cap, VADER-scored news sentiment.
                Updated every 60s via yfinance + Finnhub.
              </p>
            </div>
          </div>

          {/* SEC EDGAR */}
          <div className="glass glass-hover rounded-2xl p-6 relative overflow-hidden group">
            <div className="aurora-orb w-[200px] h-[150px] bottom-[-30px] right-[-30px]
                            bg-indigo-600/10 group-hover:bg-indigo-600/20 transition-all duration-700" />
            <div className="relative z-10">
              <div className="w-9 h-9 rounded-xl bg-indigo-500/15 border border-indigo-500/25
                              flex items-center justify-center mb-4">
                <FileText className="w-4 h-4 text-indigo-400" />
              </div>
              <h3 className="text-base font-bold text-white mb-2">SEC EDGAR Analysis</h3>
              <p className="text-sm text-white/40 leading-relaxed">
                Local XBRL: 16 quarters · 112K filings · 9.5K companies · ~55M data points.
                No rate limits.
              </p>
            </div>
          </div>

          {/* ML Signals */}
          <div className="glass glass-hover rounded-2xl p-6 relative overflow-hidden group">
            <div className="aurora-orb w-[200px] h-[150px] bottom-[-30px] right-[-30px]
                            bg-violet-600/10 group-hover:bg-violet-600/20 transition-all duration-700" />
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <div className="w-9 h-9 rounded-xl bg-violet-500/15 border border-violet-500/25
                                flex items-center justify-center">
                  <Brain className="w-4 h-4 text-violet-400" />
                </div>
                <span className="text-[10px] text-green-400 font-mono bg-green-400/10
                                 border border-green-400/20 px-2 py-0.5 rounded-full">
                  74% confidence
                </span>
              </div>
              <h3 className="text-base font-bold text-white mb-2">ML Trading Signals</h3>
              <p className="text-sm text-white/40 leading-relaxed">
                XGBoost + GradBoost ensemble, 12 features, SHAP explainability.
              </p>
            </div>
          </div>

          {/* Portfolio Risk */}
          <div className="glass glass-hover rounded-2xl p-6 relative overflow-hidden group">
            <div className="aurora-orb w-[200px] h-[150px] bottom-[-30px] right-[-30px]
                            bg-cyan-600/08 group-hover:bg-cyan-600/15 transition-all duration-700" />
            <div className="relative z-10">
              <div className="w-9 h-9 rounded-xl bg-cyan-500/15 border border-cyan-500/25
                              flex items-center justify-center mb-4">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
              </div>
              <h3 className="text-base font-bold text-white mb-2">Portfolio Risk Engine</h3>
              <p className="text-sm text-white/40 leading-relaxed">
                VaR 95/99%, Sharpe, alpha vs SPY, stop-loss alerts, concentration warnings.
              </p>
            </div>
          </div>

          {/* Bottom row: Security + Conviction + MCP */}
          <div className="md:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-4">

            <div className="glass glass-hover rounded-2xl p-6 flex items-start gap-4">
              <div className="w-9 h-9 rounded-xl bg-green-500/15 border border-green-500/25
                              flex items-center justify-center shrink-0">
                <Shield className="w-4 h-4 text-green-400" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white mb-1.5">Enterprise Security</h3>
                <p className="text-xs text-white/35 leading-relaxed">
                  TokenAuth · Prompt injection guard · Fernet encryption · Rate limiting · Sentry
                </p>
              </div>
            </div>

            <div className="glass glass-hover rounded-2xl p-6 flex items-start gap-4">
              <div className="w-9 h-9 rounded-xl bg-orange-500/15 border border-orange-500/25
                              flex items-center justify-center shrink-0">
                <Database className="w-4 h-4 text-orange-400" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white mb-1.5">Conviction Ledger</h3>
                <p className="text-xs text-white/35 leading-relaxed">
                  Append-only JSONL · 30/60/90d resolution windows · Accuracy tracking
                </p>
              </div>
            </div>

            <div className="glass glass-hover rounded-2xl p-6 flex items-start gap-4">
              <div className="w-9 h-9 rounded-xl bg-yellow-500/15 border border-yellow-500/25
                              flex items-center justify-center shrink-0">
                <Zap className="w-4 h-4 text-yellow-400" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white mb-1.5">MCP Tool Bridge</h3>
                <p className="text-xs text-white/35 leading-relaxed">
                  8 tools · stdio transport · Port 9999 · Internal token auth
                </p>
              </div>
            </div>

          </div>
        </div>
      </div>
    </section>
  );
}
