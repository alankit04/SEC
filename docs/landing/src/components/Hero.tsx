"use client";
import { ArrowRight, TrendingUp, TrendingDown } from "lucide-react";

const TICKER = [
  { s: "NVDA",  p: "$188.63", c: "+2.41%", up: true  },
  { s: "AAPL",  p: "$207.19", c: "+0.83%", up: true  },
  { s: "MSFT",  p: "$399.12", c: "-0.22%", up: false },
  { s: "META",  p: "$512.04", c: "+1.67%", up: true  },
  { s: "TSLA",  p: "$172.83", c: "-1.34%", up: false },
  { s: "AMZN",  p: "$189.41", c: "+0.55%", up: true  },
  { s: "GOOGL", p: "$167.92", c: "+0.31%", up: true  },
  { s: "SP500", p: "5,218",   c: "+0.48%", up: true  },
];

const STATS = [
  { n: "112K+",  l: "SEC filings"    },
  { n: "9,460",  l: "Companies"      },
  { n: "16",     l: "Quarters live"  },
  { n: "<3s",    l: "Avg response"   },
];

const AGENTS = [
  { name: "@market-analyst",  color: "text-blue-400",   status: "live" },
  { name: "@sec-researcher",  color: "text-violet-400",  status: "live" },
  { name: "@ml-signals",      color: "text-indigo-400",  status: "live" },
  { name: "@portfolio-risk",  color: "text-cyan-400",    status: "live" },
];

export default function Hero() {
  return (
    <section className="relative min-h-screen flex flex-col overflow-hidden">

      {/* ── Aurora background ── */}
      <div className="absolute inset-0 bg-[#060609]" />
      <div className="aurora-orb w-[900px] h-[600px] top-[-200px] left-[-100px] bg-blue-600/20" />
      <div className="aurora-orb w-[700px] h-[500px] top-[-100px] right-[-200px] bg-violet-600/15" />
      <div className="aurora-orb w-[500px] h-[400px] bottom-[0px] left-[30%] bg-indigo-500/10" />
      <div className="absolute inset-0 bg-grid-pattern bg-grid-sm opacity-100" />
      {/* Vignette */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-10%,rgba(99,102,241,0.08),transparent)]" />

      {/* ── Main content ── */}
      <div className="relative z-10 flex-1 flex flex-col items-center justify-center
                      px-5 pt-28 pb-16 text-center">

        {/* Pill badge */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1 mb-8 rounded-full
                        glass text-[11px] font-semibold tracking-[0.14em] uppercase text-white/60">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          Institutional-grade AI · Running locally
        </div>

        {/* Headline */}
        <h1 className="max-w-4xl font-bold tracking-tight mb-6"
            style={{ fontSize: "clamp(2.8rem, 7vw, 5.5rem)", lineHeight: 1.05 }}>
          <span className="text-white">The AI platform for</span>
          <br />
          <span className="text-gradient-brand">investment intelligence</span>
        </h1>

        {/* Sub */}
        <p className="max-w-lg text-[17px] text-white/45 leading-relaxed mb-10 font-light">
          Real-time market data · 16 quarters of SEC EDGAR · XGBoost ML signals.
          Orchestrated by Claude agents. Institutional memos in seconds.
        </p>

        {/* CTA row */}
        <div className="flex items-center gap-3 mb-16">
          <a href="http://localhost:9999"
             className="inline-flex items-center gap-2 px-6 py-3 text-sm font-semibold text-white
                        rounded-xl bg-gradient-to-r from-blue-600 to-violet-600
                        hover:from-blue-500 hover:to-violet-500 transition-all duration-200
                        shadow-[0_0_40px_rgba(99,102,241,0.35)] hover:shadow-[0_0_56px_rgba(99,102,241,0.55)]
                        hover:-translate-y-0.5">
            Start Analysing
            <ArrowRight className="w-4 h-4" />
          </a>
          <a href="#how-it-works"
             className="inline-flex items-center gap-1.5 px-6 py-3 text-sm font-medium
                        text-white/50 hover:text-white border border-white/10 hover:border-white/20
                        rounded-xl transition-all hover:bg-white/[0.03]">
            See how it works
          </a>
        </div>

        {/* ── Bento hero cards ── */}
        <div className="w-full max-w-4xl grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          {STATS.map(({ n, l }) => (
            <div key={l} className="glass rounded-2xl p-5 text-left relative overflow-hidden">
              <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r
                              from-transparent via-white/10 to-transparent" />
              <p className="text-3xl font-black text-white tracking-tight mb-1">{n}</p>
              <p className="text-xs text-white/35 font-medium">{l}</p>
            </div>
          ))}
        </div>

        {/* ── Agent swarm live card ── */}
        <div className="w-full max-w-4xl glass rounded-2xl overflow-hidden relative">
          {/* Animated beam */}
          <div className="absolute top-0 inset-x-0 h-px overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-blue-400/60 to-transparent
                            animate-beam" />
          </div>

          {/* Header */}
          <div className="flex items-center justify-between px-5 py-3
                          border-b border-white/[0.06] bg-white/[0.015]">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-xs font-mono text-white/40 tracking-wide">RAPHI · memo-synthesizer</span>
            </div>
            <span className="text-[10px] font-mono text-white/20">claude-opus-4-5</span>
          </div>

          <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Left: prompt */}
            <div className="font-mono text-sm space-y-3">
              <div className="flex gap-3">
                <span className="text-white/20 text-xs mt-0.5 w-8 shrink-0">You</span>
                <span className="text-white/80">Write an investment memo for NVDA</span>
              </div>
              <div className="flex gap-3">
                <span className="text-violet-400 text-xs font-bold mt-0.5 w-8 shrink-0">AI</span>
                <div className="space-y-1.5 text-xs">
                  {AGENTS.map((a) => (
                    <div key={a.name} className="flex items-center gap-2 text-white/50">
                      <span className={`${a.color} font-medium`}>{a.name}</span>
                      <span className="text-white/20">→</span>
                      <span>fetching data</span>
                      <span className="w-1 h-1 rounded-full bg-green-400 animate-pulse ml-auto" />
                    </div>
                  ))}
                  <div className="flex items-center gap-2 mt-2 text-white/80">
                    <span>Generating institutional memo</span>
                    <span className="w-0.5 h-3 bg-blue-400 animate-pulse inline-block" />
                  </div>
                </div>
              </div>
            </div>

            {/* Right: memo preview */}
            <div className="rounded-xl bg-black/30 border border-white/[0.06] p-4
                            font-mono text-[11px] text-white/40 leading-relaxed">
              <p className="text-blue-400/80 mb-2 font-semibold">## Executive Summary</p>
              <p className="text-white/60 mb-1">NVDA: <span className="text-green-400">LONG</span> — 74% confidence</p>
              <p>Price: $188.63  Target: $240.00</p>
              <p>P/E: 47.2x  Revenue +89% YoY</p>
              <p className="mt-2 text-violet-400/70">## Bull Case</p>
              <p>Jensen Huang AI capex cycle dominance,</p>
              <p>H100/H200 order book sold out 2026…</p>
              <p className="mt-2 text-red-400/60">## Bear Case</p>
              <p>AMD MI300X gaining HPC share, TSMC…</p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Ticker tape ── */}
      <div className="relative z-10 border-t border-white/[0.05] bg-white/[0.01] py-3 overflow-hidden">
        <div className="flex animate-ticker whitespace-nowrap select-none">
          {[...TICKER, ...TICKER].map((t, i) => (
            <div key={i} className="inline-flex items-center gap-2 px-5 font-mono text-xs">
              <span className="text-white/40 font-semibold">{t.s}</span>
              <span className="text-white/65">{t.p}</span>
              <span className={t.up ? "text-green-400" : "text-red-400"}>{t.c}</span>
              <span className="text-white/10 ml-3">·</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
