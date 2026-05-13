import { MessageCircle, Workflow, FileText, ArrowRight } from "lucide-react";

const STEPS = [
  {
    num: "01",
    icon: MessageCircle,
    title: "Ask anything",
    lead: "Natural-language query",
    body: "Type an investment question. RAPHI parses intent — memo, screen, risk check, filing lookup — and routes it to the right agent.",
    code: [
      { c: "POST", t: "text-violet-400" },
      { c: " /api/chat", t: "text-white/70" },
      { c: "\n\n{\n  ", t: "text-white/30" },
      { c: '"message"', t: "text-blue-400" },
      { c: ": ", t: "text-white/30" },
      { c: '"Memo for NVDA"', t: "text-green-400" },
      { c: "\n}", t: "text-white/30" },
    ],
    gradient: "from-blue-500/20 to-blue-500/0",
    accent: "text-blue-400 border-blue-500/25 bg-blue-500/10",
  },
  {
    num: "02",
    icon: Workflow,
    title: "Agents dispatch",
    lead: "Parallel orchestration",
    body: "The memo-synthesizer fans out to 4 specialists via the Task tool — all firing at once. Each calls MCP tools that proxy to local FastAPI endpoints.",
    code: [
      { c: "@market-analyst",  t: "text-blue-400"   },
      { c: "   → price, PE, news\n", t: "text-white/50" },
      { c: "@sec-researcher",  t: "text-violet-400" },
      { c: "   → 15Q revenue\n",   t: "text-white/50" },
      { c: "@ml-signals",      t: "text-indigo-400" },
      { c: "       → XGBoost sig\n",t: "text-white/50" },
      { c: "@portfolio-risk",  t: "text-cyan-400"   },
      { c: "   → VaR, Sharpe",     t: "text-white/50" },
    ],
    gradient: "from-violet-500/20 to-violet-500/0",
    accent: "text-violet-400 border-violet-500/25 bg-violet-500/10",
  },
  {
    num: "03",
    icon: FileText,
    title: "Memo arrives",
    lead: "Streamed, formatted, cited",
    body: "The investment-memo skill composes 5 sections — Summary, Bull, Bear, Trade Parameters, Risk — and streams them token by token. Every number is sourced.",
    code: [
      { c: "## Executive Summary\n", t: "text-blue-400 font-semibold" },
      { c: "NVDA: ", t: "text-white/70" },
      { c: "LONG", t: "text-green-400 font-semibold" },
      { c: " 74% confidence\n", t: "text-white/70" },
      { c: "Target: ", t: "text-white/50" },
      { c: "$240", t: "text-white/90" },
      { c: " (+27% upside)\n\n", t: "text-white/50" },
      { c: "## Bull Case", t: "text-violet-400 font-semibold" },
      { c: "  …", t: "text-white/30" },
    ],
    gradient: "from-green-500/20 to-green-500/0",
    accent: "text-green-400 border-green-500/25 bg-green-500/10",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="relative py-32 px-5 overflow-hidden">

      {/* Background */}
      <div className="aurora-orb w-[700px] h-[500px] top-[20%] left-[-200px] bg-blue-600/10" />
      <div className="aurora-orb w-[600px] h-[400px] bottom-[10%] right-[-150px] bg-violet-600/08" />

      <div className="relative max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-16 max-w-2xl">
          <p className="overline mb-3">How RAPHI works</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-white tracking-tight leading-[1.05] mb-5">
            From question to{" "}
            <span className="text-gradient-brand">institutional memo</span>
            <br />in under three seconds.
          </h2>
          <p className="text-white/45 text-lg leading-relaxed font-light">
            A three-step flow powered by the Claude Agent SDK, MCP protocol, and a
            swarm of specialist subagents running on your machine.
          </p>
        </div>

        {/* ── 3-step cards ── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-10">
          {STEPS.map((s) => {
            const Icon = s.icon;
            return (
              <div key={s.num}
                   className="relative card-base card-hover p-7 flex flex-col overflow-hidden group">

                {/* Gradient wash */}
                <div className={`absolute top-0 inset-x-0 h-40 bg-gradient-to-b ${s.gradient}
                                opacity-60 group-hover:opacity-100 transition-opacity duration-500
                                pointer-events-none`} />

                <div className="relative z-10 flex flex-col flex-1">
                  {/* Step number */}
                  <div className="flex items-start justify-between mb-6">
                    <span className="text-[11px] font-mono font-semibold tracking-[0.2em]
                                     text-white/25">
                      STEP {s.num}
                    </span>
                    <div className={`w-10 h-10 rounded-xl border flex items-center justify-center ${s.accent}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                  </div>

                  {/* Title */}
                  <p className="text-[11px] font-semibold tracking-[0.14em] uppercase
                                text-white/35 mb-1.5">
                    {s.lead}
                  </p>
                  <h3 className="text-xl font-bold text-white mb-3 tracking-tight">
                    {s.title}
                  </h3>
                  <p className="text-[13.5px] text-white/50 leading-relaxed mb-6 flex-1">
                    {s.body}
                  </p>

                  {/* Code preview */}
                  <div className="rounded-xl bg-black/40 border border-white/[0.05] p-4
                                  font-mono text-[11px] leading-relaxed whitespace-pre overflow-hidden">
                    {s.code.map((seg, i) => (
                      <span key={i} className={seg.t}>{seg.c}</span>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Flow diagram ── */}
        <div className="card-base p-8 relative overflow-hidden">
          <div className="aurora-orb w-[400px] h-[200px] top-[-50px] left-[50%] -translate-x-1/2 bg-blue-600/10" />

          <div className="relative z-10">
            <div className="flex items-center justify-between mb-8">
              <div>
                <p className="overline mb-1.5">Under the hood</p>
                <h3 className="text-base font-bold text-white">Agent orchestration</h3>
              </div>
              <div className="chip bg-green-500/10 border border-green-500/20 text-green-400">
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Live
              </div>
            </div>

            {/* Pipeline */}
            <div className="flex flex-col md:flex-row items-center gap-3 md:gap-2 font-mono text-xs">

              <div className="flex-shrink-0 px-4 py-2.5 rounded-lg border border-white/15
                              bg-white/[0.03] text-white/75 min-w-[140px] text-center">
                User Question
              </div>

              <ArrowRight className="w-4 h-4 text-white/20 rotate-90 md:rotate-0" />

              <div className="flex-shrink-0 px-4 py-2.5 rounded-lg border border-blue-500/35
                              bg-blue-500/10 text-blue-300 font-semibold min-w-[180px] text-center">
                a2a_executor · Claude SDK
              </div>

              <ArrowRight className="w-4 h-4 text-white/20 rotate-90 md:rotate-0" />

              <div className="flex-shrink-0 px-4 py-2.5 rounded-lg border border-violet-500/35
                              bg-violet-500/10 text-violet-300 font-semibold min-w-[160px] text-center">
                memo-synthesizer
              </div>

              <ArrowRight className="w-4 h-4 text-white/20 rotate-90 md:rotate-0" />

              {/* Fan-out grid */}
              <div className="flex-1 grid grid-cols-2 gap-1.5 min-w-[200px]">
                {[
                  { n: "market-analyst",  c: "border-blue-500/25 bg-blue-500/10 text-blue-300"     },
                  { n: "sec-researcher",  c: "border-indigo-500/25 bg-indigo-500/10 text-indigo-300" },
                  { n: "ml-signals",      c: "border-violet-500/25 bg-violet-500/10 text-violet-300" },
                  { n: "portfolio-risk",  c: "border-cyan-500/25 bg-cyan-500/10 text-cyan-300"       },
                ].map(a => (
                  <div key={a.n} className={`px-3 py-1.5 rounded-md border text-center text-[11px] ${a.c}`}>
                    @{a.n}
                  </div>
                ))}
              </div>

              <ArrowRight className="w-4 h-4 text-white/20 rotate-90 md:rotate-0" />

              <div className="flex-shrink-0 px-4 py-2.5 rounded-lg border border-green-500/35
                              bg-green-500/10 text-green-300 font-semibold min-w-[140px] text-center">
                Memo (SSE)
              </div>
            </div>
          </div>
        </div>

      </div>
    </section>
  );
}
