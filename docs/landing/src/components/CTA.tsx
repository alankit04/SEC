import { ArrowRight, Terminal, BookOpen } from "lucide-react";

export default function CTA() {
  return (
    <section className="relative py-32 px-5 overflow-hidden">
      <div className="section-divider absolute inset-x-0 top-0" />

      {/* Aurora background */}
      <div className="absolute inset-0 bg-grid bg-grid-fade opacity-50" />
      <div className="aurora-orb w-[900px] h-[500px] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-blue-600/15" />
      <div className="aurora-orb w-[600px] h-[400px] top-[20%] right-[10%] bg-violet-600/15" />
      <div className="aurora-orb w-[500px] h-[300px] bottom-[10%] left-[10%] bg-indigo-600/12" />

      <div className="relative max-w-5xl mx-auto">

        <div className="relative rounded-3xl overflow-hidden card-base noise">

          {/* Inner gradient */}
          <div className="absolute inset-0 bg-gradient-to-br from-blue-600/[0.08] via-transparent to-violet-600/[0.06]" />
          <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r
                          from-transparent via-blue-400/50 to-transparent" />

          <div className="relative z-10 p-10 sm:p-16 text-center">

            {/* Pill */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1 mb-8 rounded-full
                            glass text-[11px] font-semibold tracking-[0.14em] uppercase text-white/60">
              <Terminal className="w-3 h-3 text-blue-400" />
              100% Local · No cloud required
            </div>

            {/* Heading */}
            <h2 className="font-bold tracking-tight mb-6 max-w-3xl mx-auto"
                style={{ fontSize: "clamp(2.4rem, 5.5vw, 4.25rem)", lineHeight: 1.05 }}>
              <span className="text-white">Ready to invest </span>
              <span className="text-gradient-brand">intelligently?</span>
            </h2>

            <p className="text-white/50 text-lg max-w-xl mx-auto leading-relaxed mb-10 font-light">
              Launch RAPHI, ask your first question, and receive an institutional-grade
              memo in the time it takes to pour a coffee.
            </p>

            {/* Quick-start card */}
            <div className="max-w-xl mx-auto mb-10">
              <div className="card-base p-5 text-left font-mono text-sm">
                <div className="flex items-center justify-between mb-3 pb-3 border-b border-white/[0.05]">
                  <span className="text-[10px] font-semibold tracking-[0.18em] uppercase text-white/35">
                    Quick Start
                  </span>
                  <div className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-red-400/60" />
                    <span className="w-2 h-2 rounded-full bg-yellow-400/60" />
                    <span className="w-2 h-2 rounded-full bg-green-400/60" />
                  </div>
                </div>
                <p className="leading-relaxed">
                  <span className="text-white/30">$ </span>
                  <span className="text-white/85">cd raphi</span>
                  <span className="text-violet-400"> && </span>
                  <span className="text-white/85">node server.js</span>
                </p>
                <p className="text-white/30 mt-3 text-xs leading-relaxed">
                  <span className="text-green-400">✓</span> backend listening on{" "}
                  <span className="text-blue-300">http://localhost:9999</span>
                  <br/>
                  <span className="text-green-400">✓</span> MCP bridge registered (8 tools)
                  <br/>
                  <span className="text-green-400">✓</span> agent card served at{" "}
                  <span className="text-blue-300/80">/.well-known/agent-card.json</span>
                </p>
              </div>
            </div>

            {/* CTAs */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
              <a href="http://localhost:9999"
                 className="group inline-flex items-center gap-2 px-7 py-3.5 text-sm font-semibold
                            text-white rounded-xl bg-gradient-to-r from-blue-600 to-violet-600
                            hover:from-blue-500 hover:to-violet-500
                            shadow-[0_0_40px_rgba(99,102,241,0.4)] hover:shadow-[0_0_60px_rgba(99,102,241,0.6)]
                            transition-all duration-200 hover:-translate-y-0.5">
                Launch Dashboard
                <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </a>
              <a href="http://localhost:9999/docs"
                 className="inline-flex items-center gap-2 px-7 py-3.5 text-sm font-medium
                            text-white/60 hover:text-white border border-white/10
                            hover:border-white/25 rounded-xl transition-all
                            hover:bg-white/[0.03]">
                <BookOpen className="w-4 h-4" />
                API Documentation
              </a>
            </div>

            {/* Footer note */}
            <p className="mt-10 text-[11px] font-mono tracking-wide text-white/25">
              Requires Python 3.14 · Node 20+ · ~50 GB disk for SEC archive
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
