import { Database, Building2, Calendar, Layers, Wrench, Bot, Zap, Shield } from "lucide-react";

const BIG_STATS = [
  { value: "112,220", unit: "",    label: "SEC filings indexed",   sub: "10-K, 10-Q, 20-F, 8-K", icon: Database,   color: "text-blue-400"   },
  { value: "9,460",   unit: "",    label: "Listed companies",       sub: "CIK + SIC classified",  icon: Building2,  color: "text-violet-400" },
  { value: "54.9",    unit: "M",   label: "XBRL data points",       sub: "Tagged financial rows", icon: Layers,     color: "text-cyan-400"   },
  { value: "<3",      unit: "s",   label: "Avg memo latency",       sub: "Parallel dispatch",     icon: Zap,        color: "text-green-400"  },
];

const SUPPORT_STATS = [
  { value: "16", label: "Quarterly datasets",  icon: Calendar },
  { value: "8",  label: "MCP tools live",       icon: Wrench   },
  { value: "6",  label: "AI subagents",         icon: Bot      },
  { value: "10+",label: "Security controls",    icon: Shield   },
];

const QUARTERS = [
  { q: "2022q1", f: 6840 },{ q: "2022q2", f: 7020 },{ q: "2022q3", f: 7180 },{ q: "2022q4", f: 7250 },
  { q: "2023q1", f: 6910 },{ q: "2023q2", f: 7090 },{ q: "2023q3", f: 7160 },{ q: "2023q4", f: 7310 },
  { q: "2024q1", f: 6985 },{ q: "2024q2", f: 7140 },{ q: "2024q3", f: 7220 },{ q: "2024q4", f: 7410 },
  { q: "2025q1", f: 6780 },{ q: "2025q2", f: 6920 },{ q: "2025q3", f: 7020 },{ q: "2025q4", f: 7085 },
];

const MAX_F = Math.max(...QUARTERS.map(q => q.f));

export default function Stats() {
  return (
    <section id="stats" className="relative py-32 px-5 overflow-hidden">

      <div className="section-divider absolute inset-x-0 top-0" />
      <div className="aurora-orb w-[600px] h-[400px] top-[20%] right-[-10%] bg-blue-600/10" />

      <div className="relative max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-14 max-w-2xl">
          <p className="overline mb-3">Data coverage</p>
          <h2 className="text-4xl sm:text-5xl font-bold text-white tracking-tight leading-[1.05] mb-5">
            Numbers that{" "}
            <span className="text-gradient-blue">actually work offline.</span>
          </h2>
          <p className="text-white/45 text-lg leading-relaxed font-light">
            Every metric below reflects what's on disk right now. No external API calls,
            no rate limits, no vendor lock-in.
          </p>
        </div>

        {/* ── Big stats (bento style) ── */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          {BIG_STATS.map((s) => {
            const Icon = s.icon;
            return (
              <div key={s.label} className="card-base card-hover p-7 relative overflow-hidden group">
                <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r
                                from-transparent via-white/20 to-transparent
                                opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <Icon className={`w-5 h-5 ${s.color} mb-6 opacity-80`} />
                <p className="mb-2 flex items-baseline gap-1">
                  <span className="text-5xl font-black text-white tracking-tighter leading-none">
                    {s.value}
                  </span>
                  {s.unit && (
                    <span className="text-2xl font-bold text-white/50 tracking-tighter">
                      {s.unit}
                    </span>
                  )}
                </p>
                <p className="text-sm font-semibold text-white/75 mb-0.5">{s.label}</p>
                <p className="text-xs text-white/35 leading-snug">{s.sub}</p>
              </div>
            );
          })}
        </div>

        {/* ── Support stats (compact row) ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          {SUPPORT_STATS.map((s) => {
            const Icon = s.icon;
            return (
              <div key={s.label} className="card-base p-5 flex items-center gap-4">
                <div className="w-10 h-10 rounded-lg bg-white/[0.04] border border-white/[0.06]
                                flex items-center justify-center shrink-0">
                  <Icon className="w-4 h-4 text-white/55" />
                </div>
                <div className="min-w-0">
                  <p className="text-2xl font-black text-white leading-none tracking-tight">
                    {s.value}
                  </p>
                  <p className="text-[11px] text-white/40 font-medium mt-1 truncate">
                    {s.label}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Quarter coverage bar chart ── */}
        <div className="card-base p-7">
          <div className="flex items-center justify-between mb-6">
            <div>
              <p className="overline mb-1">Quarterly coverage</p>
              <h3 className="text-base font-bold text-white">
                2022 Q1 → 2025 Q4 · <span className="text-white/35 font-normal">all 16 quarters on disk</span>
              </h3>
            </div>
            <div className="chip bg-green-500/10 border border-green-500/20 text-green-400">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              Complete
            </div>
          </div>

          {/* Bar chart */}
          <div className="grid grid-cols-8 md:grid-cols-16 gap-1.5" style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}>
            {QUARTERS.map((q) => {
              const h = Math.round((q.f / MAX_F) * 100);
              return (
                <div key={q.q} className="group flex flex-col items-center gap-1.5 cursor-default">
                  {/* Bar */}
                  <div className="relative w-full h-20 flex items-end">
                    <div className="w-full rounded-t-md bg-gradient-to-t from-blue-500/30 to-blue-400/60
                                    group-hover:from-blue-500/50 group-hover:to-blue-400/90
                                    transition-all duration-300"
                         style={{ height: `${h}%` }} />
                    {/* Hover label */}
                    <div className="absolute -top-6 left-1/2 -translate-x-1/2 px-1.5 py-0.5
                                    rounded text-[9px] font-mono text-blue-300 bg-blue-500/15
                                    opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                      {q.f.toLocaleString()}
                    </div>
                  </div>
                  {/* Label */}
                  <span className="text-[8px] font-mono text-white/30 group-hover:text-white/70 transition-colors">
                    {q.q.replace("q", "Q")}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div className="mt-6 pt-4 border-t border-white/[0.05] flex items-center justify-between
                          text-[11px] font-mono text-white/35">
            <span>Filings per quarter · bar height = count</span>
            <span>max {MAX_F.toLocaleString()} · avg ≈ 7,015</span>
          </div>
        </div>

      </div>
    </section>
  );
}
