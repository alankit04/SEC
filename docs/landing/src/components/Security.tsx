import { Key, Shield, Lock, Zap, Eye, Activity, CheckCircle2 } from "lucide-react";

const CONTROLS = [
  {
    icon:   Key,
    title:  "Dual-Token Auth",
    desc:   "X-API-Key for external clients, X-Internal-Token for MCP bridge. Pure ASGI middleware — FastAPI 0.115+ compatible.",
    status: "Active",
    tag:    "AUTH",
  },
  {
    icon:   Shield,
    title:  "Prompt Injection Guard",
    desc:   "15 regex patterns block jailbreaks before they reach Claude. Input capped at 4,000 chars. All attempts sent to Sentry.",
    status: "Active",
    tag:    "INPUT",
  },
  {
    icon:   Lock,
    title:  "Session Encryption",
    desc:   "Fernet symmetric encryption on sessions.json. Key from RAPHI_SESSION_KEY env, fallback to machine-derived SHA-256.",
    status: "Active",
    tag:    "STORAGE",
  },
  {
    icon:   Zap,
    title:  "Rate Limiting",
    desc:   "Sliding window counters — 3/min ML signals, 20/min stock detail. Audit files stored chmod 0o700 in .raphi_audit/.",
    status: "Active",
    tag:    "ABUSE",
  },
  {
    icon:   Eye,
    title:  "Ticker Validation",
    desc:   "Strict regex ^[A-Z]{1,5}$ on every symbol before URL construction. Blocks path traversal and SSRF.",
    status: "Active",
    tag:    "INPUT",
  },
  {
    icon:   Activity,
    title:  "Runtime Monitoring",
    desc:   "Sentry SDK with logging + asyncio integrations. PII scrubbed via before_send. CORS locked to localhost only.",
    status: "Active",
    tag:    "OBSERVABILITY",
  },
];

export default function Security() {
  return (
    <section id="security" className="relative py-32 px-5 overflow-hidden">
      <div className="aurora-orb w-[600px] h-[400px] top-[15%] left-[5%] bg-green-500/08" />

      <div className="relative max-w-6xl mx-auto">

        {/* Header */}
        <div className="mb-14 flex flex-col md:flex-row md:items-end md:justify-between gap-4 max-w-none">
          <div className="max-w-2xl">
            <p className="overline mb-3 text-green-400/80">Enterprise security</p>
            <h2 className="text-4xl sm:text-5xl font-bold text-white tracking-tight leading-[1.05] mb-5">
              Hardened by default.<br />
              <span className="text-gradient-brand">Observable by design.</span>
            </h2>
            <p className="text-white/45 text-lg leading-relaxed font-light">
              Ten controls spanning authentication, input validation, storage,
              and runtime — every one battle-tested in production.
            </p>
          </div>

          {/* Live status card */}
          <div className="card-base p-5 min-w-[240px]">
            <div className="flex items-center gap-2 mb-2">
              <span className="relative flex w-2 h-2">
                <span className="absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75 animate-ping" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
              </span>
              <span className="text-[11px] font-semibold tracking-[0.14em] uppercase text-green-400">
                All systems nominal
              </span>
            </div>
            <p className="text-[11px] font-mono text-white/35 leading-relaxed">
              last audit: Round 4 · 2026-04-14<br/>
              6/6 controls · 0 active incidents
            </p>
          </div>
        </div>

        {/* ── Status-board grid ── */}
        <div className="card-base overflow-hidden">

          {/* Header bar */}
          <div className="flex items-center justify-between px-5 py-3 border-b border-white/[0.06]
                          bg-white/[0.015] font-mono text-[11px] tracking-wide">
            <span className="text-white/35">raphi://security/controls</span>
            <span className="flex items-center gap-2 text-green-400">
              <CheckCircle2 className="w-3 h-3" />
              6 / 6 ACTIVE
            </span>
          </div>

          {/* Control rows */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-white/[0.04]">
            {CONTROLS.map((c) => {
              const Icon = c.icon;
              return (
                <div key={c.title}
                     className="group bg-[#060609] p-6 hover:bg-white/[0.02] transition-colors duration-300">
                  {/* Top row */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-10 h-10 rounded-xl bg-green-500/10 border border-green-500/20
                                    flex items-center justify-center text-green-400">
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[9px] font-mono tracking-[0.14em] text-white/25">
                        {c.tag}
                      </span>
                      <span className="flex items-center gap-1 text-[10px] font-semibold text-green-400">
                        <span className="w-1 h-1 rounded-full bg-green-400" />
                        {c.status}
                      </span>
                    </div>
                  </div>

                  {/* Title + desc */}
                  <h3 className="text-[15px] font-bold text-white mb-2 tracking-tight">
                    {c.title}
                  </h3>
                  <p className="text-[13px] text-white/45 leading-relaxed">
                    {c.desc}
                  </p>
                </div>
              );
            })}
          </div>

          {/* Footer */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2
                          px-5 py-3 border-t border-white/[0.06] bg-white/[0.015]
                          font-mono text-[11px] text-white/30">
            <span>Memory stores % changes only · Backend URL never exposed</span>
            <span>CORS · localhost only · no external egress</span>
          </div>
        </div>

      </div>
    </section>
  );
}
