import { BarChart2, Github, ExternalLink, Terminal, BookOpen, Server, Shield } from "lucide-react";

const COLUMNS = [
  {
    title: "Platform",
    links: [
      { label: "Features",        href: "#features"     },
      { label: "How it works",    href: "#how-it-works" },
      { label: "Coverage",        href: "#stats"        },
      { label: "Security",        href: "#security"     },
    ],
  },
  {
    title: "Developer",
    links: [
      { label: "API docs",        href: "http://localhost:9999/docs",                       external: true },
      { label: "Health check",    href: "http://localhost:9999/api/health",                 external: true },
      { label: "Agent card",      href: "http://localhost:9999/.well-known/agent-card.json", external: true },
      { label: "MCP bridge",      href: "http://localhost:9999",                            external: true },
    ],
  },
  {
    title: "Resources",
    links: [
      { label: "Investment memo skill", href: "#" },
      { label: "Sector screen skill",    href: "#" },
      { label: "Portfolio review skill", href: "#" },
      { label: "Conviction ledger",      href: "#" },
    ],
  },
  {
    title: "Stack",
    links: [
      { label: "Claude Agent SDK", href: "https://github.com/anthropics/claude-agent-sdk-python", external: true },
      { label: "Model Context Protocol", href: "https://modelcontextprotocol.io",               external: true },
      { label: "FastAPI",          href: "https://fastapi.tiangolo.com", external: true },
      { label: "SEC EDGAR",        href: "https://www.sec.gov/edgar",    external: true },
    ],
  },
];

export default function Footer() {
  return (
    <footer className="relative border-t border-white/[0.06] bg-[#050507] overflow-hidden">
      <div className="aurora-orb w-[800px] h-[300px] bottom-[-200px] left-1/2 -translate-x-1/2 bg-blue-600/[0.05]" />

      <div className="relative max-w-6xl mx-auto px-5">

        {/* ── Top section ── */}
        <div className="pt-16 pb-12 grid grid-cols-2 md:grid-cols-5 gap-10">

          {/* Brand column */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600
                              flex items-center justify-center glow-brand">
                <BarChart2 className="w-4 h-4 text-white" />
              </div>
              <span className="text-base font-bold text-white tracking-tight">RAPHI</span>
              <span className="text-[9px] text-white/30 font-mono border border-white/10 rounded px-1 py-0.5">
                v2.4
              </span>
            </div>
            <p className="text-[13px] text-white/40 leading-relaxed mb-5 max-w-[220px]">
              The AI platform for investment intelligence. Institutional memos on your own machine.
            </p>

            {/* Live status */}
            <div className="flex items-center gap-2 text-[11px] font-mono text-white/40">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              <span>Live · localhost:9999</span>
            </div>
          </div>

          {/* Link columns */}
          {COLUMNS.map((col) => (
            <div key={col.title}>
              <h4 className="text-[11px] font-semibold tracking-[0.14em] uppercase text-white/75 mb-4">
                {col.title}
              </h4>
              <ul className="space-y-2.5">
                {col.links.map((link) => (
                  <li key={link.label}>
                    <a href={link.href}
                       target={"external" in link && link.external ? "_blank" : undefined}
                       rel={"external" in link && link.external ? "noopener noreferrer" : undefined}
                       className="group inline-flex items-center gap-1.5 text-[13px] text-white/40
                                  hover:text-white transition-colors">
                      {link.label}
                      {"external" in link && link.external && (
                        <ExternalLink className="w-2.5 h-2.5 opacity-0 group-hover:opacity-60 transition-opacity" />
                      )}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* ── Divider ── */}
        <div className="section-divider" />

        {/* ── Bottom bar ── */}
        <div className="py-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-white/30">
            <span>© 2026 RAPHI · MIT License</span>
            <span className="hidden md:inline text-white/15">·</span>
            <span className="flex items-center gap-1.5">
              <Shield className="w-3 h-3" />
              TokenAuth secured
            </span>
            <span className="hidden md:inline text-white/15">·</span>
            <span className="flex items-center gap-1.5">
              <Server className="w-3 h-3" />
              Runs locally
            </span>
          </div>

          <p className="text-[11px] font-mono tracking-wide text-white/25">
            Built with Next.js · Tailwind · Claude Agent SDK · MCP
          </p>
        </div>
      </div>
    </footer>
  );
}
