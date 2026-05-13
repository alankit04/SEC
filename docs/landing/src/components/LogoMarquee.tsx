const LOGOS = [
  { name: "Anthropic", tag: "Claude 4.5 Agent SDK" },
  { name: "SEC EDGAR", tag: "XBRL Financial Data" },
  { name: "yfinance",  tag: "Real-time Quotes"    },
  { name: "XGBoost",   tag: "Ensemble ML"          },
  { name: "FastAPI",   tag: "Python Backend"       },
  { name: "MCP",       tag: "Tool Protocol"        },
  { name: "Finnhub",   tag: "News & Sentiment"     },
  { name: "Next.js",   tag: "Dashboard UI"         },
];

export default function LogoMarquee() {
  return (
    <section className="relative py-14 border-y border-white/[0.06]
                        bg-gradient-to-b from-transparent via-white/[0.015] to-transparent">
      <div className="max-w-6xl mx-auto px-5">
        <p className="text-center text-[11px] font-semibold tracking-[0.24em] uppercase
                      text-white/35 mb-8">
          Built on best-in-class infrastructure
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-px
                        bg-white/[0.04] rounded-xl overflow-hidden">
          {LOGOS.map((logo) => (
            <div key={logo.name}
                 className="group bg-[#060609] px-4 py-5 flex flex-col items-center
                            justify-center text-center gap-1 hover:bg-white/[0.02]
                            transition-colors duration-300">
              <span className="text-[13px] font-bold text-white/60 group-hover:text-white
                               transition-colors tracking-tight">
                {logo.name}
              </span>
              <span className="text-[9px] font-medium text-white/25 tracking-wide uppercase">
                {logo.tag}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
