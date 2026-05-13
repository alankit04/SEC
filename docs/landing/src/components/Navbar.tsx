"use client";
import { useState, useEffect } from "react";
import { BarChart2, Menu, X } from "lucide-react";

const NAV = [
  { href: "#features",     label: "Features"     },
  { href: "#how-it-works", label: "How It Works" },
  { href: "#stats",        label: "Coverage"     },
  { href: "#security",     label: "Security"     },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen]         = useState(false);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 16);
    window.addEventListener("scroll", fn, { passive: true });
    return () => window.removeEventListener("scroll", fn);
  }, []);

  return (
    <nav className={`fixed top-0 inset-x-0 z-50 transition-all duration-500 ${
      scrolled ? "bg-[#060609]/80 backdrop-blur-2xl border-b border-white/[0.06]" : ""
    }`}>
      <div className="max-w-6xl mx-auto px-5 h-14 flex items-center justify-between">

        {/* Logo */}
        <a href="#" className="flex items-center gap-2 group">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600
                          flex items-center justify-center shadow-[0_0_12px_rgba(139,92,246,0.4)]">
            <BarChart2 className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="font-semibold text-sm tracking-tight text-white">RAPHI</span>
          <span className="text-[9px] text-white/25 font-mono border border-white/10 rounded px-1 py-0.5">v2.4</span>
        </a>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-1">
          {NAV.map(({ href, label }) => (
            <a key={href} href={href}
               className="px-3 py-1.5 text-[13px] text-white/50 hover:text-white
                          rounded-md hover:bg-white/[0.04] transition-all font-medium">
              {label}
            </a>
          ))}
        </div>

        {/* Desktop CTA */}
        <div className="hidden md:flex items-center gap-3">
          <a href="http://localhost:9999"
             className="text-[13px] text-white/40 hover:text-white transition-colors font-medium">
            Sign In
          </a>
          <a href="http://localhost:9999"
             className="px-3.5 py-1.5 text-[13px] font-semibold text-white rounded-lg
                        bg-gradient-to-r from-blue-600 to-violet-600
                        hover:from-blue-500 hover:to-violet-500
                        shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_28px_rgba(99,102,241,0.5)]
                        transition-all duration-200">
            Get Started
          </a>
        </div>

        {/* Mobile */}
        <button className="md:hidden p-1.5 text-white/40 hover:text-white" onClick={() => setOpen(!open)}>
          {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {open && (
        <div className="md:hidden bg-[#060609]/95 backdrop-blur-2xl border-t border-white/[0.06] px-5 py-4 space-y-1">
          {NAV.map(({ href, label }) => (
            <a key={href} href={href} onClick={() => setOpen(false)}
               className="block px-3 py-2.5 text-sm text-white/50 hover:text-white
                          hover:bg-white/[0.04] rounded-lg transition-all font-medium">
              {label}
            </a>
          ))}
          <div className="pt-3 border-t border-white/[0.06]">
            <a href="http://localhost:9999"
               className="block w-full text-center py-2.5 text-sm font-semibold text-white rounded-lg
                          bg-gradient-to-r from-blue-600 to-violet-600">
              Get Started
            </a>
          </div>
        </div>
      )}
    </nav>
  );
}
