import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        "ticker":    "ticker 35s linear infinite",
        "ticker2":   "ticker 50s linear infinite",
        "fade-up":   "fadeUp 0.7s ease-out forwards",
        "glow-pulse":"glowPulse 3s ease-in-out infinite",
        "beam":      "beam 6s ease-in-out infinite",
      },
      keyframes: {
        ticker:    { "0%": { transform: "translateX(0)" }, "100%": { transform: "translateX(-50%)" } },
        fadeUp:    { "0%": { opacity: "0", transform: "translateY(24px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
        glowPulse: { "0%,100%": { opacity: "0.4" }, "50%": { opacity: "0.8" } },
        beam:      { "0%,100%": { transform: "translateX(-100%)" }, "50%": { transform: "translateX(400%)" } },
      },
      backgroundImage: {
        "grid-pattern": "linear-gradient(to right,rgba(255,255,255,0.03) 1px,transparent 1px),linear-gradient(to bottom,rgba(255,255,255,0.03) 1px,transparent 1px)",
      },
      backgroundSize: {
        "grid-sm": "48px 48px",
      },
    },
  },
  plugins: [],
};
export default config;
