import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAPHI — Real-time Agentic Platform for Human Investment Intelligence",
  description:
    "Institutional-grade AI investment platform powered by Claude agents, live market data, SEC EDGAR filings, and XGBoost ML signals. Built for serious investors.",
  keywords: ["investment intelligence", "AI trading", "SEC filings", "market analysis", "portfolio management"],
  openGraph: {
    title: "RAPHI — Investment Intelligence Platform",
    description: "Institutional-grade AI platform combining live market data, SEC EDGAR analysis, and ML signals.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#0a0a0f] overflow-x-hidden">
        {children}
      </body>
    </html>
  );
}
