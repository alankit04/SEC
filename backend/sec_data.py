"""
sec_data.py  —  Local SEC EDGAR data reader.
Reads sub.txt (fast) and num.txt (chunked) from quarterly directories.
"""

import re
import time
import json
from pathlib import Path
import pandas as pd

try:
    from paths import COMPANY_TICKERS_FILE, DATA_DIR
except ImportError:  # pragma: no cover - package import path
    from backend.paths import COMPANY_TICKERS_FILE, DATA_DIR

QUARTERS = [
    "2022q1","2022q2","2022q3","2022q4",
    "2023q1","2023q2","2023q3","2023q4",
    "2024q1","2024q2","2024q3","2024q4",
    "2025q1","2025q2","2025q3","2025q4",
]

# Key XBRL tags we care about
FINANCIAL_TAGS = {
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "NetIncomeLoss": "net_income",
    "EarningsPerShareBasic": "eps",
    "EarningsPerShareDiluted": "eps_diluted",
    "Assets": "total_assets",
    "StockholdersEquity": "equity",
    "OperatingIncomeLoss": "operating_income",
    "ResearchAndDevelopmentExpense": "rd_expense",
    "GrossProfit": "gross_profit",
    "LongTermDebt": "long_term_debt",
    "CashAndCashEquivalentsAtCarryingValue": "cash",
}

SIC_INDUSTRIES = {
    "01":"Agriculture","08":"Forestry","10":"Mining","12":"Coal Mining",
    "13":"Oil & Gas","14":"Stone/Clay/Glass","15":"Building Contractors",
    "20":"Food","22":"Textiles","23":"Apparel","24":"Lumber",
    "25":"Furniture","26":"Paper","27":"Printing","28":"Chemicals",
    "29":"Petroleum","30":"Rubber/Plastic","31":"Leather",
    "32":"Stone/Glass","33":"Primary Metals","34":"Fabricated Metals",
    "35":"Industrial Machinery","36":"Electronic Equipment",
    "37":"Transportation Equipment","38":"Instruments","39":"Misc. Manufacturing",
    "40":"Railroads","41":"Bus Transit","42":"Trucking",
    "44":"Water Transport","45":"Air Transport","47":"Transport Services",
    "48":"Communications","49":"Utilities",
    "50":"Wholesale (Durable)","51":"Wholesale (Non-Durable)",
    "52":"Retail (Building)","53":"Retail (General)","54":"Food Stores",
    "55":"Auto Dealers","56":"Apparel Stores","57":"Furniture Stores",
    "58":"Eating/Drinking","59":"Misc. Retail",
    "60":"Banking","61":"Credit Instit.","62":"Securities",
    "63":"Insurance","64":"Insurance Agents","65":"Real Estate",
    "67":"Holding Companies",
    "70":"Hotels","72":"Personal Services","73":"Business Services",
    "75":"Auto Repair","76":"Misc. Repair","78":"Motion Pictures",
    "79":"Amusement","80":"Health Services","81":"Legal Services",
    "82":"Education","83":"Social Services","87":"Engineering/Mgmt",
    "99":"Non-classifiable",
}


class SECData:
    def __init__(self, base_path: Path):
        base_path = Path(base_path)
        self.project_root = base_path
        self.base = base_path / "data" if (base_path / "data").exists() else base_path
        if not any((self.base / q).exists() for q in QUARTERS) and DATA_DIR.exists():
            self.base = DATA_DIR
        self._sub_cache: dict  = {}
        self._ticker_cik: dict = {}  # ticker → CIK (built lazily)
        self._cik_ticker: dict = {}
        self._cik_loaded = False

    # ------------------------------------------------------------------
    def _load_sub(self, quarter: str) -> pd.DataFrame:
        if quarter in self._sub_cache:
            return self._sub_cache[quarter]
        f = self.base / quarter / "sub.txt"
        if not f.exists():
            return pd.DataFrame()
        df = pd.read_csv(f, sep="\t", encoding="utf-8",
                         encoding_errors="replace", low_memory=False,
                         on_bad_lines="skip")
        self._sub_cache[quarter] = df
        return df

    # ------------------------------------------------------------------
    def _build_ticker_cik(self):
        """Build ticker-to-CIK mapping from SEC company_tickers.json."""
        if self._cik_loaded:
            return
        if COMPANY_TICKERS_FILE.exists():
            try:
                with open(COMPANY_TICKERS_FILE, encoding="utf-8") as f:
                    raw = json.load(f)
                rows = raw.values() if isinstance(raw, dict) else raw
                for row in rows:
                    ticker = str(row.get("ticker", "")).strip().upper()
                    cik_raw = row.get("cik_str")
                    if not ticker or cik_raw is None:
                        continue
                    cik = str(int(cik_raw))
                    self._ticker_cik.setdefault(ticker, cik)
                    self._cik_ticker.setdefault(cik, ticker)
            except Exception:
                pass
        self._cik_loaded = True

    # ------------------------------------------------------------------
    def cik_for_ticker(self, ticker: str) -> str | None:
        self._build_ticker_cik()
        return self._ticker_cik.get(ticker.upper())

    def ticker_filings(self, ticker: str, limit: int = 20) -> list:
        """Return recent filings for a ticker from the local filing index."""
        cik = self.cik_for_ticker(ticker)
        if cik is None:
            return []
        results = []
        for q in reversed(QUARTERS):
            df = self._load_sub(q)
            if df.empty or "cik" not in df.columns:
                continue
            mask = df["cik"].apply(lambda x: str(int(x)) == cik
                                   if pd.notna(x) else False)
            rows = df[mask]
            for _, row in rows.iterrows():
                results.append({
                    "adsh":    str(row.get("adsh", "")),
                    "form":    str(row.get("form", "")),
                    "filed":   str(row.get("filed", "")),
                    "period":  str(row.get("period", "")),
                    "quarter": q,
                })
            if len(results) >= limit:
                break
        return results[:limit]

    # ------------------------------------------------------------------
    def search_companies(self, query: str, limit: int = 20) -> list:
        """Case-insensitive search through company names."""
        q_lower = query.lower()
        seen_ciks = set()
        results   = []
        for q in reversed(QUARTERS):
            df = self._load_sub(q)
            if df.empty or "name" not in df.columns:
                continue
            mask = df["name"].str.lower().str.contains(q_lower, na=False)
            for _, row in df[mask].iterrows():
                cik = str(int(row["cik"])) if pd.notna(row.get("cik")) else ""
                if cik in seen_ciks:
                    continue
                seen_ciks.add(cik)
                sic = str(row.get("sic", "")).split(".")[0].zfill(4)
                results.append({
                    "cik":      cik,
                    "name":     str(row.get("name", "")),
                    "sic":      sic,
                    "industry": SIC_INDUSTRIES.get(sic[:2], "Other"),
                    "ticker":   self._ticker_for_cik(cik),
                })
                if len(results) >= limit:
                    return results
        return results

    # ------------------------------------------------------------------
    def company_universe(
        self,
        *,
        q: str = "",
        sic: str = "",
        industry: str = "",
        form: str = "",
        tickered_only: bool = True,
        limit: int = 100,
    ) -> dict:
        """
        Return a deduped SEC company universe from local sub.txt metadata.

        This is intentionally sub.txt-only so it can scan the full local SEC
        corpus quickly without touching the much larger num.txt files.
        """
        q_lower = q.strip().lower()
        sic_prefix = str(sic).strip()
        industry_lower = industry.strip().lower()
        form_upper = form.strip().upper()
        limit = max(1, min(int(limit or 100), 500))

        seen_ciks: set[str] = set()
        companies: list[dict] = []
        scanned_filings = 0

        for quarter in reversed(QUARTERS):
            df = self._load_sub(quarter)
            if df.empty or "cik" not in df.columns:
                continue
            scanned_filings += len(df)

            rows = df
            if form_upper and "form" in rows.columns:
                rows = rows[rows["form"].astype(str).str.upper() == form_upper]
            if q_lower and "name" in rows.columns:
                rows = rows[rows["name"].astype(str).str.lower().str.contains(q_lower, na=False)]
            if sic_prefix and "sic" in rows.columns:
                rows = rows[
                    rows["sic"].apply(
                        lambda x: str(x).split(".")[0].zfill(4).startswith(sic_prefix)
                        if pd.notna(x) else False
                    )
                ]
            if industry_lower and "sic" in rows.columns:
                rows = rows[
                    rows["sic"].apply(
                        lambda x: industry_lower in SIC_INDUSTRIES.get(
                            str(x).split(".")[0].zfill(4)[:2], "Other"
                        ).lower()
                        if pd.notna(x) else False
                    )
                ]

            for _, row in rows.iterrows():
                cik = str(int(row["cik"])) if pd.notna(row.get("cik")) else ""
                if not cik or cik in seen_ciks:
                    continue
                ticker = self._ticker_for_cik(cik)
                if tickered_only and not ticker:
                    continue
                seen_ciks.add(cik)

                sic_code = str(row.get("sic", "")).split(".")[0].zfill(4)
                companies.append({
                    "cik": cik,
                    "ticker": ticker,
                    "name": str(row.get("name", "")),
                    "sic": sic_code,
                    "industry": SIC_INDUSTRIES.get(sic_code[:2], "Other"),
                    "latest_form": str(row.get("form", "")),
                    "latest_filed": str(row.get("filed", "")),
                    "latest_period": str(row.get("period", "")),
                    "latest_quarter": quarter,
                })
                if len(companies) >= limit:
                    return {
                        "companies": companies,
                        "count": len(companies),
                        "scanned_filings": scanned_filings,
                        "filters": {
                            "q": q,
                            "sic": sic,
                            "industry": industry,
                            "form": form,
                            "tickered_only": tickered_only,
                            "limit": limit,
                        },
                    }

        return {
            "companies": companies,
            "count": len(companies),
            "scanned_filings": scanned_filings,
            "filters": {
                "q": q,
                "sic": sic,
                "industry": industry,
                "form": form,
                "tickered_only": tickered_only,
                "limit": limit,
            },
        }

    # ------------------------------------------------------------------
    def industry_summary(self) -> dict:
        """Aggregate local SEC company counts by 2-digit SIC industry."""
        latest_by_cik: dict[str, dict] = {}
        for quarter in reversed(QUARTERS):
            df = self._load_sub(quarter)
            if df.empty or "cik" not in df.columns:
                continue
            for _, row in df.iterrows():
                cik = str(int(row["cik"])) if pd.notna(row.get("cik")) else ""
                if not cik or cik in latest_by_cik:
                    continue
                sic_code = str(row.get("sic", "")).split(".")[0].zfill(4)
                latest_by_cik[cik] = {
                    "sic": sic_code,
                    "industry": SIC_INDUSTRIES.get(sic_code[:2], "Other"),
                    "ticker": self._ticker_for_cik(cik),
                }

        buckets: dict[str, dict] = {}
        for company in latest_by_cik.values():
            sic2 = company["sic"][:2]
            bucket = buckets.setdefault(sic2, {
                "sic2": sic2,
                "industry": company["industry"],
                "companies": 0,
                "tickered_companies": 0,
            })
            bucket["companies"] += 1
            if company["ticker"]:
                bucket["tickered_companies"] += 1

        rows = sorted(buckets.values(), key=lambda x: x["companies"], reverse=True)
        return {"industries": rows, "total_industries": len(rows)}

    # ------------------------------------------------------------------
    def _ticker_for_cik(self, cik: str) -> str:
        self._build_ticker_cik()
        return self._cik_ticker.get(str(int(cik)) if str(cik).isdigit() else str(cik), "")

    # ------------------------------------------------------------------
    def company_financial_entries(self, ticker: str, limit_filings: int = 8) -> list:
        """
        Return detailed XBRL financial rows for recent 10-K/10-Q/20-F filings.
        This is intentionally chunked because num.txt is hundreds of MB per quarter.
        """
        cik = self.cik_for_ticker(ticker)
        if cik is None:
            return []

        filings = self.ticker_filings(ticker, limit=limit_filings)
        filing_by_adsh = {
            f["adsh"]: f
            for f in filings
            if f["form"] in ("10-K", "10-Q", "20-F")
        }
        target_adshs = set(filing_by_adsh)
        if not target_adshs:
            return []

        entries: list = []
        seen = set()
        for q in reversed(QUARTERS):
            num_file = self.base / q / "num.txt"
            if not num_file.exists():
                continue

            for chunk in pd.read_csv(num_file, sep="\t", encoding="utf-8",
                                     encoding_errors="replace", low_memory=False,
                                     on_bad_lines="skip", chunksize=80_000):
                if "adsh" not in chunk.columns or "tag" not in chunk.columns:
                    break
                rows = chunk[
                    chunk["adsh"].isin(target_adshs)
                    & chunk["tag"].isin(FINANCIAL_TAGS.keys())
                ]
                for _, row in rows.iterrows():
                    tag = str(row.get("tag", ""))
                    key = (
                        str(row.get("adsh", "")),
                        tag,
                        str(row.get("ddate", "")),
                        str(row.get("qtrs", "")),
                        str(row.get("uom", "")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        val = float(row.get("value", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    ddate = str(row.get("ddate", ""))
                    period = (
                        f"{ddate[:4]}-{ddate[4:6]}-{ddate[6:8]}"
                        if re.match(r"^\d{8}$", ddate)
                        else ddate
                    )
                    filing = filing_by_adsh.get(str(row.get("adsh", "")), {})
                    entries.append({
                        "adsh": str(row.get("adsh", "")),
                        "form": filing.get("form", ""),
                        "filed": filing.get("filed", ""),
                        "period": period,
                        "quarter": q,
                        "tag": tag,
                        "metric": FINANCIAL_TAGS[tag],
                        "uom": str(row.get("uom", "")),
                        "qtrs": int(row.get("qtrs", 0) or 0),
                        "val": val,
                    })

            if entries:
                break

        entries.sort(key=lambda x: (x.get("period", ""), x.get("filed", "")), reverse=True)
        return entries

    # ------------------------------------------------------------------
    def company_financials(self, ticker: str) -> dict:
        """
        Extract key financial metrics from num.txt for the most recent
        10-K or 10-Q filing. Reads in chunks to avoid loading 464 MB.
        """
        metrics: dict = {}
        for entry in self.company_financial_entries(ticker):
            metrics.setdefault(entry["metric"], entry["val"])
        return metrics

    # ------------------------------------------------------------------
    def summary_stats(self) -> dict:
        """Overall database summary (fast — from sub.txt only)."""
        total_filings  = 0
        total_cos      = set()
        quarter_counts = {}
        for q in QUARTERS:
            df = self._load_sub(q)
            if df.empty:
                continue
            n = len(df)
            total_filings += n
            quarter_counts[q] = n
            if "cik" in df.columns:
                total_cos.update(df["cik"].dropna().astype(int).astype(str))
        return {
            "total_quarters": len(quarter_counts),
            "total_filings":  total_filings,
            "total_companies": len(total_cos),
            "by_quarter":     quarter_counts,
        }
