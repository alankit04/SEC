"""
generate_dashboard_data.py
Reads SEC EDGAR sub.txt files (one per quarter) and outputs dashboard_data.js
with summary statistics for raphi_dashboard.html.

Only reads sub.txt (~6-7 K rows per quarter) for speed.
Estimates num.txt data-point counts from file size rather than reading all rows.

Run:  python generate_dashboard_data.py
"""

import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_PATH = PROJECT_ROOT / "data"

QUARTERS = [
    "2022q1", "2022q2", "2022q3", "2022q4",
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
]

# SIC major-group → industry label (first 2 digits of SIC)
SIC_INDUSTRIES = {
    "01": "Agriculture", "08": "Forestry", "10": "Mining", "12": "Coal Mining",
    "13": "Oil & Gas", "14": "Stone/Clay/Glass", "15": "Building Contractors",
    "20": "Food", "22": "Textiles", "23": "Apparel", "24": "Lumber",
    "25": "Furniture", "26": "Paper", "27": "Printing", "28": "Chemicals",
    "29": "Petroleum", "30": "Rubber/Plastic", "31": "Leather",
    "32": "Stone/Glass", "33": "Primary Metals", "34": "Fabricated Metals",
    "35": "Industrial Machinery", "36": "Electronic Equipment", "37": "Transportation Equipment",
    "38": "Instruments", "39": "Misc. Manufacturing",
    "40": "Railroads", "41": "Bus Transit", "42": "Trucking",
    "44": "Water Transport", "45": "Air Transport", "47": "Transport Services",
    "48": "Communications", "49": "Utilities",
    "50": "Wholesale (Durable)", "51": "Wholesale (Non-Durable)",
    "52": "Retail (Building)", "53": "Retail (General)", "54": "Food Stores",
    "55": "Auto Dealers", "56": "Apparel Stores", "57": "Furniture Stores",
    "58": "Eating/Drinking Places", "59": "Misc. Retail",
    "60": "Banking", "61": "Credit Instit.", "62": "Securities",
    "63": "Insurance", "64": "Insurance Agents", "65": "Real Estate",
    "67": "Holding Companies",
    "70": "Hotels", "72": "Personal Services", "73": "Business Services",
    "75": "Auto Repair", "76": "Misc. Repair", "78": "Motion Pictures",
    "79": "Amusement", "80": "Health Services", "81": "Legal Services",
    "82": "Education", "83": "Social Services", "84": "Museums",
    "86": "Membership Orgs", "87": "Engineering/Mgmt",
    "99": "Non-classifiable",
}

def sic_industry(sic_val):
    s = str(sic_val).strip().split(".")[0].zfill(4)
    return SIC_INDUSTRIES.get(s[:2], "Other")


def estimate_num_rows(quarter: str) -> int:
    """Estimate row count of num.txt from file size (avoids reading 3.5 M rows)."""
    f = BASE_PATH / quarter / "num.txt"
    if not f.exists():
        return 0
    # Empirical: 2024q3 num.txt is ~500 MB / 3.52 M rows ≈ 142 bytes/row
    return max(0, int(f.stat().st_size / 142) - 1)


quarters_data = []
company_registry: dict = {}  # cik_str -> {name, sic, filings}
total_filings = 0

for quarter in QUARTERS:
    folder = BASE_PATH / quarter
    sub_file = folder / "sub.txt"
    if not folder.exists() or not sub_file.exists():
        print(f"  Skipping missing quarter: {quarter}")
        continue

    print(f"Loading {quarter}/sub.txt …", end=" ", flush=True)
    df = pd.read_csv(
        sub_file,
        sep="\t",
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
        on_bad_lines="skip",
    )
    print(f"{len(df):,} rows")

    # Form-type distribution
    form_dist: dict = {}
    if "form" in df.columns:
        form_dist = {k: int(v) for k, v in df["form"].value_counts().head(10).items()}

    # Accumulate company registry
    for _, row in df.iterrows():
        raw_cik = row.get("cik")
        if pd.isna(raw_cik):
            continue
        cik = str(int(raw_cik))
        if cik not in company_registry:
            company_registry[cik] = {
                "name": str(row.get("name", "")).strip(),
                "sic": str(row.get("sic", "")).split(".")[0] if pd.notna(row.get("sic")) else "",
                "filings": 0,
            }
        company_registry[cik]["filings"] += 1

    n_filings = len(df)
    total_filings += n_filings
    quarters_data.append({
        "id": quarter,
        "filings": n_filings,
        "form_types": form_dist,
        "num_data_points": estimate_num_rows(quarter),
    })

total_companies = len(company_registry)
total_data_points_est = sum(q["num_data_points"] for q in quarters_data)

# Aggregate form-type distribution across all quarters
all_form_dist: dict = {}
for q in quarters_data:
    for form, count in q["form_types"].items():
        all_form_dist[form] = all_form_dist.get(form, 0) + count
all_form_dist = dict(sorted(all_form_dist.items(), key=lambda x: x[1], reverse=True)[:8])

# Top 15 companies by total filing count
top_companies = sorted(company_registry.items(), key=lambda x: x[1]["filings"], reverse=True)[:15]
top_companies_list = [
    {
        "cik": cik,
        "name": info["name"],
        "sic": info["sic"],
        "filings": info["filings"],
        "industry": sic_industry(info["sic"]),
    }
    for cik, info in top_companies
]

output = {
    "generated_at": datetime.now().isoformat(),
    "summary": {
        "total_quarters": len(quarters_data),
        "total_filings": total_filings,
        "total_companies": total_companies,
        "total_data_points_est": total_data_points_est,
    },
    "quarters": quarters_data,
    "top_companies": top_companies_list,
    "form_type_dist": all_form_dist,
}

# Write dashboard_data.js — loaded as a plain <script> tag (works with file:// protocol)
out_js = PROJECT_ROOT / "dashboard_data.js"
with open(out_js, "w", encoding="utf-8") as fh:
    fh.write("// Auto-generated by generate_dashboard_data.py — do not edit manually.\n")
    fh.write(f"window.RAPHI_DATA = {json.dumps(output, indent=2)};\n")

print(f"\n✓ Written: {out_js}")
print(f"  Quarters:    {len(quarters_data)}")
print(f"  Filings:     {total_filings:,}")
print(f"  Companies:   {total_companies:,}")
print(f"  Data pts*:   {total_data_points_est:,}  (* estimated from file size)")
