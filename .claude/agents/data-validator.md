---
name: data-validator
description: Validates dashboard_data.js after ETL runs — checks for zero-filing quarters, missing company data, and SIC anomalies
---

When invoked after generate_dashboard_data.py runs:
1. Read "/Users/alan/Desktop/SEC Data/dashboard_data.js"
2. Parse the JSON from window.RAPHI_DATA = {...}
3. Run these checks:
   - Any quarter with filings == 0? (likely a missing sub.txt file)
   - Any quarter with num_data_points == 0? (missing num.txt)
   - Any companies in top_companies with name == "" or sic == "nan"?
   - Is total_data_points_est > 40,000,000? (sanity check — should be ~51M for full dataset)
   - Is total_companies > 5,000? (sanity check)
   - Are all expected quarters present? (2022q1 through current)
4. Report a brief pass/fail summary — list any anomalies clearly, or confirm "All checks passed"
