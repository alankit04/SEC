---
name: add-quarter
description: Download a new SEC EDGAR quarter (e.g. 2025q4) and add it to the project. Pass the quarter name as an argument, e.g. /add-quarter 2026q1
disable-model-invocation: true
---

Arguments: quarter name like "2025q4"

Steps:
1. Validate the quarter format (YYYYqN, e.g. 2026q1)
2. Check if folder already exists at "/Users/alan/Desktop/SEC Data/{quarter}/"
   - If it does, confirm with the user whether to re-download
3. Tell the user to download the bulk zip from:
   https://www.sec.gov/dera/data/financial-statements
   Under the row for the requested quarter, download the zip and extract these files into
   "/Users/alan/Desktop/SEC Data/{quarter}/":
     - sub.txt
     - num.txt
     - tag.txt
     - pre.txt
4. Once the user confirms files are in place, add the quarter to the QUARTERS list in
   "/Users/alan/Desktop/SEC Data/generate_dashboard_data.py" (in chronological order)
5. Run the refresh-data skill to regenerate dashboard_data.js
