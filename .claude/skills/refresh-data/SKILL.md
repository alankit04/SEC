---
name: refresh-data
description: Re-run the ETL pipeline to regenerate dashboard_data.js from all quarterly sub.txt files
disable-model-invocation: true
---

Run the ETL pipeline:
1. Check that .venv exists at "/Users/alan/Desktop/SEC Data/.venv"
2. Run: cd "/Users/alan/Desktop/SEC Data" && .venv/bin/python generate_dashboard_data.py
3. Confirm dashboard_data.js was updated (check mtime)
4. Report: quarters processed, total filings, companies, estimated data points
