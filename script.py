import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_PATH = PROJECT_ROOT / "data"

QUARTERS = [
    "2022q1", "2022q2", "2022q3", "2022q4",
    "2023q1", "2023q2", "2023q3", "2023q4",
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
]

TABLES = ["num", "pre", "sub", "tag"]

data = {}

for quarter in QUARTERS:
    folder = BASE_PATH / quarter
    if not folder.exists():
        print(f"Skipping missing quarter: {quarter}")
        continue

    data[quarter] = {}
    for table in TABLES:
        file = folder / f"{table}.txt"
        if not file.exists():
            print(f"  Missing file: {quarter}/{table}.txt")
            continue

        # Read first line to detect delimiter
        with open(file, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else ","

        df = pd.read_csv(
            file,
            sep=delimiter,
            encoding="utf-8",
            encoding_errors="replace",
            low_memory=False,
            on_bad_lines="skip",   
        )

        # Parse date columns
        for col in ["filed", "period", "ddate", "changed"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

        if "accepted" in df.columns:
            df["accepted"] = pd.to_datetime(df["accepted"], errors="coerce")

        data[quarter][table] = df
        print(f"Loaded {quarter}/{table}: {len(df):,} rows x {len(df.columns)} cols")

print(f"\nDone. Loaded {len(data)} quarters.")
