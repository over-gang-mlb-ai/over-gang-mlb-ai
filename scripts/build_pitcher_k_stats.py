#!/usr/bin/env python3
"""
Build pitcher strikeout-rate stats for K props.

Reads the existing real source file that contains K/9 and writes a standalone
data/pitcher_k_stats.csv without changing the production pitcher_stats.csv.
"""

import os
from datetime import datetime, timezone

import pandas as pd


SOURCE_FILE = "data/pitcher_stats_with_xera_with_collumns.csv"
OUTFILE = "data/pitcher_k_stats.csv"
SOURCE_LABEL = os.path.basename(SOURCE_FILE)


def build_pitcher_k_stats(source_file: str = SOURCE_FILE, outfile: str = OUTFILE) -> pd.DataFrame:
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Required source file not found: {source_file}")

    df = pd.read_csv(source_file)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"Name", "K/9"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source_file} missing required column(s): {missing}")

    out = pd.DataFrame()
    out["Name"] = df["Name"].astype(str).str.strip()
    if "Team" in df.columns:
        out["Team"] = df["Team"]

    out["IP"] = pd.to_numeric(df["IP"], errors="coerce") if "IP" in df.columns else pd.NA
    out["K9"] = pd.to_numeric(df["K/9"], errors="coerce")

    if "WHIP" in df.columns:
        out["WHIP"] = pd.to_numeric(df["WHIP"], errors="coerce")
    if "xERA" in df.columns:
        out["xERA"] = pd.to_numeric(df["xERA"], errors="coerce")

    out = out[(out["Name"] != "") & out["K9"].notna()].copy()
    out["Source"] = SOURCE_LABEL
    out["Updated_At"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    tmp_path = f"{outfile}.tmp"
    out.to_csv(tmp_path, index=False)
    os.replace(tmp_path, outfile)
    return out


def main() -> None:
    out = build_pitcher_k_stats()
    print(f"Saved {OUTFILE} ({len(out)} rows)")


if __name__ == "__main__":
    main()
