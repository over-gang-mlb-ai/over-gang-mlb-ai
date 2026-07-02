#!/usr/bin/env python3
"""
Refresh data/pitcher_handedness.csv from current MLB pitcher IDs.

Source of current pitcher universe:
- data/pitcher_k_stats.csv, refreshed by scripts/build_pitcher_k_stats.py

Source of handedness:
- MLB StatsAPI /api/v1/people/{mlb_id}
- people[0].pitchHand.code in {"R", "L"}

This script does not change model math, fire logic, predictions schema, or Telegram.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


MLB_BASE = "https://statsapi.mlb.com/api/v1"
K_STATS_CSV = Path("data/pitcher_k_stats.csv")
OUTFILE = Path("data/pitcher_handedness.csv")
SOURCE_FETCHED = "mlb_statsapi_people"
SOURCE_CACHED = "mlb_statsapi_people_cached"
TIMEOUT = 20


def norm_name(value: object) -> str:
    s = str(value or "").strip().lower()
    s = s.replace(".", "")
    s = s.replace("-", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def safe_int(value: object) -> Optional[int]:
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_existing_hands() -> tuple[dict[int, str], dict[str, str]]:
    by_id: dict[int, str] = {}
    by_name: dict[str, str] = {}

    if not OUTFILE.exists():
        return by_id, by_name

    try:
        df = pd.read_csv(OUTFILE)
    except Exception as exc:
        print(f"⚠️ Existing handedness file unreadable; rebuilding from API where needed: {exc}")
        return by_id, by_name

    name_col = "name_norm" if "name_norm" in df.columns else "name"
    if "hand" not in df.columns or name_col not in df.columns:
        return by_id, by_name

    for _, row in df.iterrows():
        hand = str(row.get("hand") or "").strip().upper()
        if hand not in {"R", "L"}:
            continue

        pid = safe_int(row.get("mlb_id"))
        if pid is not None:
            by_id[pid] = hand

        n = norm_name(row.get(name_col))
        if n:
            by_name[n] = hand

    return by_id, by_name


def fetch_pitch_hand(session: requests.Session, mlb_id: int) -> tuple[Optional[str], str]:
    url = f"{MLB_BASE}/people/{int(mlb_id)}"
    resp = session.get(url, headers={"User-Agent": "over-gang-mlb-ai/1.0"}, timeout=TIMEOUT)
    resp.raise_for_status()

    body = resp.json()
    people = body.get("people") if isinstance(body, dict) else None
    if not people:
        return None, ""

    person = people[0] or {}
    full_name = str(person.get("fullName") or "").strip()
    pitch_hand = person.get("pitchHand") or {}
    hand = str(pitch_hand.get("code") or "").strip().upper()

    if hand not in {"R", "L"}:
        return None, full_name

    return hand, full_name


def build_pitcher_handedness() -> pd.DataFrame:
    if not K_STATS_CSV.exists():
        raise FileNotFoundError(f"{K_STATS_CSV} does not exist; run build_pitcher_k_stats.py first")

    kdf = pd.read_csv(K_STATS_CSV)
    required = {"mlb_id", "Name"}
    missing = sorted(required - set(kdf.columns))
    if missing:
        raise ValueError(f"{K_STATS_CSV} missing required columns: {missing}")

    kdf = kdf[["mlb_id", "Name"]].copy()
    kdf["mlb_id"] = pd.to_numeric(kdf["mlb_id"], errors="coerce")
    kdf["Name"] = kdf["Name"].astype(str).str.strip()
    kdf = kdf.dropna(subset=["mlb_id"])
    kdf = kdf[kdf["Name"].ne("")]
    kdf["mlb_id"] = kdf["mlb_id"].astype(int)
    kdf = kdf.sort_values(["Name", "mlb_id"]).drop_duplicates(subset=["mlb_id"], keep="first")

    if kdf.empty:
        raise ValueError(f"No valid pitcher rows found in {K_STATS_CSV}")

    existing_by_id, existing_by_name = load_existing_hands()
    updated_at = datetime.now(timezone.utc).isoformat()

    rows = []
    fetched = reused = missing_hand = 0

    session = requests.Session()

    for _, row in kdf.iterrows():
        mlb_id = int(row["mlb_id"])
        name = str(row["Name"]).strip()
        name_norm = norm_name(name)

        hand = existing_by_id.get(mlb_id) or existing_by_name.get(name_norm)
        source = SOURCE_CACHED

        if hand not in {"R", "L"}:
            try:
                hand, api_name = fetch_pitch_hand(session, mlb_id)
                fetched += 1
                source = SOURCE_FETCHED
                if api_name:
                    name = api_name
                    name_norm = norm_name(api_name)
                time.sleep(0.03)
            except Exception as exc:
                print(f"⚠️ hand fetch failed mlb_id={mlb_id} name={name!r}: {exc}")
                hand = None

        else:
            reused += 1

        if hand not in {"R", "L"}:
            missing_hand += 1
            continue

        rows.append({
            "name": name,
            "name_norm": name_norm,
            "hand": hand,
            "mlb_id": mlb_id,
            "updated_at": updated_at,
            "source": source,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No pitcher handedness rows produced")

    out = out.sort_values(["name_norm", "mlb_id"]).drop_duplicates(subset=["name_norm"], keep="first")
    out = out[["name", "name_norm", "hand", "mlb_id", "updated_at", "source"]].reset_index(drop=True)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTFILE.with_suffix(OUTFILE.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, OUTFILE)

    print(f"Saved {OUTFILE} ({len(out)} rows)")
    print(f"source_rows={len(kdf)} reused={reused} fetched={fetched} missing_hand={missing_hand}")
    print("hand_counts:")
    print(out["hand"].value_counts().to_string())

    return out


def main() -> None:
    build_pitcher_handedness()


if __name__ == "__main__":
    main()
