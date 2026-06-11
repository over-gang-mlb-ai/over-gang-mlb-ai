#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_team_offense_splits.py — Over Gang MLB

Source-only script.

Purpose:
- Build clean MLB team offense vs pitcher-handedness splits from MLB StatsAPI.
- No FanGraphs.
- No RotoWire.
- No model math.
- No fire logic.
- No Telegram.
- No cron changes.

Output:
- data/team_offense_splits.csv

Important:
- MLB StatsAPI does not provide official team split wOBA/wRC+ here.
- This script computes Estimated_wOBA from raw events using fixed default weights.
- It does NOT create or claim official wRC+.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests


MLB_BASE = "https://statsapi.mlb.com/api/v1"
DATA_DIR = Path("data")
OUT_FILE = DATA_DIR / "team_offense_splits.csv"

TIMEOUT = 30
RETRIES = 3
SLEEP_BASE = 1.0

# Default estimated wOBA weights.
# These are intentionally named as estimated/static weights, not official season Guts constants.
WOBA_WEIGHTS = {
    "wBB": 0.690,
    "wHBP": 0.720,
    "w1B": 0.880,
    "w2B": 1.247,
    "w3B": 1.578,
    "wHR": 2.031,
}

REQUIRED_RAW = [
    "plateAppearances",
    "atBats",
    "hits",
    "doubles",
    "triples",
    "homeRuns",
    "baseOnBalls",
    "intentionalWalks",
    "hitByPitch",
    "sacFlies",
    "strikeOuts",
    "avg",
    "obp",
    "slg",
    "ops",
]

SPLITS = {
    "R": "vr",  # vs right-handed pitcher
    "L": "vl",  # vs left-handed pitcher
}


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(str(value).strip()))
        except Exception:
            return None


def rate_pct(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return round((numerator / denominator) * 100.0, 3)


def get_json(session: requests.Session, url: str, params: Optional[dict] = None, label: str = "") -> Optional[dict]:
    last_status = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            last_status = r.status_code
            if r.status_code == 200:
                return r.json()
            log(f"⚠️ {label or url} HTTP {r.status_code}; retry {attempt}/{RETRIES}")
        except requests.RequestException as exc:
            log(f"⚠️ {label or url} request error: {exc}; retry {attempt}/{RETRIES}")
        time.sleep(SLEEP_BASE * attempt)
    log(f"❌ {label or url} failed after {RETRIES} attempts; last_status={last_status}")
    return None


def fetch_teams(session: requests.Session, season: int) -> list:
    data = get_json(
        session,
        f"{MLB_BASE}/teams",
        params={"sportId": 1, "season": season},
        label="teams",
    )
    if not data:
        return []
    teams = data.get("teams", [])
    # Keep only normal MLB clubs with IDs/names.
    return [t for t in teams if t.get("id") and t.get("name")]


def fetch_team_split_stat(session: requests.Session, team_id: int, season: int, sit_code: str) -> Dict[str, Any]:
    data = get_json(
        session,
        f"{MLB_BASE}/teams/{team_id}/stats",
        params={
            "stats": "statSplits",
            "group": "hitting",
            "season": season,
            "sitCodes": sit_code,
        },
        label=f"team_id={team_id} sit={sit_code}",
    )
    if not data:
        return {}

    stats_blocks = data.get("stats", [])
    if not stats_blocks:
        return {}

    splits = stats_blocks[0].get("splits", [])
    if not splits:
        return {}

    return splits[0].get("stat", {}) or {}


def build_row(team: Dict[str, Any], pitcher_hand: str, stat: Dict[str, Any], season: int, updated_at: str) -> Dict[str, Any]:
    pa = safe_int(stat.get("plateAppearances"))
    ab = safe_int(stat.get("atBats"))
    hits = safe_int(stat.get("hits"))
    doubles = safe_int(stat.get("doubles"))
    triples = safe_int(stat.get("triples"))
    home_runs = safe_int(stat.get("homeRuns"))
    walks = safe_int(stat.get("baseOnBalls"))
    ibb = safe_int(stat.get("intentionalWalks"))
    hbp = safe_int(stat.get("hitByPitch"))
    sac_flies = safe_int(stat.get("sacFlies"))
    strikeouts = safe_int(stat.get("strikeOuts"))

    avg = safe_float(stat.get("avg"))
    obp = safe_float(stat.get("obp"))
    slg = safe_float(stat.get("slg"))
    ops = safe_float(stat.get("ops"))

    singles = None
    if all(v is not None for v in [hits, doubles, triples, home_runs]):
        singles = hits - doubles - triples - home_runs

    iso = None
    if slg is not None and avg is not None:
        iso = round(slg - avg, 3)

    bb_pct = rate_pct(walks, pa)
    k_pct = rate_pct(strikeouts, pa)

    ubb = None
    if walks is not None and ibb is not None:
        ubb = walks - ibb
    ubb_pct = rate_pct(ubb, pa)

    estimated_woba = None
    denominator = None
    if all(v is not None for v in [ab, walks, ibb, sac_flies, hbp]):
        denominator = ab + walks - ibb + sac_flies + hbp

    if (
        denominator not in (None, 0)
        and all(v is not None for v in [ubb, hbp, singles, doubles, triples, home_runs])
    ):
        estimated_woba = (
            WOBA_WEIGHTS["wBB"] * ubb
            + WOBA_WEIGHTS["wHBP"] * hbp
            + WOBA_WEIGHTS["w1B"] * singles
            + WOBA_WEIGHTS["w2B"] * doubles
            + WOBA_WEIGHTS["w3B"] * triples
            + WOBA_WEIGHTS["wHR"] * home_runs
        ) / denominator
        estimated_woba = round(estimated_woba, 3)

    required_present = all(stat.get(k) not in (None, "") for k in REQUIRED_RAW)
    quality = "mlb_raw_complete" if required_present and denominator not in (None, 0) else "mlb_raw_incomplete"

    return {
        "Season": season,
        "Team_ID": team.get("id"),
        "Team": team.get("abbreviation", ""),
        "Team_Code": team.get("teamCode", ""),
        "File_Code": team.get("fileCode", ""),
        "Team_Name": team.get("name", ""),
        "Pitcher_Hand": pitcher_hand,

        "PA": pa,
        "AB": ab,
        "H": hits,
        "1B": singles,
        "2B": doubles,
        "3B": triples,
        "HR": home_runs,
        "BB": walks,
        "IBB": ibb,
        "uBB": ubb,
        "HBP": hbp,
        "SF": sac_flies,
        "SO": strikeouts,

        "AVG": avg,
        "OBP": obp,
        "SLG": slg,
        "OPS": ops,
        "ISO": iso,
        "BB_Pct": bb_pct,
        "K_Pct": k_pct,
        "uBB_Pct": ubb_pct,
        "Estimated_wOBA": estimated_woba,
        "wOBA_Source": "estimated_static_weights",
        "wRC_Plus": "",

        "Source": "mlb_statsapi_raw_team_splits",
        "Quality": quality,
        "Updated_At": updated_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=datetime.now(timezone.utc).year)
    args = parser.parse_args()

    season = int(args.season)
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) OverGangMLB/1.0",
        "Accept": "application/json,text/plain,*/*",
    })

    log(f"🔄 Building team offense splits from MLB StatsAPI raw team statSplits season={season}")

    teams = fetch_teams(session, season)
    log(f"Teams loaded: {len(teams)}")

    if len(teams) != 30:
        log(f"❌ Expected 30 MLB teams, got {len(teams)}")
        return 1

    rows = []

    for team in teams:
        team_id = int(team["id"])
        abbr = team.get("abbreviation", "")
        name = team.get("name", "")

        for pitcher_hand, sit_code in SPLITS.items():
            stat = fetch_team_split_stat(session, team_id, season, sit_code)
            if not stat:
                log(f"⚠️ Missing split stat for {abbr} {name} vs {pitcher_hand}HP")
            rows.append(build_row(team, pitcher_hand, stat, season, updated_at))

    df = pd.DataFrame(rows)

    cols = [
        "Season", "Team_ID", "Team", "Team_Code", "File_Code", "Team_Name", "Pitcher_Hand",
        "PA", "AB", "H", "1B", "2B", "3B", "HR", "BB", "IBB", "uBB", "HBP", "SF", "SO",
        "AVG", "OBP", "SLG", "OPS", "ISO", "BB_Pct", "K_Pct", "uBB_Pct", "Estimated_wOBA",
        "wOBA_Source", "wRC_Plus", "Source", "Quality", "Updated_At",
    ]
    df = df[cols].sort_values(["Team_Name", "Pitcher_Hand"]).reset_index(drop=True)

    row_count = len(df)
    hand_counts = df.groupby("Pitcher_Hand").size().to_dict()
    quality_counts = df["Quality"].value_counts().to_dict()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    log(f"💾 Saved {row_count} rows → {OUT_FILE}")
    log(f"Pitcher_Hand counts: {hand_counts}")
    log(f"Quality counts: {quality_counts}")

    if row_count != 60:
        log(f"❌ Expected 60 rows, got {row_count}")
        return 1

    if hand_counts.get("R") != 30 or hand_counts.get("L") != 30:
        log(f"❌ Expected 30 rows per hand, got {hand_counts}")
        return 1

    incomplete = int((df["Quality"] != "mlb_raw_complete").sum())
    if incomplete:
        log(f"⚠️ {incomplete} incomplete split rows found. File written, but inspect before model use.")
    else:
        log("✅ All 60 split rows are mlb_raw_complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
