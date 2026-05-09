#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_batters.py — Over Gang AI (MLB-only; no FanGraphs)

What it does
------------
1) Pulls MLB teams (sportId=1) and iterates active rosters to collect hitters.
2) For each hitter, fetches overall season batting stats (type=season, group=hitting).
3) Attempts to fetch vs-pitcher-hand splits using MLB Stats API (stats=statSplits + sitCodes vr/vl).
   - Writes both legacy columns (vsR_OBP, vsR_OPS, …) and advanced columns expected by
     core/batters.py lineup scoring: vsR_wOBA, vsL_wOBA, vsR_wRC+, vsL_wRC+, vsR_ISO, vsL_ISO
     when the API stat dict includes woba / wrcPlus / iso (or derivable ISO from slg−avg).
   - If splits are missing/empty for a player, preserves last-good splits from the previous CSV.
4) Merges Baseball Savant expected-stat leaderboard CSV (xwOBA, wOBA, xBA, xSLG, PA) by MLBAM
   player_id — data enrichment only; does not fill vsR_/vsL_ wOBA from season totals.
5) Writes a unified CSV: data/batter_stats.csv

Usage
-----
python scripts/update_batters.py [--season 2025]

Environment
-----------
None required.

Logging
-------
Prints to stdout; cron should redirect to logs/batters*.log.
"""

import argparse
import io
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

# -----------------------
# Config
# -----------------------
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "batter_stats.csv"

MLB_BASE = "https://statsapi.mlb.com"
TIMEOUT = 30
RETRIES = 3
SLEEP_BASE = 1.5  # backoff base seconds: 1.5, 3.0, 4.5

DEFAULT_SEASON = pd.Timestamp.now(tz="UTC").year

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

SPLIT_PREFIXES = ("vsR_", "vsL_")

SAVANT_EXPECTED_BATTER_URL = "https://baseballsavant.mlb.com/leaderboard/expected_statistics"


# -----------------------
# Logging helpers
# -----------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    print(f"🔄 {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"✅ {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"⚠️ {msg}", flush=True)


# -----------------------
# Networking
# -----------------------
session = requests.Session()
session.headers.update(HEADERS)


def get_json(url: str, params: Optional[dict] = None, label: str = "") -> Optional[dict]:
    last = None
    for i in range(RETRIES):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
            last = r.status_code
            if r.status_code == 200:
                return r.json()
            warn(f"{label or 'GET'} HTTP {r.status_code} on {r.url} — retrying in {SLEEP_BASE*(i+1):.1f}s")
        except requests.RequestException as e:
            warn(f"{label or 'GET'} exception {e} — retrying in {SLEEP_BASE*(i+1):.1f}s")
        time.sleep(SLEEP_BASE * (i + 1))
    warn(f"{label or 'GET'} failed after {RETRIES} attempts (last={last}).")
    return None


# -----------------------
# Helpers
# -----------------------
def norm_key(name: str) -> str:
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode()
    return s.lower().strip()


def _safe_float(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).replace("%", ""))
        except Exception:
            return None


def _safe_int(x) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _hitting_split_advanced(stat: Optional[dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extract wOBA, wRC+, ISO from an MLB hitting `stat` dict (platoon or season).
    Tries common camelCase keys; ISO falls back to slg - avg when `iso` is absent.
    """
    if not stat:
        return None, None, None
    woba = _safe_float(stat.get("woba") or stat.get("wOBA"))
    wrc = _safe_float(stat.get("wrcPlus") or stat.get("wRCPlus"))
    iso = _safe_float(stat.get("iso") or stat.get("ISO"))
    if iso is None:
        slg = _safe_float(stat.get("slg"))
        avg = _safe_float(stat.get("avg"))
        if slg is not None and avg is not None:
            iso = slg - avg
    return woba, wrc, iso


def fetch_savant_batter_expected_stats(season: int) -> Optional[pd.DataFrame]:
    """
    Season-level expected stats from Baseball Savant (MLBAM player_id keyed).
    Returns columns: player_id, Savant_PA, wOBA, xwOBA, xBA, xSLG,
    Batter_Advanced_Source, Advanced_Hitting_Available.
    None = HTTP/parse failure (caller should not crash).
    Empty DataFrame = request OK but no rows.
    """
    params = {
        "type": "batter",
        "year": season,
        "season": season,
        "csv": "true",
    }
    try:
        r = session.get(SAVANT_EXPECTED_BATTER_URL, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            warn(f"Savant expected_statistics HTTP {r.status_code} for season={season}")
            return None
        raw = pd.read_csv(io.BytesIO(r.content))
    except Exception as e:
        warn(f"Savant expected_statistics request/parse failed (season={season}): {e}")
        return None

    if raw is None or not len(raw):
        return pd.DataFrame(
            columns=[
                "player_id",
                "Savant_PA",
                "wOBA",
                "xwOBA",
                "xBA",
                "xSLG",
                "Batter_Advanced_Source",
                "Advanced_Hitting_Available",
            ]
        )

    lc = {str(c).strip().lower(): c for c in raw.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            k = n.lower()
            if k in lc:
                return lc[k]
        return None

    c_pid = pick("player_id")
    if not c_pid:
        warn("Savant CSV missing player_id column; skipping advanced merge.")
        return None

    c_pa = pick("pa")
    c_woba = pick("woba")
    c_xwoba = pick("est_woba")
    c_xba = pick("est_ba")
    c_xslg = pick("est_slg")

    out = pd.DataFrame()
    out["player_id"] = pd.to_numeric(raw[c_pid], errors="coerce")
    _nan_col = float("nan")
    out["Savant_PA"] = (
        pd.to_numeric(raw[c_pa], errors="coerce") if c_pa else _nan_col
    )
    out["wOBA"] = (
        pd.to_numeric(raw[c_woba], errors="coerce") if c_woba else _nan_col
    )
    out["xwOBA"] = (
        pd.to_numeric(raw[c_xwoba], errors="coerce") if c_xwoba else _nan_col
    )
    out["xBA"] = pd.to_numeric(raw[c_xba], errors="coerce") if c_xba else _nan_col
    out["xSLG"] = (
        pd.to_numeric(raw[c_xslg], errors="coerce") if c_xslg else _nan_col
    )

    out = out.dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype(int)
    out = out.drop_duplicates(subset=["player_id"], keep="first")

    stat_present = (
        out["wOBA"].notna()
        | out["xwOBA"].notna()
        | out["xBA"].notna()
        | out["xSLG"].notna()
    )
    out["Batter_Advanced_Source"] = "baseball_savant"
    out["Advanced_Hitting_Available"] = stat_present
    out.loc[~stat_present, "Batter_Advanced_Source"] = ""

    return out


@dataclass
class PlayerRow:
    player_id: int
    name: str
    team_name: str
    pos: str
    season: int
    # core totals (MLB naming)
    pa: Optional[int] = None
    ab: Optional[int] = None
    h: Optional[int] = None
    hr: Optional[int] = None
    bb: Optional[int] = None
    so: Optional[int] = None
    avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    ops: Optional[float] = None
    # vsR / vsL (will fill later or preserve)
    vsR_PA: Optional[int] = None
    vsR_OBP: Optional[float] = None
    vsR_SLG: Optional[float] = None
    vsR_OPS: Optional[float] = None
    vsL_PA: Optional[int] = None
    vsL_OBP: Optional[float] = None
    vsL_SLG: Optional[float] = None
    vsL_OPS: Optional[float] = None
    # Advanced platoon stats — CSV columns vsR_wRC+ / vsL_wRC+ (mapped from *_wrc_plus below)
    vsR_wOBA: Optional[float] = None
    vsL_wOBA: Optional[float] = None
    vsR_wrc_plus: Optional[float] = None
    vsL_wrc_plus: Optional[float] = None
    vsR_ISO: Optional[float] = None
    vsL_ISO: Optional[float] = None


# -----------------------
# MLB fetchers
# -----------------------
def fetch_teams() -> List[dict]:
    params = {"sportId": 1}  # MLB
    data = get_json(f"{MLB_BASE}/api/v1/teams", params=params, label="teams")
    teams = (data or {}).get("teams", [])
    return teams


def fetch_team_active_roster(team_id: int) -> List[dict]:
    data = get_json(f"{MLB_BASE}/api/v1/teams/{team_id}/roster/Active", label=f"roster team={team_id}")
    return (data or {}).get("roster", []) or []


def fetch_person_stats_overall(player_id: int, season: int) -> Optional[dict]:
    # people/{id}?hydrate=stats(type=season,group=hitting,season=YYYY,gameType=R)
    params = {
        "hydrate": f"stats(group=hitting,type=season,season={season},gameType=R)"
    }
    data = get_json(f"{MLB_BASE}/api/v1/people/{player_id}", params=params, label=f"overall stats pid={player_id}")
    if not data or "people" not in data or not data["people"]:
        return None

    person = data["people"][0]
    stats = person.get("stats") or []
    for block in stats:
        if block.get("group", {}).get("displayName") == "hitting" and block.get("type", {}).get("displayName") == "season":
            splits = block.get("splits") or []
            if splits:
                split0 = splits[0] or {}
                stat = dict(split0.get("stat") or {})
                team = split0.get("team") or {}
                stat["_team_id"] = team.get("id")
                stat["_team_name"] = team.get("name")
                return stat
    return None


def fetch_person_splits_vs_pitching(player_id: int, season: int) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Platoon splits: GET /people/{id}/stats with stats=statSplits (not vsPitching hydrate; that stat type
    is invalid for hitting and returns empty stats). Situation codes vr = PA vs RHP, vl = PA vs LHP
    (MLB situation metadata). Returns (vsR dict, vsL dict) with keys like plateAppearances, obp, slg, ops.
    """
    # statSplits + sitCodes vr,vl — valid hitting platoon splits (vs RHP / vs LHP as batter).
    params = {
        "stats": "statSplits",
        "group": "hitting",
        "season": season,
        "gameType": "R",
        "sportIds": 1,
        "sitCodes": "vr,vl",
    }
    data = get_json(
        f"{MLB_BASE}/api/v1/people/{player_id}/stats",
        params=params,
        label=f"statSplits platoon pid={player_id}",
    )
    if not data:
        return None, None

    stats = data.get("stats") or []
    vsR = None
    vsL = None
    for block in stats:
        if block.get("group", {}).get("displayName") != "hitting":
            continue
        if block.get("type", {}).get("displayName") != "statSplits":
            continue
        for sp in block.get("splits") or []:
            stat = dict(sp.get("stat") or {})
            team = sp.get("team") or {}
            stat["_team_id"] = team.get("id")
            stat["_team_name"] = team.get("name")
            split_meta = sp.get("split")
            code = None
            if isinstance(split_meta, dict):
                code = (split_meta.get("code") or "").strip().lower()
            elif isinstance(split_meta, str):
                s = split_meta.strip().upper()
                if s == "R" or "RHP" in s:
                    code = "vr"
                elif s == "L" or "LHP" in s:
                    code = "vl"
            hand = sp.get("pitcherThrows") or (sp.get("opponentHand") or {}).get("code")
            if not code and hand:
                code = "vr" if str(hand).upper().startswith("R") else "vl" if str(hand).upper().startswith("L") else None
            if code == "vr":
                vsR = stat
            elif code == "vl":
                vsL = stat

    return vsR, vsL


# -----------------------
# Main ETL
# -----------------------
def collect_hitters(season: int) -> pd.DataFrame:
    step("Fetching MLB teams…")
    teams = fetch_teams()
    if not teams:
        raise RuntimeError("Could not fetch MLB teams")

    rows: List[PlayerRow] = []
    total_players = 0

    step("Collecting active rosters and season stats for hitters…")
    for t in teams:
        team_id = t.get("id")
        team_name = t.get("name")
        if not team_id:
            continue

        roster = fetch_team_active_roster(team_id)
        for r in roster:
            person = r.get("person") or {}
            pos = (r.get("position") or {}).get("abbreviation") or ""
            pid = person.get("id")
            name = person.get("fullName") or ""

            if pos.upper() == "P":
                continue

            total_players += 1
            core = fetch_person_stats_overall(pid, season) or {}
            # Prefer the season-stats team as the authoritative identity signal; if missing,
            # fall back to the split response before accepting the roster membership.
            vsR, vsL = fetch_person_splits_vs_pitching(pid, season)
            auth_team_id = core.get("_team_id")
            auth_team_name = core.get("_team_name") or ""
            if auth_team_id is None:
                split_team = (vsR or vsL) or {}
                auth_team_id = split_team.get("_team_id")
                auth_team_name = split_team.get("_team_name") or auth_team_name
            if auth_team_id is not None:
                try:
                    auth_team_mismatch = int(auth_team_id) != int(team_id)
                except Exception:
                    auth_team_mismatch = str(auth_team_id) != str(team_id)
            else:
                auth_team_mismatch = bool(auth_team_name) and norm_key(auth_team_name) != norm_key(team_name or "")
            if auth_team_mismatch:
                warn(
                    f"Skipping hitter with roster/team mismatch: roster={team_name} ({team_id}) "
                    f"stats={auth_team_name or 'unknown'} ({auth_team_id}) "
                    f"player={name} ({pid})"
                )
                continue

            row = PlayerRow(
                player_id=pid,
                name=name,
                team_name=team_name or "",
                pos=pos or "",
                season=season,
                pa=int(core.get("plateAppearances") or 0) or None,
                ab=int(core.get("atBats") or 0) or None,
                h=int(core.get("hits") or 0) or None,
                hr=int(core.get("homeRuns") or 0) or None,
                bb=int(core.get("baseOnBalls") or 0) or None,
                so=int(core.get("strikeOuts") or 0) or None,
                avg=_safe_float(core.get("avg")),
                obp=_safe_float(core.get("obp")),
                slg=_safe_float(core.get("slg")),
                ops=_safe_float(core.get("ops")),
            )

            # Try vs-hand splits
            if vsR:
                row.vsR_PA = _safe_int(vsR.get("plateAppearances"))
                row.vsR_OBP = _safe_float(vsR.get("obp"))
                row.vsR_SLG = _safe_float(vsR.get("slg"))
                row.vsR_OPS = _safe_float(vsR.get("ops"))
                rw, rrw, riso = _hitting_split_advanced(vsR)
                row.vsR_wOBA = rw
                row.vsR_wrc_plus = rrw
                row.vsR_ISO = riso
            if vsL:
                row.vsL_PA = _safe_int(vsL.get("plateAppearances"))
                row.vsL_OBP = _safe_float(vsL.get("obp"))
                row.vsL_SLG = _safe_float(vsL.get("slg"))
                row.vsL_OPS = _safe_float(vsL.get("ops"))
                lw, lrw, liso = _hitting_split_advanced(vsL)
                row.vsL_wOBA = lw
                row.vsL_wrc_plus = lrw
                row.vsL_ISO = liso

            rows.append(row)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.rename(columns={"vsR_wrc_plus": "vsR_wRC+", "vsL_wrc_plus": "vsL_wRC+"})
    ok(f"Collected {len(df)} hitters from {len(teams)} teams (scanned {total_players} active roster slots).")
    return df


# -----------------------
# Preserve last-good splits if MLB didn't return them
# -----------------------
def preserve_last_good_splits(current: pd.DataFrame) -> pd.DataFrame:
    """
    If some vsR_/vsL_ columns are missing/empty for players, fill from previous CSV (if present).
    Align by player_id when possible, else normalized name. Handles duplicate keys safely.
    """
    if not DATA_FILE.exists():
        warn("No previous batter_stats.csv to preserve splits from.")
        return current

    try:
        prev = pd.read_csv(DATA_FILE)
    except Exception as e:
        warn(f"Failed reading previous {DATA_FILE}: {e}")
        return current

    # Build stable key: prefer player_id, else normalized name
    def with_key(df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
        df = df.copy()
        df["__key__"] = df.apply(
            lambda r: f"{int(r.get('player_id'))}" if pd.notna(r.get("player_id")) else norm_key(str(r.get(name_col))),
            axis=1,
        )
        return df

    cur = with_key(current)
    prv = with_key(prev)

    # Drop duplicate keys deterministically
    cur = cur[~cur["__key__"].duplicated(keep="first")].set_index("__key__", drop=False)
    prv = prv[~prv["__key__"].duplicated(keep="last")].set_index("__key__", drop=False)

    split_cols = [c for c in prv.columns if c.startswith(SPLIT_PREFIXES)]
    if not split_cols:
        warn("Previous file has no vsR_/vsL_ columns to preserve.")
        return current

    # Align by key; rows missing in prv will be NaN
    restored = prv.reindex(cur.index)[split_cols]

    # Fill only missing values in current with restored ones
    for c in split_cols:
        if c in cur.columns:
            cur[c] = cur[c].where(cur[c].notna(), restored[c])
        else:
            cur[c] = restored[c]

    ok(f"Preserved vs-splits from previous file for missing players/fields ({len(split_cols)} columns).")
    return cur.reset_index(drop=True)


# -----------------------
# Entry
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON, help="Season year, e.g. 2025")
    args = parser.parse_args()
    season = args.season

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    step(f"Updating MLB batter stats (season={season}) without FanGraphs…")
    df = collect_hitters(season)

    # Savant season-level advanced (merge only; core/batters scoring unchanged)
    savant_df = fetch_savant_batter_expected_stats(season)
    savant_value_cols = ["Savant_PA", "wOBA", "xwOBA", "xBA", "xSLG"]
    if savant_df is None:
        warn(
            "Savant advanced stats fetch failed; advanced columns left blank, "
            "Batter_Data_Quality_Flag=savant_fetch_failed."
        )
        for c in savant_value_cols:
            df[c] = None
        df["Batter_Advanced_Source"] = ""
        df["Advanced_Hitting_Available"] = False
        df["Batter_Data_Quality_Flag"] = "savant_fetch_failed"
    else:
        df = df.merge(savant_df, on="player_id", how="left")
        for c in savant_value_cols:
            if c not in df.columns:
                df[c] = None
        df["Batter_Advanced_Source"] = df["Batter_Advanced_Source"].fillna("")
        df["Advanced_Hitting_Available"] = (
            df["Advanced_Hitting_Available"].fillna(False).astype(bool)
        )
        df["Batter_Data_Quality_Flag"] = ""
        df.loc[~df["Advanced_Hitting_Available"], "Batter_Data_Quality_Flag"] = (
            "missing_savant_advanced"
        )

    # Expected columns
    base_cols = [
        "player_id", "name", "team_name", "pos", "season",
        "pa", "ab", "h", "hr", "bb", "so",
        "avg", "obp", "slg", "ops",
    ]
    advanced_cols = [
        "Savant_PA",
        "wOBA",
        "xwOBA",
        "xBA",
        "xSLG",
        "Batter_Advanced_Source",
        "Advanced_Hitting_Available",
        "Batter_Data_Quality_Flag",
    ]
    split_cols = [
        "vsR_PA", "vsR_OBP", "vsR_SLG", "vsR_OPS",
        "vsL_PA", "vsL_OBP", "vsL_SLG", "vsL_OPS",
        "vsR_wOBA",
        "vsL_wOBA",
        "vsR_wRC+",
        "vsL_wRC+",
        "vsR_ISO",
        "vsL_ISO",
    ]
    # Ensure all expected split columns exist
    for c in split_cols:
        if c not in df.columns:
            df[c] = None

    # If splits are sparse, preserve last good ones
    try:
        missing_rate = df[split_cols].isna().mean().mean()
    except Exception:
        missing_rate = 1.0

    if pd.isna(missing_rate) or missing_rate > 0.25:
        warn(f"Detected {round((missing_rate or 0)*100,2)}% missing in vs-splits; preserving last-good splits where possible…")
        try:
            df = preserve_last_good_splits(df)
        except Exception as e:
            warn(f"Split preservation failed (continuing without backfill): {e}")

    # Reorder nicely
    ordered_cols = base_cols + advanced_cols + split_cols + [
        c for c in df.columns if c not in base_cols + advanced_cols + split_cols
    ]
    df = df[ordered_cols]

    # Stable sort
    df["__key__"] = df.apply(
        lambda r: f"{r.get('player_id')}" if pd.notna(r.get("player_id")) else norm_key(str(r.get("name"))),
        axis=1,
    )
    df = df.sort_values(["team_name", "pos", "__key__"]).drop(columns="__key__", errors="ignore")

    df.to_csv(DATA_FILE, index=False)
    ok(f"Saved {len(df)} hitters → {DATA_FILE}")

    # Small summary
    n_with_vsR = int(df["vsR_OPS"].notna().sum())
    n_with_vsL = int(df["vsL_OPS"].notna().sum())
    n_vsR_wrc = int(df["vsR_wRC+"].notna().sum()) if "vsR_wRC+" in df.columns else 0
    n_vsR_woba = int(df["vsR_wOBA"].notna().sum()) if "vsR_wOBA" in df.columns else 0
    log(f"vsR_OPS available: {n_with_vsR} | vsL_OPS available: {n_with_vsL}")
    log(f"vsR_wRC+ available: {n_vsR_wrc} | vsR_wOBA available: {n_vsR_woba} (lineup scoring columns)")
    n_adv = int(df["Advanced_Hitting_Available"].fillna(False).astype(bool).sum())
    log(f"Savant advanced columns populated (Advanced_Hitting_Available): {n_adv}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        warn("Interrupted.")
        sys.exit(130)
