#!/usr/bin/env python3
"""
Build pitcher strikeout-rate stats for K props.

Reads real MLB StatsAPI season pitching strikeout fields and writes a
standalone data/pitcher_k_stats.csv without changing production pitcher_stats.csv.
"""

import os
from datetime import datetime, timezone
from urllib.parse import urlencode

import pandas as pd
import requests


STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
OUTFILE = "data/pitcher_k_stats.csv"
SOURCE_LABEL = "mlb_statsapi_pitching"
DEFAULT_SEASON = datetime.now().year


def ip_to_float(ip_raw) -> float | None:
    """Parse MLB innings strings like 4.2 as 4 + 2/3, not 4.2 decimal."""
    if ip_raw in (None, ""):
        return None
    try:
        s = str(ip_raw).strip()
        if not s:
            return None
        if "." in s:
            whole, frac = s.split(".", 1)
        else:
            whole, frac = s, "0"
        frac_map = {"0": 0.0, "1": 1.0 / 3.0, "2": 2.0 / 3.0}
        if frac not in frac_map:
            return None
        return float(int(whole or 0)) + frac_map[frac]
    except (TypeError, ValueError):
        return None


def _float_or_none(val) -> float | None:
    if val in (None, ""):
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def derive_starter_workload_fields(ip, games_started, games_pitched) -> dict:
    """
    Canonical starter-workload classification for pitcher_k_stats.csv.

    Raw MLB StatsAPI fields are total season stats. IP / Games_Started is only
    a valid starter-workload proxy for clean starter profiles where every
    appearance is a start. Mixed-role pitchers keep raw fields but do not receive
    Expected_Starter_IP, preventing downstream consumers from deriving impossible
    workload values.
    """
    ip_f = _float_or_none(ip)
    gs_f = _float_or_none(games_started)
    gp_f = _float_or_none(games_pitched)

    out = {
        "Raw_IP_Per_Start": pd.NA,
        "Expected_Starter_IP": pd.NA,
        "Expected_Starter_IP_Source": "",
        "Starter_Workload_Profile": "invalid_workload",
        "Workload_Eligible": False,
    }

    if ip_f is None or gs_f is None or gp_f is None or ip_f <= 0 or gs_f < 0 or gp_f <= 0:
        return out

    if gs_f <= 0:
        out["Starter_Workload_Profile"] = "relief_only"
        return out

    raw_ip_per_start = ip_f / gs_f
    out["Raw_IP_Per_Start"] = round(float(raw_ip_per_start), 3)

    if abs(gp_f - gs_f) > 0.001:
        out["Starter_Workload_Profile"] = "mixed_role"
        return out

    if raw_ip_per_start < 4.0 or raw_ip_per_start > 7.5:
        out["Starter_Workload_Profile"] = "clean_starter_raw_out_of_range"
        return out

    out["Starter_Workload_Profile"] = "clean_starter"
    out["Expected_Starter_IP"] = round(float(max(4.0, min(6.5, raw_ip_per_start))), 3)
    out["Expected_Starter_IP_Source"] = "mlb_statsapi_ip_per_start_clean_starter"
    out["Workload_Eligible"] = True
    return out

def fetch_pitching_splits(season: int = DEFAULT_SEASON) -> list:
    params = {
        "stats": "season",
        "group": "pitching",
        "season": season,
        "gameType": "R",
        "limit": 1000,
        "playerPool": "ALL_CURRENT",
    }
    headers = {"User-Agent": "over-gang-mlb-ai/1.0"}
    print(f"Fetching MLB StatsAPI pitching K stats: {STATS_URL}?{urlencode(params)}")
    resp = requests.get(STATS_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    stats_list = body.get("stats") if isinstance(body, dict) else None
    if not isinstance(stats_list, list) or not stats_list:
        raise ValueError("MLB StatsAPI returned no stats array.")
    splits = stats_list[0].get("splits") if isinstance(stats_list[0], dict) else []
    if not isinstance(splits, list):
        raise ValueError("MLB StatsAPI returned no pitching splits.")
    return splits


def build_pitcher_k_stats(outfile: str = OUTFILE, season: int = DEFAULT_SEASON) -> pd.DataFrame:
    rows = []
    updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for split in fetch_pitching_splits(season):
        if not isinstance(split, dict):
            continue
        player = split.get("player") or {}
        stat = split.get("stat") or {}
        name = (player.get("fullName") or "").strip()
        if not name:
            continue
        try:
            mlb_id = int(player.get("id"))
        except (TypeError, ValueError):
            continue

        ip = ip_to_float(stat.get("inningsPitched"))
        so = _float_or_none(stat.get("strikeOuts"))
        k9 = _float_or_none(stat.get("strikeoutsPer9Inn"))
        if k9 is None and so is not None and ip is not None and ip > 0:
            k9 = so / ip * 9.0
        if k9 is None:
            continue

        games_started = _float_or_none(stat.get("gamesStarted"))
        games_pitched = _float_or_none(stat.get("gamesPitched"))
        workload = derive_starter_workload_fields(ip, games_started, games_pitched)

        rows.append({
            "mlb_id": mlb_id,
            "Name": name,
            "IP": round(ip, 3) if ip is not None else pd.NA,
            "SO": int(so) if so is not None else pd.NA,
            "K9": round(float(k9), 2),
            "Batters_Faced": _float_or_none(stat.get("battersFaced")),
            "Games_Started": games_started,
            "Games_Pitched": games_pitched,
            "Raw_IP_Per_Start": workload["Raw_IP_Per_Start"],
            "Expected_Starter_IP": workload["Expected_Starter_IP"],
            "Expected_Starter_IP_Source": workload["Expected_Starter_IP_Source"],
            "Starter_Workload_Profile": workload["Starter_Workload_Profile"],
            "Workload_Eligible": workload["Workload_Eligible"],
            "ERA": _float_or_none(stat.get("era")),
            "WHIP": _float_or_none(stat.get("whip")),
            "Source": SOURCE_LABEL,
            "Updated_At": updated_at,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No pitcher K rows produced from MLB StatsAPI.")
    out = out[out["Name"].astype(str).str.strip().ne("") & out["K9"].notna()].copy()

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
