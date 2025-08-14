#!/usr/bin/env python3
# scripts/update_bullpen.py
# Build team bullpen stats (relief-only) from the MLB Stats API + Baseball Savant xERA.
# Outputs: data/bullpen_stats.csv with columns:
# Team,ERA,IP,WHIP,xERA,Relievers,IP_Week
#
# xERA sources (in order):
#   1) Baseball Savant CSV map by MLBAM player_id
#   2) data/pitcher_stats.csv map by normalized name (from your DataManager updater)

import os
import io
import re
import math
import time
import json
import logging
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from unidecode import unidecode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SEASON = datetime.now().year
OUTFILE = "data/bullpen_stats.csv"
PITCHER_STATS_FILE = "data/pitcher_stats.csv"  # built by your DataManager.update_pitcher_stats()
BASE = "https://statsapi.mlb.com/api/v1"
UA = {"User-Agent": "over-gang-mlb-ai/1.3"}
SLEEP = 0.12  # polite throttle between requests

# Tunables for deriving xERA if Savant CSV lacks it
L_ERA = float(os.getenv("OGP_LEAGUE_ERA", "4.25"))
L_WOBA = float(os.getenv("OGP_LEAGUE_WOBA", "0.310"))
WOBA_TO_ERA_SLOPE = float(os.getenv("OGP_WOBA_TO_ERA_SLOPE", "20.0"))
# rule of thumb: ~0.010 wOBA ≈ 0.20 ERA ⇒ slope ≈ 20


# -----------------------------
# Name normalization (matches your DataManager)
# -----------------------------
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = unidecode(name)
    name = re.sub(r"[^a-zA-Z ]", "", name.lower().strip())
    for suffix in ["jr", "sr", "ii", "iii", "iv"]:
        name = re.sub(fr"\s*{suffix}$", "", name)
    nick = {
        "nick": "nicholas", "nate": "nathan", "mike": "michael",
        "matt": "matthew", "jim": "james", "chris": "christopher",
        "joe": "joseph", "andy": "andrew", "dan": "daniel",
        "tom": "thomas", "tim": "timothy", "bob": "robert",
        "rob": "robert", "dave": "david", "will": "william",
        "alex": "alexander", "josh": "joshua", "sam": "samuel"
    }
    parts = [p for p in name.split() if len(p) > 0]
    return " ".join([nick.get(p, p) for p in parts])


# -----------------------------
# HTTP helpers
# -----------------------------
def get_text(url, params=None, tries=3):
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, headers=UA, timeout=35)
            if r.status_code == 200:
                return r.text
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last_err}")


def get_json(url, params=None, tries=3):
    txt = get_text(url, params=params, tries=tries)
    try:
        return requests.models.complexjson.loads(txt)
    except Exception:
        r = requests.get(url, params=params or {}, headers=UA, timeout=35)
        r.raise_for_status()
        return r.json()


# -----------------------------
# Savant xERA fetch
# -----------------------------
def _pick_col(cols, *candidates):
    """Choose a column by normalized name or prefix matches."""
    norm = {str(c).lower().replace(" ", "").replace("_", ""): c for c in cols}
    wants = [c.lower().replace(" ", "").replace("_", "") for c in candidates]
    for k, original in norm.items():
        if k in wants:
            return original
    for c in cols:
        k = str(c).lower().replace(" ", "").replace("_", "")
        if any(k.startswith(w) for w in wants):
            return c
    return None


def fetch_savant_xera_map(year: int) -> dict[int, float]:
    """
    Return {mlbam_id: xERA} using Savant CSV.
    If xERA is missing, derive from est_wOBA.
    """
    csv_urls = [
        f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitchers&year={year}&season={year}&csv=true",
        f"https://baseballsavant.mlb.com/leaderboard/expected_statistics?type=pitchers&year={year}&csv=true",
    ]
    last_err = None
    df = None

    for i, url in enumerate(csv_urls, 1):
        try:
            csv_text = get_text(url)
            df = pd.read_csv(io.StringIO(csv_text))
            if not df.empty:
                break
        except Exception as e:
            last_err = f"CSV attempt {i} failed: {e}"
            df = None

    if df is None or df.empty:
        raise RuntimeError(f"Savant CSV empty. Last error: {last_err}")

    id_candidates = [
        "player_id", "playerid", "player id",
        "pitcher_id", "mlbamid", "mlbam_id", "mlb_id"
    ]
    pid_col = _pick_col(df.columns, *id_candidates)
    if not pid_col:
        raise RuntimeError(f"Savant CSV missing id column (looked for: {id_candidates})")

    xera_col    = _pick_col(df.columns, "xERA", "xera", "expectedera", "exera")
    estwoba_col = _pick_col(df.columns, "est_wOBA", "estwoba", "expectedwoba", "xwoba")

    df[pid_col] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[pid_col])

    xera_map: dict[int, float] = {}

    if xera_col and xera_col in df.columns:
        df[xera_col] = pd.to_numeric(df[xera_col], errors="coerce")
        for _, row in df.dropna(subset=[xera_col]).iterrows():
            xera_map[int(row[pid_col])] = float(row[xera_col])

    if estwoba_col and estwoba_col in df.columns:
        df[estwoba_col] = pd.to_numeric(df[estwoba_col], errors="coerce")
        for _, row in df.iterrows():
            pid = row[pid_col]
            if pd.isna(pid):
                continue
            pid = int(pid)
            if pid in xera_map:
                continue
            w = row[estwoba_col]
            if pd.notna(w):
                xera = L_ERA + WOBA_TO_ERA_SLOPE * (float(w) - L_WOBA)
                xera = max(2.0, min(8.5, xera))
                xera_map[pid] = float(round(xera, 2))

    logging.info(f"✅ Savant xERA map size: {len(xera_map)}")
    return xera_map


# -----------------------------
# MLB data fetch
# -----------------------------
def teams():
    """Active MLB teams."""
    data = get_json(f"{BASE}/teams", {"sportId": 1, "activeStatus": "Y"})
    return [{"id": t["id"], "name": t["name"].strip()} for t in data.get("teams", [])]


def team_pitchers(team_id):
    """Active roster pitcher IDs + names for a team."""
    data = get_json(f"{BASE}/teams/{team_id}/roster", {"rosterType": "active"})
    out = []
    for it in data.get("roster", []):
        pos = (it.get("position") or {}).get("abbreviation")
        if pos == "P":
            person = it.get("person") or {}
            pid = person.get("id")
            name = (person.get("fullName") or "").strip()
            if pid:
                out.append({"id": int(pid), "name": name})
    time.sleep(SLEEP)
    return out


def gamelog(pid):
    """Pitcher game log (regular season) with pitching group."""
    data = get_json(
        f"{BASE}/people/{pid}/stats",
        {"stats": "gameLog", "group": "pitching", "season": SEASON, "gameType": "R"},
    )
    time.sleep(SLEEP)
    try:
        return data["stats"][0]["splits"]
    except Exception:
        return []


# -----------------------------
# Utilities
# -----------------------------
def ip_to_float(ip_str):
    """Convert innings string like '12.1'/'12.2' to decimal innings."""
    if not ip_str:
        return 0.0
    s = str(ip_str)
    if "." not in s:
        try:
            return float(s)
        except Exception:
            return 0.0
    whole, frac = s.split(".", 1)
    try:
        whole = int(whole or 0)
    except Exception:
        whole = 0
    return whole + {"0": 0.0, "1": 1.0 / 3.0, "2": 2.0 / 3.0}.get(frac, 0.0)


def parse_game_date(d):
    """
    Parse MLB Stats API game date (can be 'YYYY-MM-DD' or ISO).
    Returns a naive date() (no tz).
    """
    if not d:
        return None
    try:
        dt = datetime.fromisoformat(str(d).replace("Z", "+00:00"))
        return dt.date()
    except Exception:
        pass
    try:
        dt = datetime.strptime(str(d), "%Y-%m-%d")
        return dt.date()
    except Exception:
        return None


# -----------------------------
# Load xERA by name from pitcher_stats.csv
# -----------------------------
def load_name_xera_map(path=PITCHER_STATS_FILE) -> dict[str, float]:
    """
    Load your saved pitcher stats (Name,xERA,...) and return {normalized_name: xERA}.
    """
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if "Name" not in df.columns or "xERA" not in df.columns:
            return {}
        df["Name"] = df["Name"].astype(str).apply(normalize_name)
        df["xERA"] = pd.to_numeric(df["xERA"], errors="coerce")
        df = df.dropna(subset=["Name", "xERA"])
        # keep the last occurrence if duplicates
        df = df.drop_duplicates(subset=["Name"], keep="last")
        mp = {row["Name"]: float(row["xERA"]) for _, row in df.iterrows()}
        logging.info(f"✅ Name→xERA map size from pitcher_stats.csv: {len(mp)}")
        return mp
    except Exception as e:
        logging.warning(f"Failed to read pitcher_stats.csv: {e}")
        return {}


# -----------------------------
# Aggregation
# -----------------------------
def agg_relief(team_name, roster, week_cutoff_date, xera_by_id: dict[int, float] | None, xera_by_name: dict[str, float]):
    """
    Aggregate TEAM bullpen stats using only relief appearances (gamesStarted == 0).
    Returns dict with Team, ERA, IP, WHIP, xERA, Relievers, IP_Week.
    xERA is IP-weighted avg of relief-only IP. Lookup order: xera_by_id, then xera_by_name.
    """
    ER = H = BB = IP = 0.0
    relievers = set()
    IP_week = 0.0

    xera_num = 0.0
    xera_den = 0.0

    savant_hits = 0
    name_hits = 0

    for p in roster:
        pid = p["id"]
        pname = normalize_name(p.get("name", ""))

        splits = gamelog(pid)
        if not splits:
            continue

        relief_ip_for_pitcher = 0.0

        for s in splits:
            st = s.get("stat", {}) or {}
            gs = st.get("gamesStarted")
            try:
                gs = int(gs) if gs is not None else 0
            except Exception:
                gs = 0
            if gs != 0:
                continue  # skip starts

            ip = ip_to_float(st.get("inningsPitched"))
            ER += float(st.get("earnedRuns") or 0)
            H  += float(st.get("hits") or 0)
            BB += float(st.get("baseOnBalls") or 0)
            IP += ip
            relief_ip_for_pitcher += ip

            gdate = parse_game_date(s.get("date"))
            if gdate and gdate >= week_cutoff_date:
                IP_week += ip

        if relief_ip_for_pitcher > 0:
            relievers.add(pid)
            x = None
            if xera_by_id and pid in xera_by_id:
                x = float(xera_by_id[pid])
                savant_hits += 1
            elif pname and pname in xera_by_name:
                x = float(xera_by_name[pname])
                name_hits += 1

            if x is not None:
                xera_num += x * relief_ip_for_pitcher
                xera_den += relief_ip_for_pitcher

    ERA = (ER * 9 / IP) if IP > 0 else float("nan")
    WHIP = ((BB + H) / IP) if IP > 0 else float("nan")

    team_xera = ""
    if xera_den > 0:
        team_xera = round(xera_num / xera_den, 2)

    logging.debug(f"[xERA] {team_name}: id-matches={savant_hits}, name-matches={name_hits}, relievers={len(relievers)}")

    return {
        "Team": team_name,
        "ERA": round(ERA, 2) if not math.isnan(ERA) else "",
        "IP": round(IP, 1),
        "WHIP": round(WHIP, 2) if not math.isnan(WHIP) else "",
        "xERA": team_xera,  # blank if no matches
        "Relievers": len(relievers),
        "IP_Week": round(IP_week, 2),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    logging.info("🔄 Building bullpen stats from MLB Stats API…")

    # Savant xERA map (by MLBAM id)
    try:
        xera_by_id = fetch_savant_xera_map(SEASON)
    except Exception as e:
        logging.warning(f"⚠️ Savant xERA unavailable: {e}")
        xera_by_id = None

    # Name→xERA map from your saved pitcher_stats.csv (DataManager output)
    xera_by_name = load_name_xera_map(PITCHER_STATS_FILE)

    # Compare by DATE only (avoid tz-naive vs tz-aware issues)
    week_cutoff_date = (datetime.now(timezone.utc) - timedelta(days=7)).date()

    rows = []
    for t in teams():
        try:
            roster = team_pitchers(t["id"])
            if not roster:
                logging.info(f"{t['name']}: no active pitchers")
                continue
            agg = agg_relief(t["name"], roster, week_cutoff_date, xera_by_id, xera_by_name)
            rows.append(agg)
            logging.info(f"✔ {t['name']}")
        except Exception as e:
            logging.error(f"❌ {t['name']}: {e}")

    if not rows:
        raise RuntimeError("No bullpen rows produced")

    df = pd.DataFrame(rows)

    # League Average row (IP-weighted ERA/WHIP; IP-weighted xERA where available)
    try:
        ip_sum = df["IP"].sum()
        era_la = (pd.to_numeric(df["ERA"], errors="coerce").fillna(0) * df["IP"]).sum() / max(ip_sum, 1)
        whip_la = (pd.to_numeric(df["WHIP"], errors="coerce").fillna(0) * df["IP"]).sum() / max(ip_sum, 1)

        mask_x = pd.to_numeric(df["xERA"], errors="coerce").notna()
        ip_sum_x = df.loc[mask_x, "IP"].sum()
        xera_la = ""
        if ip_sum_x > 0:
            xera_la = round((pd.to_numeric(df.loc[mask_x, "xERA"], errors="coerce") * df.loc[mask_x, "IP"]).sum() / ip_sum_x, 2)

        la = pd.DataFrame([{
            "Team": "League Average",
            "ERA": round(era_la, 2),
            "IP": round(df["IP"].mean(), 1),
            "WHIP": round(whip_la, 2),
            "xERA": xera_la,
            "Relievers": int(df["Relievers"].mean()),
            "IP_Week": round(df["IP_Week"].mean(), 2),
        }])
        df = pd.concat([la, df], ignore_index=True)
    except Exception as e:
        logging.warning(f"League Average row failed: {e}")

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    df.to_csv(OUTFILE, index=False)
    logging.info(f"✅ Saved {OUTFILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
