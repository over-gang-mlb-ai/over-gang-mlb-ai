"""
OVER GANG MLB PREDICTOR v4.0 — True projection model

Projects expected runs per team (away + home), sums to a game total, then compares
to the Vegas total for edge, confidence, and recommended bet.

Inputs: starter xERA/WHIP, bullpen ERA, bullpen workload (IP_Week vs reliever-expected IP), park factors, velocity
drops, lineup impact, public betting %. Output: projected_total, away_runs, home_runs,
edge vs Vegas, pick (OVER/UNDER), confidence, recommended_units.
"""
import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone, utc
import os
import re
import time
import traceback
from zoneinfo import ZoneInfo
import io
import json
import requests
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from unidecode import unidecode
from rapidfuzz import fuzz, process
from functools import lru_cache
from statsapi import schedule
import numpy as np
import subprocess
import sys
from pathlib import Path

from scrapers.velocity_tracker import VelocityTracker
from core.public_betting_dummy import DUMMY_PUBLIC_BETTING
from core.public_betting_scraper import active_slate_date_mt
from core.public_betting_loader import load_public_betting_data
public_data = load_public_betting_data()
from core.public_betting_loader import split_game_key
from core.ml_predictor import get_team_ml_data, calculate_team_win_probability
from core.public_betting_loader import normalize_team_name
from core.kelly_utils import calculate_kelly_units
from core.odds_api import fetch_mlb_odds, get_game_odds
from core.batters import Batters, LineupImpact, BATTER_DF
from model.data_manager import DataManager
manual_fallback_df = DataManager.load_manual_fallback_pitchers()

# Chadwick register cache for redirect-safe FG id lookup (avoids playerid_lookup 308/empty issues).
_CHADWICK_BASE = "https://raw.githubusercontent.com/chadwickbureau/register/master/data"
_CHADWICK_ID_CACHE = {}

def _resolve_fangraphs_id_from_chadwick(name):
    """Resolve key_fangraphs from Chadwick register shard-by-shard. Returns int or None."""
    parts = (name or "").strip().split()
    if not parts:
        return None
    last = parts[-1]
    first = " ".join(parts[:-1]) if len(parts) > 1 else ""
    last_norm = (unidecode(last).lower().strip() if last else "")
    first_norm = (unidecode(first).lower().strip() if first else "")

    cache_key = (first_norm, last_norm)
    if cache_key in _CHADWICK_ID_CACHE:
        return _CHADWICK_ID_CACHE[cache_key]

    # Load one shard at a time to keep EC2 memory stable.
    for suffix in "0 1 2 3 4 5 6 7 8 9 a b c d e f".split():
        try:
            url = f"{_CHADWICK_BASE}/people-{suffix}.csv"
            r = requests.get(url, timeout=15, allow_redirects=True)
            r.raise_for_status()
            if not r.content:
                continue

            # Read only the columns we need when possible (reduces memory).
            try:
                df = pd.read_csv(
                    io.BytesIO(r.content),
                    usecols=["name_first", "name_last", "key_fangraphs"],
                    low_memory=False,
                )
            except Exception:
                df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
                keep_cols = [c for c in ["name_first", "name_last", "key_fangraphs"] if c in df.columns]
                if len(keep_cols) < 3:
                    continue
                df = df[keep_cols]

            if df.empty or "key_fangraphs" not in df.columns:
                continue

            df["_last_n"] = df["name_last"].fillna("").astype(str).str.strip().apply(lambda s: unidecode(s).lower())
            df["_first_n"] = df["name_first"].fillna("").astype(str).str.strip().apply(lambda s: unidecode(s).lower())

            exact = df[(df["_last_n"] == last_norm) & (df["_first_n"] == first_norm)]
            if exact.empty:
                continue

            with_fg = exact[
                exact["key_fangraphs"].notna() & (exact["key_fangraphs"].astype(str).str.strip() != "")
            ]
            row = with_fg.iloc[0] if not with_fg.empty else exact.iloc[0]

            try:
                k = row["key_fangraphs"]
                if pd.isna(k) or k == "" or (isinstance(k, float) and pd.isna(k)):
                    _CHADWICK_ID_CACHE[cache_key] = None
                    return None
                player_id = int(float(k))
                _CHADWICK_ID_CACHE[cache_key] = player_id
                return player_id
            finally:
                # Encourage memory release for large shard frames.
                try:
                    del df
                except Exception:
                    pass
        except Exception:
            continue

    _CHADWICK_ID_CACHE[cache_key] = None
    return None

# ================================
# 🧰 FanGraphs scrape diagnostics
# ================================
# Monkey-patch DataManager.scrape_fangraphs_pitcher() to print actionable exception details.
# Logging-only: the scrape behavior/return values remain the same.
try:
    if not getattr(DataManager, "_overgang_scrape_debug_patched", False):
        _orig_scrape_fangraphs_pitcher = DataManager.scrape_fangraphs_pitcher

        def _overgang_scrape_fangraphs_pitcher_debug(name):
            # OG_TEST_SCRAPER: isolated direct scrape tests only (not used by live match_pitcher_row).
            if os.getenv("OG_TEST_SCRAPER") != "1":
                return None
            original = name
            normalized = DataManager.normalize_name(name)
            alias_used = globals().get("OG_LAST_ALIAS_USED_FOR_SCRAPE", "?")
            print(
                "[SCRAPE DEBUG] "
                f"original={repr(original)} | normalized={repr(normalized)} | alias_used={alias_used}"
            )
            try:
                # Lookup via Chadwick register with redirect-safe fetch (avoids playerid_lookup 308/empty).
                player_id = _resolve_fangraphs_id_from_chadwick(name)
                if player_id is None:
                    print("❌ No FanGraphs match found (Chadwick register).")
                    return None
                full_name = (normalized or name or "").strip().lower()

                stats_url = f"https://www.fangraphs.com/players/id/{player_id}/stats?position=P"
                dfs = pd.read_html(stats_url)
                stats_df = dfs[0]

                current_year = str(datetime.now().year)
                current_season = stats_df[stats_df["Season"].astype(str).str.startswith(current_year)]
                if current_season.empty:
                    print("⚠️ No stats found for this season.")
                    return None

                row = current_season.iloc[0]
                xera = float(row.get("xERA", 4.50))
                whip = float(row.get("WHIP", 1.30))
                ip = float(row.get("IP", 0.0))
                low_ip = ip < 60

                return {"Name": full_name, "xERA": xera, "WHIP": whip, "IP": ip, "LowIP": low_ip}
            except Exception as e:
                print(
                    "[SCRAPE ERROR] "
                    f"pitcher={repr(original)} | exc_type={type(e).__name__} | exc={repr(e)}"
                )
                return None

        DataManager.scrape_fangraphs_pitcher = staticmethod(_overgang_scrape_fangraphs_pitcher_debug)
        DataManager._overgang_scrape_debug_patched = True
except Exception:
    # If anything goes wrong with patching, fail open (keep existing behavior).
    pass

try:
    BATTER_DF = Batters.load_batter_table()  # reads data/batter_stats.csv
    print(f"✅ Loaded batter table: {len(BATTER_DF)} rows")
except Exception as e:
    print(f"⚠️ Could not load batter table: {e}")
    BATTER_DF = pd.DataFrame()

def safe_get(obj, key, default):
    if isinstance(obj, dict):
        return obj.get(key, default)
    elif hasattr(obj, '__getitem__') and key in obj:
        return obj[key]
    return default

# ================================
# ⚙️ CONFIGURATION
# ================================
TELEGRAM_BOT_TOKEN = '7660295294:AAHakWClywbZP9hdgC5DomgT8EyBa14w-wU'
TELEGRAM_CHAT_ID = '1821580164'
MIN_CONFIDENCE_ALERT = 0.85
# ML side-signal fire: max(home_win_prob, away_win_prob) from calculate_team_win_probability (not gated on O/U totals).
MIN_ML_WIN_PROB_FOR_FIRE = 0.55
# Customer Telegram only (CSV/export unchanged): min confidence to send a message.
TELEGRAM_OU_MIN_CONFIDENCE_PCT = 90.0
TELEGRAM_ML_MIN_CONFIDENCE_PCT = 70.0
DATA_DIR = "data"
ARCHIVE_DIR = "archive"
STATS_FILE = os.path.join(DATA_DIR, "pitcher_stats.csv")
AUTO_UPDATE_DATA = True

# Dynamic IP thresholds
MIN_PITCHER_IP_EARLY = 10
MIN_PITCHER_IP_MID = 20
MIN_PITCHER_IP_LATE = 15

# Thresholds
NAME_MATCH_THRESHOLD = 85
FATIGUE_THRESHOLD = 4.25
VELOCITY_DROP_THRESHOLD = -1.5

# Session-only: schedule probable (lower) → pitcher_stats index key for targeted MLB id + no reg-season-stat cases.
RUNTIME_TARGETED_LOCAL_RECOVERY_ALIASES = {}

# Per run_predictions() run: suppress duplicate [EMPTY BOOK DETAIL] / [LIVE TOTAL CHECK] for same lookup_key.
_LIVE_TOTAL_BLOCKER_DIAG_KEYS_EMITTED = set()


def _build_pitcher_alias_reversed_dict():
    """Variation (lower) -> official (lower); same as match_pitcher_row alias step."""
    alias_file = os.path.join(DATA_DIR, "pitcher_aliases.json")
    reversed_aliases = {}
    if not os.path.exists(alias_file):
        return reversed_aliases
    try:
        with open(alias_file, "r") as f:
            alias_map = json.load(f)
        for official, variations in (alias_map or {}).items():
            if isinstance(variations, list):
                for a in variations:
                    if a:
                        reversed_aliases[a.lower()] = official.lower()
            elif isinstance(variations, str) and variations.strip():
                reversed_aliases[variations.lower()] = official.lower()
    except Exception:
        pass
    return reversed_aliases


def _pitcher_resolvable_locally_without_league_average(
    pitcher_name: str,
    stats_df: pd.DataFrame,
    manual_fallback_df: pd.DataFrame,
    reversed_aliases: dict,
) -> bool:
    """
    True if local resolution would yield real data (not the league-average dict).
    Order: league sentinels, alias, direct CSV index (non-null row), fuzzy+last-name guard, manual_fallback CSV.
    """
    if not pitcher_name or not str(pitcher_name).strip():
        return False
    pn = str(pitcher_name).strip()
    if not pn or pn.strip().upper() == "TBD":
        return False

    alias_key = pn.lower()
    if RUNTIME_TARGETED_LOCAL_RECOVERY_ALIASES and alias_key in RUNTIME_TARGETED_LOCAL_RECOVERY_ALIASES:
        clean_name = DataManager.normalize_name(RUNTIME_TARGETED_LOCAL_RECOVERY_ALIASES[alias_key])
    elif reversed_aliases and alias_key in reversed_aliases:
        clean_name = DataManager.normalize_name(reversed_aliases[alias_key])
    else:
        clean_name = DataManager.normalize_name(pitcher_name)
    if clean_name in {"league avg away", "league avg home"}:
        return True

    if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
        if clean_name in stats_df.index:
            row = stats_df.loc[clean_name]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if not row.isnull().any():
                return True
        choices = stats_df.index.tolist()
        result = process.extractOne(clean_name, choices, scorer=fuzz.WRatio, score_cutoff=NAME_MATCH_THRESHOLD)
        if result:
            best_match, score = result[0], result[1]
            try:
                if clean_name.split()[-1] == best_match.split()[-1]:
                    return True
            except Exception:
                pass

    if manual_fallback_df is not None and isinstance(manual_fallback_df, pd.DataFrame) and not manual_fallback_df.empty:
        if pn.lower() in manual_fallback_df.index:
            return True

    return False


def _local_pitcher_stats_index_key(pitcher_name: str, stats_df: pd.DataFrame, reversed_aliases: dict):
    """
    If pitcher_stats.csv (stats_df index) already has a usable row for this name, return the index key.
    Same alias + fuzzy + last-name rules as _pitcher_resolvable_locally_without_league_average (no manual CSV).
    """
    if not pitcher_name or not str(pitcher_name).strip():
        return None
    pn = str(pitcher_name).strip()
    if pn.upper() == "TBD":
        return None
    if not isinstance(stats_df, pd.DataFrame) or stats_df.empty:
        return None
    alias_key = pn.lower()
    if reversed_aliases and alias_key in reversed_aliases:
        clean_name = DataManager.normalize_name(reversed_aliases[alias_key])
    else:
        clean_name = DataManager.normalize_name(pitcher_name)
    if clean_name in stats_df.index:
        row = stats_df.loc[clean_name]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if not row.isnull().any():
            return clean_name
    choices = stats_df.index.tolist()
    result = process.extractOne(clean_name, choices, scorer=fuzz.WRatio, score_cutoff=NAME_MATCH_THRESHOLD)
    if result:
        best_match, score = result[0], result[1]
        try:
            if clean_name.split()[-1] == best_match.split()[-1]:
                return best_match
        except Exception:
            pass
    return None


def _pitch_row_from_mlb_stat_split(sp: dict):
    """One MLB StatsAPI season pitching split → {mlb_id, Name, ERA, WHIP, IP} or None."""
    if not isinstance(sp, dict):
        return None
    player = sp.get("player") or {}
    st = sp.get("stat") or {}
    pid = player.get("id")
    name_full = (player.get("fullName") or "").strip()
    if pid is None:
        return None
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    era_raw = st.get("era") if st.get("era") is not None else st.get("ERA")
    whip_raw = st.get("whip") if st.get("whip") is not None else st.get("WHIP")
    ip_raw = st.get("inningsPitched") if st.get("inningsPitched") is not None else st.get("ip") or st.get("IP")
    era = None
    if era_raw not in (None, ""):
        try:
            era = float(era_raw)
        except (TypeError, ValueError):
            pass
    whip = None
    if whip_raw not in (None, ""):
        try:
            whip = float(whip_raw)
        except (TypeError, ValueError):
            pass
    ip = None
    if ip_raw is not None and ip_raw != "":
        try:
            if isinstance(ip_raw, (int, float)):
                ip = float(ip_raw)
            else:
                s = str(ip_raw).strip()
                if "." in s:
                    whole, frac = s.split(".", 1)
                else:
                    whole, frac = s, "0"
                frac_dec = {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(frac, 0.0)
                ip = float(whole) + frac_dec
        except Exception:
            try:
                ip = float(ip_raw)
            except Exception:
                pass
    if not any(v is not None for v in (era, whip, ip)):
        return None
    return {"mlb_id": pid, "Name": name_full, "ERA": era, "WHIP": whip, "IP": ip}


def _is_mlb_pitcher_person(p: dict) -> bool:
    pos = p.get("primaryPosition") or {}
    return pos.get("code") == "1" or pos.get("abbreviation") == "P"


def _mlb_targeted_resolve_pitcher_id(probable_name: str):
    """
    Resolve probable display name → (mlb_id, full_name) via MLB StatsAPI people/search?names=...
    Does not use FanGraphs or full-file pitcher refresh.
    """
    if not probable_name or not str(probable_name).strip():
        return None, None
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = "https://statsapi.mlb.com/api/v1/people/search"
    try:
        resp = requests.get(
            search_url, params={"names": str(probable_name).strip()}, headers=headers, timeout=15
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        print(f"[Targeted backfill] MLB id lookup failed for '{probable_name}': {e}")
        return None, None
    people = body.get("people") if isinstance(body, dict) else None
    if not isinstance(people, list) or not people:
        print(f"[Targeted backfill] MLB search returned no people for '{probable_name}'")
        return None, None
    pitchers = [p for p in people if isinstance(p, dict) and _is_mlb_pitcher_person(p)]
    norm_t = DataManager.normalize_name(probable_name)
    if not norm_t:
        return None, None

    def _pick_id_name(candidate_list):
        if len(candidate_list) == 1:
            p = candidate_list[0]
            try:
                return int(p["id"]), (p.get("fullName") or "").strip() or probable_name
            except (KeyError, TypeError, ValueError):
                return None, None
        choices = []
        rows = []
        for p in candidate_list:
            fn = (p.get("fullName") or "").strip()
            if not fn:
                continue
            nn = DataManager.normalize_name(fn)
            if not nn:
                continue
            choices.append(nn)
            rows.append(p)
        if not choices:
            return None, None
        res = process.extractOne(norm_t, choices, scorer=fuzz.WRatio, score_cutoff=NAME_MATCH_THRESHOLD)
        if not res:
            return None, None
        best_nn, _sc = res[0], res[1]
        try:
            idx = res[2] if len(res) > 2 else choices.index(best_nn)
        except (ValueError, IndexError):
            return None, None
        try:
            if norm_t.split()[-1] != best_nn.split()[-1]:
                return None, None
        except Exception:
            return None, None
        p = rows[idx]
        try:
            return int(p["id"]), (p.get("fullName") or "").strip() or probable_name
        except (KeyError, TypeError, ValueError):
            return None, None

    if not pitchers:
        print(
            f"[Targeted backfill] MLB search: no primaryPosition=P in results for '{probable_name}' "
            f"({len(people)} hit(s))"
        )
        return None, None
    return _pick_id_name(pitchers)


def _mlb_targeted_fetch_player_season_pitching(player_id: int, season_year: int):
    """
    GET /people/{id}/stats season pitching for one player. Returns best split row dict or None.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://statsapi.mlb.com/api/v1/people/{int(player_id)}/stats"
    params = {"stats": "season", "group": "pitching", "season": season_year, "gameType": "R"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        body = resp.json()
    except Exception as e:
        print(f"[Targeted backfill] Per-player stats request failed id={player_id} year={season_year}: {e}")
        return None
    stats_list = body.get("stats") if isinstance(body, dict) else None
    if not isinstance(stats_list, list) or not stats_list:
        return None
    splits = stats_list[0].get("splits") if isinstance(stats_list[0], dict) else []
    if not isinstance(splits, list) or not splits:
        return None
    parsed = []
    for sp in splits:
        row = _pitch_row_from_mlb_stat_split(sp)
        if row is not None:
            parsed.append(row)
    if not parsed:
        return None
    # Multiple splits (e.g. traded): keep row with most IP for a single representative line.
    best = max(parsed, key=lambda r: float(r["IP"]) if r.get("IP") is not None else -1.0)
    return best


def _targeted_pitcher_csv_row_from_mlb_row(mlb_row: dict, savant_df: pd.DataFrame, min_ip: float) -> dict:
    """Apply Savant xERA + LowIP (same semantics as update_pitcher_stats merge) for one pitcher."""
    mlb_id = int(mlb_row["mlb_id"])
    display = (mlb_row.get("Name") or "").strip()
    norm_name = DataManager.normalize_name(display)
    era = mlb_row.get("ERA")
    whip = mlb_row.get("WHIP")
    ip_val = mlb_row.get("IP")
    try:
        ip_f = float(ip_val) if ip_val is not None and pd.notna(ip_val) else 0.0
    except (TypeError, ValueError):
        ip_f = 0.0
    xera = np.nan
    if savant_df is not None and not savant_df.empty and "player_id" in savant_df.columns:
        sub = savant_df.loc[savant_df["player_id"] == mlb_id]
        if not sub.empty:
            xera = sub.iloc[0].get("xERA")
    if pd.isna(xera):
        xera = float(era) if era is not None and pd.notna(era) else 4.25
    else:
        try:
            xera = float(xera)
        except (TypeError, ValueError):
            xera = float(era) if era is not None and pd.notna(era) else 4.25
    try:
        whip_f = float(whip) if whip is not None and pd.notna(whip) else 1.30
    except (TypeError, ValueError):
        whip_f = 1.30
    return {
        "Name": norm_name,
        "xERA": float(xera),
        "WHIP": float(whip_f),
        "IP": ip_f,
        "LowIP": bool(ip_f < float(min_ip)),
    }


def _upsert_pitcher_stats_rows(new_rows: list) -> None:
    """Merge rows into data/pitcher_stats.csv without replacing the whole file (preserves SAFE MODE intent)."""
    if not new_rows:
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(STATS_FILE):
        df = pd.read_csv(STATS_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "xERA", "WHIP", "IP", "LowIP"])
    for c in ("Name", "xERA", "WHIP", "IP", "LowIP"):
        if c not in df.columns:
            df[c] = np.nan if c != "Name" else ""
    df["Name"] = df["Name"].astype(str)
    for nr in new_rows:
        norm_key = DataManager.normalize_name(str(nr.get("Name", "")))
        if not norm_key:
            continue
        norms = df["Name"].apply(lambda x: DataManager.normalize_name(str(x)))
        mask = norms == norm_key
        if mask.any():
            ix = int(df.index[mask][0])
            df.at[ix, "Name"] = norm_key
            df.at[ix, "xERA"] = nr["xERA"]
            df.at[ix, "WHIP"] = nr["WHIP"]
            df.at[ix, "IP"] = nr["IP"]
            df.at[ix, "LowIP"] = bool(nr["LowIP"])
        else:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "Name": norm_key,
                                "xERA": nr["xERA"],
                                "WHIP": nr["WHIP"],
                                "IP": nr["IP"],
                                "LowIP": bool(nr["LowIP"]),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    df.to_csv(STATS_FILE, index=False)
    print(f"[Targeted backfill] Wrote {len(new_rows)} pitcher row(s) into {STATS_FILE} (upsert)")


# ================================
# Live predictor: match_pitcher_row without FanGraphs scrape (league-average fallback directly)
# ================================
try:
    if not getattr(DataManager, "_overgang_match_pitcher_no_scrape", False):

        def _overgang_match_pitcher_row_no_scrape(df: pd.DataFrame, pitcher_name: str, alias_log=None):
            """Same resolution order as model.data_manager.DataManager.match_pitcher_row except no scrape step."""
            print(f"🔎 match_pitcher_row() called with: {pitcher_name}")
            if df.empty or not pitcher_name:
                return None

            print(f"📋 DataFrame index sample: {list(df.index)[:10]}")
            print(f"🎯 Trying to match: {pitcher_name} → {DataManager.normalize_name(pitcher_name)}")
            clean_name = DataManager.normalize_name(pitcher_name)
            print(f"🎯 Trying to match: {pitcher_name} → {clean_name}")

            if clean_name in {"league avg away", "league avg home"}:
                if clean_name in df.index:
                    row = df.loc[clean_name]
                    if isinstance(row, dict):
                        row = pd.Series(row, name=clean_name)
                    for k, v in {"xERA": 4.20, "WHIP": 1.30, "IP": 150.0, "LowIP": False}.items():
                        if k not in row.index:
                            row[k] = v
                    return row[["xERA", "WHIP", "IP", "LowIP"]]
                return pd.Series({"xERA": 4.20, "WHIP": 1.30, "IP": 150.0, "LowIP": False}, name=clean_name)

            alias_key = pitcher_name.strip().lower()
            if alias_key in RUNTIME_TARGETED_LOCAL_RECOVERY_ALIASES:
                clean_name = DataManager.normalize_name(RUNTIME_TARGETED_LOCAL_RECOVERY_ALIASES[alias_key])
                print(f"🔁 Targeted local-recovery alias: {pitcher_name} → {clean_name}")
                if alias_log is not None:
                    alias_log.append(f"{pitcher_name} → {clean_name} [targeted_local_recovery]")
            else:
                alias_file = os.path.join(DATA_DIR, "pitcher_aliases.json")
                if os.path.exists(alias_file):
                    try:
                        with open(alias_file, "r") as f:
                            alias_map = json.load(f)

                        reversed_aliases = {}
                        for official, variations in alias_map.items():
                            if isinstance(variations, list):
                                for a in variations:
                                    reversed_aliases[a.lower()] = official.lower()
                            elif isinstance(variations, str) and variations.strip():
                                reversed_aliases[variations.lower()] = official.lower()

                        if alias_key in reversed_aliases:
                            clean_name = DataManager.normalize_name(reversed_aliases[alias_key])
                            print(f"🔁 Alias matched: {pitcher_name} → {clean_name}")
                            if alias_log is not None:
                                alias_log.append(f"{pitcher_name} → {clean_name}")
                    except Exception as e:
                        print(f"⚠️ Alias file error: {e}")

            if clean_name in df.index:
                row = df.loc[clean_name]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                if row.isnull().any():
                    print(f"🚫 Skipping '{clean_name}' due to NaN values:\n{row}")
                    return None
                return row

            choices = df.index.tolist()
            result = process.extractOne(clean_name, choices, scorer=fuzz.WRatio, score_cutoff=NAME_MATCH_THRESHOLD)
            if result:
                best_match, score = result[0], result[1]
                if clean_name.split()[-1] == best_match.split()[-1]:
                    print(f"🟨 Fuzzy matched: {pitcher_name} → {best_match} ({score}%)")
                    return df.loc[best_match]
                print(f"⛔ Reject fuzzy match due to last-name mismatch: {pitcher_name} → {best_match}")

            manual_fallback_path = os.path.join(DATA_DIR, "manual_fallback_pitchers.csv")
            if os.path.exists(manual_fallback_path):
                try:
                    fdf = pd.read_csv(manual_fallback_path)
                    fdf.set_index(fdf["Name"].str.lower(), inplace=True)
                    key = pitcher_name.lower()
                    if key in fdf.index:
                        print(f"📦 Using manual fallback for: {pitcher_name}")
                        return fdf.loc[key].to_dict()
                except Exception as e:
                    print(f"⚠️ Failed to load manual fallback for {pitcher_name}: {e}")

            print(f"⚠️ Falling back to league average for: {pitcher_name}")
            try:
                from model.overgang_model import send_telegram_alert  # type: ignore

                send_telegram_alert(
                    f"⚠️ *Over Gang Alert*: No pitcher match for `{pitcher_name}`, using *league average* ⚾"
                )
            except Exception:
                pass

            return {"xERA": 4.50, "WHIP": 1.30, "LowIP": True}

        DataManager.match_pitcher_row = staticmethod(_overgang_match_pitcher_row_no_scrape)
        DataManager._overgang_match_pitcher_no_scrape = True
except Exception:
    pass

# ================================
# 📐 PROJECTION MODEL CONSTANTS (true expected runs)
# ================================
LEAGUE_RUNS_PER_TEAM = 4.25   # league avg runs per team per 9
LEAGUE_ERA = 4.25
STARTER_IP_SHARE = 0.60      # ~60% of IP from starter, 40% bullpen
BULLPEN_IP_SHARE = 0.40
WHIP_LEAGUE = 1.30            # baseline WHIP for modifier
EDGE_THRESHOLD = 0.25        # min |edge| to recommend OVER/UNDER (runs); tune up (e.g. 0.35) if too aggressive
EDGE_FOR_FULL_UNIT = 0.5     # edge >= this gets 1.0 unit; scale below; tune up (e.g. 0.6) for conservative sizing
# Unified bullpen workload fatigue (single run multiplier — do not stack with a second IP rule):
# expected_weekly_ip ≈ reliever_count * BULLPEN_EXPECTED_IP_PER_RELIEVER_WEEK
# fatigue_ratio = IP_Week / expected_weekly_ip — no penalty at or below neutral; modest +runs when elevated.
BULLPEN_EXPECTED_IP_PER_RELIEVER_WEEK = 3.5
BULLPEN_FATIGUE_RATIO_NEUTRAL = 1.0
BULLPEN_FATIGUE_RUNS_PER_EXCESS_RATIO = 0.22
BULLPEN_FATIGUE_RUNS_MULT_MAX = 1.08
VELO_DROP_RUNS_PER_MPH = 0.02      # +2% runs per mph velocity drop below 0
LINEUP_IMPACT_CAP = 0.10     # cap (1 + lineup_impact) to ±10%; kept smaller than offense_mult
LOW_IP_XERA_PENALTY = 0.75   # add to xERA when pitcher has low IP (unreliable)
OFFENSE_MULT_MIN = 0.90      # clamp offense_mult (team offense strength vs pitcher hand)
OFFENSE_MULT_MAX = 1.10

# Park Factors
PARK_FACTORS = {
    "Coors Field": (1.25, 0.80),
    "Fenway Park": (1.15, 0.90),
    "Globe Life Field": (1.15, 0.90),
    "Camden Yards": (1.10, 0.92),
    "Great American Ball Park": (1.12, 0.91),
    "Wrigley Field": (1.08, 0.95),
    "Petco Park": (0.92, 1.10),
    "Oracle Park": (0.85, 1.15),
    "Citi Field": (0.90, 1.12),
    "Dodger Stadium": (0.95, 1.08),
    "T-Mobile Park": (0.88, 1.14),
    "Unknown": (1.0, 1.0)
}


def project_team_runs(
    opponent_starter_xera: float,
    opponent_starter_whip: float,
    opponent_bullpen_era: float,
    opponent_bullpen_ip_week: float,
    opponent_bullpen_relievers: int,
    park_runs_factor: float,
    lineup_impact: float,
    opponent_velo_drop: float,
    opponent_low_ip: bool = False,
    offense_mult: float = 1.0,
) -> float:
    """
    Project expected runs scored by one team in the game (they face opponent starter + bullpen).

    offense_mult: team offense strength vs opposing pitcher hand (Batters.offense_vs_hand_dict "mult");
      clamped to OFFENSE_MULT_MIN..OFFENSE_MULT_MAX. 1.0 when batter data unavailable.
    lineup_impact: smaller adjustment from LineupImpact.score_lineup (capped by LINEUP_IMPACT_CAP).
    opponent_bullpen_relievers: active reliever count for expected weekly IP baseline (workload fatigue ratio).
    """
    if opponent_low_ip:
        opponent_starter_xera = min(opponent_starter_xera + LOW_IP_XERA_PENALTY, 6.0)
    effective_era = (
        STARTER_IP_SHARE * opponent_starter_xera + BULLPEN_IP_SHARE * opponent_bullpen_era
    )
    effective_era = max(2.5, min(7.0, effective_era))
    base_era = 0.88 * effective_era + 0.12 * LEAGUE_ERA
    runs = LEAGUE_RUNS_PER_TEAM * (base_era / LEAGUE_ERA)

    runs *= park_runs_factor

    offense_mult = max(OFFENSE_MULT_MIN, min(OFFENSE_MULT_MAX, float(offense_mult)))
    runs *= offense_mult

    lineup_mult = 1.0 + max(-LINEUP_IMPACT_CAP, min(LINEUP_IMPACT_CAP, lineup_impact))
    runs *= lineup_mult

    whip_mult = opponent_starter_whip / WHIP_LEAGUE
    whip_mult = max(0.90, min(1.15, whip_mult))
    runs *= whip_mult

    if opponent_velo_drop < 0:
        velo_mult = 1.0 - (VELO_DROP_RUNS_PER_MPH * opponent_velo_drop)
        runs *= min(1.10, velo_mult)

    # One bullpen workload term: IP_Week vs expected IP implied by reliever headcount (no separate IP cutoff rule).
    try:
        n_rel = max(1, int(round(float(opponent_bullpen_relievers))))
    except (TypeError, ValueError):
        n_rel = 7
    expected_weekly_ip = n_rel * BULLPEN_EXPECTED_IP_PER_RELIEVER_WEEK
    if expected_weekly_ip > 0:
        try:
            ip_w = float(opponent_bullpen_ip_week)
        except (TypeError, ValueError):
            ip_w = 0.0
        if ip_w > 0:
            fatigue_ratio = ip_w / expected_weekly_ip
            excess = max(0.0, fatigue_ratio - BULLPEN_FATIGUE_RATIO_NEUTRAL)
            fatigue_mult = 1.0 + min(
                BULLPEN_FATIGUE_RUNS_MULT_MAX - 1.0,
                BULLPEN_FATIGUE_RUNS_PER_EXCESS_RATIO * excess,
            )
            runs *= fatigue_mult

    runs = min(runs, 6.5)
    return round(runs, 2)


# ================================
# ⚾ BULLPEN DATA (MLB Stats API)
# ================================
class BullpenManager:
    BULLPEN_CSV = os.path.join("data", "bullpen_stats.csv")
    UPDATER = os.path.join("scripts", "update_bullpen.py")

    TEAM_NAME_FIXES = {
        "Athletics": "Oakland Athletics",
        "Angels": "Los Angeles Angels",
        "Dodgers": "Los Angeles Dodgers",
        "Giants": "San Francisco Giants",
        "Cubs": "Chicago Cubs",
        "White Sox": "Chicago White Sox",
        "Cardinals": "St. Louis Cardinals",
        "Guardians": "Cleveland Guardians",
        "D-backs": "Arizona Diamondbacks",
        "Red Sox": "Boston Red Sox",
        "Yankees": "New York Yankees",
        "Mets": "New York Mets",
        "Rays": "Tampa Bay Rays",
        "Nationals": "Washington Nationals",
        "Marlins": "Miami Marlins",
    }

    @staticmethod
    def _safe_read_csv(path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def update_bullpen_data(max_age_hours: int = 6) -> pd.DataFrame:
        """Ensure MLB-API bullpen CSV exists and is fresh; return as DataFrame."""
        need_refresh = True
        if os.path.exists(BullpenManager.BULLPEN_CSV):
            age_hours = (time.time() - os.path.getmtime(BullpenManager.BULLPEN_CSV)) / 3600.0
            need_refresh = age_hours > max_age_hours

        if need_refresh:
            print("🔄 Updating bullpen stats (MLB Stats API)…")
            try:
                # run the updater in project root so relative paths resolve
                subprocess.check_call(["python", "-u", BullpenManager.UPDATER])
            except Exception as e:
                print(f"⚠️ Bullpen update failed to run updater: {e}")

        df = BullpenManager._safe_read_csv(BullpenManager.BULLPEN_CSV)
        if df.empty:
            print(f"⚠️ {BullpenManager.BULLPEN_CSV} missing or empty; using empty DataFrame")
        return df

    @staticmethod
    def get_bullpen_stats(team: str) -> dict:
        """Return dict with ERA, IP_Week, Relievers for a team; league-average fallback."""
        # Fix short/common names to MLB API full names
        team_fixed = BullpenManager.TEAM_NAME_FIXES.get(team, team)

        df = BullpenManager._safe_read_csv(BullpenManager.BULLPEN_CSV)
        if df.empty:
            return {'ERA': 4.25, 'IP_Week': 12.0, 'Relievers': 7, 'source': 'League Avg'}

        # Normalize
        if "Team" not in df.columns:
            # unexpected schema — fail safe
            return {'ERA': 4.25, 'IP_Week': 12.0, 'Relievers': 7, 'source': 'League Avg'}

        df["Team_norm"] = df["Team"].astype(str).str.strip().str.lower()
        lookup = team_fixed.strip().lower()

        # Try exact
        hit = df[df["Team_norm"] == lookup]
        if hit.empty:
            # try original (in case caller already had the full name)
            hit = df[df["Team_norm"] == team.strip().lower()]

        if hit.empty:
            print(f"⚠️ Team not found in bullpen file: {team} (resolved: {team_fixed})")
            return {'ERA': 4.25, 'IP_Week': 12.0, 'Relievers': 7, 'source': 'League Avg'}

        row = hit.iloc[0].to_dict()
        # ensure numeric; NaN/inf from CSV must not propagate (would break ERA blend + clamp in project_team_runs)
        def _finite_float(v, default):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return default
            if not np.isfinite(x):
                return default
            return x

        era = _finite_float(row.get("ERA"), 4.25)
        ipw = _finite_float(row.get("IP_Week"), 12.0)
        rel = int(_finite_float(row.get("Relievers"), 7.0))

        return {
            "ERA": era,
            "IP_Week": ipw,
            "Relievers": rel,
            "source": "MLB Stats API",
        }

# ================================
# 🧠 VEGAS LINE + VELO TRACKING (The Odds API + CSV fallback)
# ================================
def _default_odds_info():
    return {"total_line": 8.5, "over_juice": -110, "under_juice": -110, "ml_home": None, "ml_away": None, "book": "", "_has_real_total": False}


@lru_cache(maxsize=1)
def _load_manual_totals():
    """
    Optional trusted manual totals override from data/manual_totals.csv.
    Expected columns: Game, Total_Line, Over_Juice, Under_Juice, Book.
    Returns dict: normalized 'away @ home' -> {total_line, over_juice, under_juice, book}.
    """
    csv_path = "data/manual_totals.csv"
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ manual_totals.csv load failed: {e}")
        return {}
    required = {"Game", "Total_Line", "Over_Juice", "Under_Juice", "Book"}
    if not required.issubset(set(df.columns)):
        print(f"⚠️ manual_totals.csv missing required columns; found {list(df.columns)}")
        return {}
    manual = {}
    for _, row in df.iterrows():
        game = str(row.get("Game") or "").strip()
        if " @ " not in game:
            continue
        away_raw, home_raw = game.split(" @ ", 1)
        away = normalize_team_name(away_raw.strip())
        home = normalize_team_name(home_raw.strip())
        key = f"{away} @ {home}"
        try:
            total_line = float(row.get("Total_Line"))
        except (TypeError, ValueError):
            continue
        try:
            over_juice = int(row.get("Over_Juice"))
        except (TypeError, ValueError):
            over_juice = -110
        try:
            under_juice = int(row.get("Under_Juice"))
        except (TypeError, ValueError):
            under_juice = -110
        book = str(row.get("Book") or "").strip()
        manual[key] = {
            "total_line": total_line,
            "over_juice": over_juice,
            "under_juice": under_juice,
            "ml_home": None,
            "ml_away": None,
            "book": book,
        }
    if manual:
        print(f"[ODDS] Loaded manual_totals.csv overrides for {len(manual)} games.")
    return manual


class VegasLines:
    @staticmethod
    def get_vegas_line(home_team, away_team, odds_map=None, *, emit_live_total_diagnostics=True):
        """
        Return (vegas_line_float, odds_info_dict).
        odds_info_dict has total_line, over_juice, under_juice, ml_home, ml_away, book.
        Uses The Odds API when odds_map provided and match found; else CSV; else 8.5 + default juice/book.
        Set emit_live_total_diagnostics=False for silent classification (e.g. preflight counts) — same logic, no prints.
        """
        lookup_key = f"{normalize_team_name(away_team)} @ {normalize_team_name(home_team)}"

        # 1) Trusted manual overrides
        manual = _load_manual_totals()
        if manual and lookup_key in manual:
            row = manual[lookup_key]
            raw_line = row.get("total_line")
            try:
                line = float(raw_line)
            except (TypeError, ValueError):
                line = 8.5
            info = dict(row)
            info["_source"] = "manual_totals_csv"
            info["_lookup_key"] = lookup_key
            info["_match_found"] = True
            info["_has_real_total"] = True
            return (line, info)

        # 2) API / merged odds map
        if odds_map is not None:
            match_found = lookup_key in odds_map
            row = get_game_odds(away_team, home_team, odds_map)
            # get_game_odds returns DEFAULT_TOTAL when there is no map row OR when the row has no total_line.
            # Real-total classification must use the matched map row only — never treat default 8.5 as a book line.
            if match_found:
                mrow = odds_map.get(lookup_key) or {}
                raw_line = mrow.get("total_line")
                src_for_flags = mrow
                info = dict(row)
                for _k in ("sportsbook_id", "sportsbook_url", "odd_type"):
                    if _k in mrow:
                        info[_k] = mrow.get(_k)
            else:
                raw_line = None
                src_for_flags = row
                info = dict(row)
            raw_line_missing = raw_line is None or raw_line == ""
            try:
                line = float(row.get("total_line", 8.5))
            except (TypeError, ValueError):
                line = 8.5
            book_empty = not (src_for_flags.get("book") or "").strip()
            book_scrambled = (src_for_flags.get("book") or "").strip().lower() == "scrambled"
            line_realistic = (5 <= line <= 15) if isinstance(line, (int, float)) else False
            has_real_total = (
                bool(match_found)
                and (not raw_line_missing)
                and (not book_scrambled)
                and line_realistic
            )
            is_fallback_line = not has_real_total
            if match_found:
                source = "fallback (scrambled book)" if book_scrambled else ("8.5 fallback" if is_fallback_line else "Odds API")
            else:
                source = "8.5 fallback (no match in odds_map)"
            info["_source"] = source
            info["_lookup_key"] = lookup_key
            info["_match_found"] = match_found
            info["_has_real_total"] = bool(has_real_total)
            if match_found:
                # Lightweight reason flags for LIVE TOTAL BLOCKERS logging (no classification behavior change)
                info["_blocker_scrambled_book"] = bool(book_scrambled)
                info["_blocker_empty_book"] = bool(book_empty and not book_scrambled)
                info["_blocker_missing_total_line"] = bool(raw_line_missing and (not book_empty) and (not book_scrambled))
                info["_blocker_unrealistic_total"] = bool((not raw_line_missing) and (not line_realistic) and (not book_empty) and (not book_scrambled))
                info["_blocker_real_total_pass"] = bool(has_real_total)
                if emit_live_total_diagnostics:
                    _emit_lt_diag = lookup_key not in _LIVE_TOTAL_BLOCKER_DIAG_KEYS_EMITTED
                    if _emit_lt_diag:
                        _LIVE_TOTAL_BLOCKER_DIAG_KEYS_EMITTED.add(lookup_key)
                    if _emit_lt_diag:
                        if book_empty:
                            _raw_detail = {}
                            if isinstance(odds_map, dict):
                                _raw_detail = odds_map.get(lookup_key) or {}
                            _sportsbook_id = _raw_detail.get("sportsbook_id") or _raw_detail.get("SportsbookId") or _raw_detail.get("sportsbookId") or ""
                            _sportsbook_url = _raw_detail.get("sportsbook_url") or _raw_detail.get("SportsbookUrl") or _raw_detail.get("sportsbookUrl") or ""
                            _odd_type = _raw_detail.get("odd_type") or _raw_detail.get("OddType") or _raw_detail.get("oddType") or ""
                            print(
                                "[EMPTY BOOK DETAIL] "
                                f"key={lookup_key} | sportsbook_id={repr(_sportsbook_id)} | sportsbook_url={repr(_sportsbook_url)} | "
                                f"odd_type={repr(_odd_type)} | source={source}"
                            )
                        print(
                            "[LIVE TOTAL CHECK] "
                            f"key={lookup_key} | raw_total_line={repr(raw_line)} | parsed_line={line} | "
                            f"book={repr(row.get('book'))} | source={source} | realistic={line_realistic} | "
                            f"scrambled={book_scrambled} | _has_real_total={bool(has_real_total)}"
                        )
            return (line, info)
        csv_path = "data/public_betting.csv"
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            info = _default_odds_info()
            info["_source"] = "8.5 fallback (no odds_map, CSV missing/empty)"
            info["_lookup_key"] = ""
            info["_match_found"] = False
            info["_has_real_total"] = False
            return (8.5, info)
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                if f.read().strip() == "":
                    return (8.5, _default_odds_info())
        except Exception:
            pass
        try:
            df = pd.read_csv(csv_path)
            game_key = f"{away_team.lower()} @ {home_team.lower()}"
            r = df[df["Game"].str.lower() == game_key]
            if not r.empty:
                total_current = float(r.iloc[0].get("total_current", 8.5))
                info = _default_odds_info()
                info["total_line"] = total_current
                info["book"] = "CSV"
                info["_source"] = "CSV"
                info["_lookup_key"] = game_key
                info["_match_found"] = True
                info["_has_real_total"] = True
                return (total_current, info)
        except Exception as e:
            print(f"⚠️ Vegas line fetch failed: {e}")
        info = _default_odds_info()
        info["_source"] = "8.5 fallback (CSV read failed or no row)"
        info["_lookup_key"] = ""
        info["_match_found"] = False
        info["_has_real_total"] = False
        return (8.5, info)


def fetch_mlb_odds_by_date_allow_empty_book(target_date_yyyy_mm_dd):
    """
    SportsDataIO GameOddsByDate → same dict shape as core.sportsdataio.fetch_mlb_odds_by_date.

    Delegates to the core implementation (single source of truth). PregameOdds rows with a valid
    OverUnder are not dropped solely because Sportsbook/SportsbookName is blank; SportsbookId
    still ranks/filters via book_key.
    """
    from core import sportsdataio as sio

    return sio.fetch_mlb_odds_by_date(target_date_yyyy_mm_dd)


# ================================
# 📈 PREDICTION ENGINE (true projection model)
# ================================
def generate_prediction(
    away_stats,
    home_stats,
    bullpen_home,
    bullpen_away,
    velo_drop_away,
    velo_drop_home,
    park_factors,
    vegas_data=None,
    public_data=None,
    away_lineup_impact=0.0,
    home_lineup_impact=0.0,
    away_offense_mult=1.0,
    home_offense_mult=1.0,
    has_real_total=True,
    data_quality_degraded=False,
):
    """
    Project expected runs for each team, sum to projected total, then compare to Vegas.

    Returns dict with: projected_total, away_runs, home_runs, edge, pick, prediction (str),
    confidence, total_open, total_current, recommended_units, skip (bool).
    """
    if public_data is None:
        public_data = {}
    try:
        if isinstance(vegas_data, dict):
            vegas_line = float(vegas_data.get("total_current", 8.5))
        else:
            vegas_line = float(vegas_data)
    except Exception as e:
        print(f"⚠️ Failed to extract vegas_line from vegas_data: {e} → using default 8.5")
        vegas_line = 8.5

    if not isinstance(public_data, dict):
        print(f"❌ Invalid public_data type: {type(public_data)} → {public_data}")
        return {
            "skip": True,
            "prediction": "SKIPPED",
            "confidence": 0.0,
            "total_open": vegas_line,
            "total_current": vegas_line,
        }

    away_xera = safe_get(away_stats, "xERA", 4.50)
    home_xera = safe_get(home_stats, "xERA", 4.50)
    away_whip = safe_get(away_stats, "WHIP", 1.30)
    home_whip = safe_get(home_stats, "WHIP", 1.30)
    bullpen_home_era = safe_get(bullpen_home, "ERA", 4.25)
    bullpen_away_era = safe_get(bullpen_away, "ERA", 4.25)
    bullpen_home_ip_week = safe_get(bullpen_home, "IP_Week", 12.0)
    bullpen_away_ip_week = safe_get(bullpen_away, "IP_Week", 12.0)
    try:
        bullpen_home_rel = max(1, int(round(float(safe_get(bullpen_home, "Relievers", 7)))))
    except (TypeError, ValueError):
        bullpen_home_rel = 7
    try:
        bullpen_away_rel = max(1, int(round(float(safe_get(bullpen_away, "Relievers", 7)))))
    except (TypeError, ValueError):
        bullpen_away_rel = 7
    over_boost, _ = park_factors

    # ---------- Project runs for each team ----------
    # Away offense faces home pitcher + home bullpen; away_offense_mult from Batters.offense_vs_hand_dict(away_team vs home_hand)
    away_runs = project_team_runs(
        opponent_starter_xera=home_xera,
        opponent_starter_whip=home_whip,
        opponent_bullpen_era=bullpen_home_era,
        opponent_bullpen_ip_week=bullpen_home_ip_week,
        opponent_bullpen_relievers=bullpen_home_rel,
        park_runs_factor=over_boost,
        lineup_impact=away_lineup_impact,
        opponent_velo_drop=velo_drop_home,
        opponent_low_ip=safe_get(home_stats, "LowIP", False),
        offense_mult=away_offense_mult,
    )
    # Home offense faces away pitcher + away bullpen; home_offense_mult from Batters.offense_vs_hand_dict(home_team vs away_hand)
    home_runs = project_team_runs(
        opponent_starter_xera=away_xera,
        opponent_starter_whip=away_whip,
        opponent_bullpen_era=bullpen_away_era,
        opponent_bullpen_ip_week=bullpen_away_ip_week,
        opponent_bullpen_relievers=bullpen_away_rel,
        park_runs_factor=over_boost,
        lineup_impact=home_lineup_impact,
        opponent_velo_drop=velo_drop_away,
        opponent_low_ip=safe_get(away_stats, "LowIP", False),
        offense_mult=home_offense_mult,
    )

    projected_total = round(away_runs + home_runs, 2)
    edge = round(projected_total - vegas_line, 2)

    # ---------- Pick: OVER / UNDER based on edge vs threshold ----------
    if edge >= EDGE_THRESHOLD:
        pick = "OVER"
        strength = "🚨 " if edge >= 0.6 else ""
        prediction_str = f"{strength}OVER {vegas_line:.1f}"
    elif edge <= -EDGE_THRESHOLD:
        pick = "UNDER"
        strength = "🔒 " if edge <= -0.6 else ""
        prediction_str = f"{strength}UNDER {vegas_line:.1f}"
    else:
        pick = "LEAN_OVER" if edge > 0 else "LEAN_UNDER"
        prediction_str = f"LEAN {'OVER' if edge > 0 else 'UNDER'} {vegas_line:.1f} (proj {projected_total:.1f})"

    # ---------- Base confidence from |edge| ----------
    abs_edge = abs(edge)
    if abs_edge >= 1.0:
        confidence = 0.85
    elif abs_edge >= 0.5:
        confidence = 0.75
    elif abs_edge >= EDGE_THRESHOLD:
        confidence = 0.65
    else:
        confidence = 0.45

    # (Bullpen/park already in projection; avoid re-using for confidence to prevent bias)

    # Velocity drop (tired starter): slight confidence reduction (data noisier)
    if velo_drop_away < VELOCITY_DROP_THRESHOLD:
        confidence *= 0.98
    if velo_drop_home < VELOCITY_DROP_THRESHOLD:
        confidence *= 0.99

    relievers = (safe_get(bullpen_away, "Relievers", 7) + safe_get(bullpen_home, "Relievers", 7)) / 14.0
    reliever_factor = min(1.0, relievers)
    if pick == "OVER":
        confidence *= 1.0 + (1.0 - reliever_factor) * 0.1
    else:
        confidence *= 1.0 + reliever_factor * 0.05

    # Loader uses total_open / total_current; accept legacy Total_Open / Total_Current.
    _raw_to = public_data.get("total_open")
    if _raw_to is None or _raw_to == "":
        _raw_to = public_data.get("Total_Open")
    _raw_tc = public_data.get("total_current")
    if _raw_tc is None or _raw_tc == "":
        _raw_tc = public_data.get("Total_Current")
    total_open = safe_float(_raw_to, default=vegas_line)
    total_current = safe_float(_raw_tc, default=total_open)

    # Public betting: fade heavy public (contrarian)
    try:
        ou_pct = float(public_data.get("ou_bets_pct_over", 50))
        if pick == "OVER":
            if ou_pct > 65:
                confidence += 0.05
            elif ou_pct < 35:
                confidence -= 0.05
        elif pick == "UNDER":
            if ou_pct > 65:
                confidence -= 0.05
            elif ou_pct < 35:
                confidence += 0.05
        if (pick == "OVER" and ou_pct >= 80) or (pick == "UNDER" and (100 - ou_pct) >= 80):
            confidence = min(0.99, confidence + 0.10)
    except Exception as e:
        print(f"⚠️ Public data ou_pct error: {e}")

    if total_current > total_open:
        confidence += 0.02
    elif total_current < total_open:
        confidence -= 0.02

    confidence = max(0.01, min(confidence, 0.99))
    if not has_real_total:
        confidence = min(0.59, confidence * 0.65)
        prediction_str = "NO BET (fallback total only)"
    if data_quality_degraded:
        confidence = min(confidence, 0.79)

    # Recommended units from edge size (only bet when |edge| >= threshold)
    if abs_edge < EDGE_THRESHOLD:
        recommended_units = 0.0
    elif abs_edge >= EDGE_FOR_FULL_UNIT:
        recommended_units = 1.0
    else:
        recommended_units = round(0.5 + 0.5 * (abs_edge - EDGE_THRESHOLD) / (EDGE_FOR_FULL_UNIT - EDGE_THRESHOLD), 2)

    print(
        f"📦 Projection: {away_runs:.2f} + {home_runs:.2f} = {projected_total:.2f} | "
        f"Edge: {edge:+.2f} | {prediction_str} | Conf: {confidence:.2f}"
    )
    return {
        "skip": False,
        "projected_total": projected_total,
        "away_runs": away_runs,
        "home_runs": home_runs,
        "vegas_line": vegas_line,
        "edge": edge,
        "pick": pick,
        "prediction": prediction_str,
        "confidence": round(confidence, 2),
        "total_open": total_open,
        "total_current": total_current,
        "recommended_units": recommended_units,
    }

# ================================
# 💬 TELEGRAM UTILS
# ================================
def send_telegram_alert(message):
    for attempt in range(3):
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                    "disable_notification": False
                },
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            if attempt == 2:
                print(f"💥 Telegram failed: {e}")
            time.sleep(2)
    return False

def _alert_formatted_time(game_data: dict) -> str:
    try:
        game_utc_time = datetime.strptime(game_data["Datetime"], "%Y-%m-%dT%H:%M:%SZ")
        mt_time = game_utc_time.replace(tzinfo=utc).astimezone(timezone("US/Mountain"))
        return mt_time.strftime("%I:%M %p MT")
    except Exception:
        return "TBD"


def _confidence_emoji_for_percent(raw_conf: float) -> str:
    if raw_conf >= 95:
        return "🔥"
    if raw_conf >= 90:
        return "💪"
    if raw_conf >= 80:
        return "👍"
    if raw_conf >= 70:
        return "🤞"
    return "😬"


def _ou_confidence_display(game_data: dict) -> tuple:
    """O/U row: Confidence_Value (0–1) or Confidence string (percent)."""
    try:
        raw_conf = game_data.get("Confidence_Value")
        if raw_conf is None:
            raw_conf = float(str(game_data.get("Confidence", "0")).replace("%", ""))
        else:
            raw_conf = float(raw_conf)
            if 0 <= raw_conf <= 1.0:
                raw_conf = raw_conf * 100.0
        clean = f"{raw_conf:.0f}%"
        emoji = _confidence_emoji_for_percent(raw_conf)
        return clean, emoji
    except Exception:
        return str(game_data.get("Confidence", "?")), ""


def _ml_confidence_display(game_data: dict) -> tuple:
    """ML row: ML_Confidence string like '64%'."""
    try:
        raw = float(str(game_data.get("ML_Confidence", "0")).replace("%", "").strip())
        clean = f"{raw:.0f}%"
        emoji = _confidence_emoji_for_percent(raw)
        return clean, emoji
    except Exception:
        return str(game_data.get("ML_Confidence", "?")), ""


def _ou_confidence_percent_for_telegram_gate(game_data: dict) -> float:
    """O/U confidence on 0–100 scale for Telegram send gating (Confidence_Value 0–1 or Confidence %)."""
    try:
        raw_conf = game_data.get("Confidence_Value")
        if raw_conf is None:
            return float(str(game_data.get("Confidence", "0")).replace("%", ""))
        raw_conf = float(raw_conf)
        if 0 <= raw_conf <= 1.0:
            return raw_conf * 100.0
        return raw_conf
    except Exception:
        return float("-inf")


def _ml_confidence_percent_for_telegram_gate(game_data: dict) -> float:
    """ML_Confidence on 0–100 scale for Telegram send gating (parses percent strings)."""
    try:
        return float(str(game_data.get("ML_Confidence", "0")).replace("%", "").strip())
    except Exception:
        return float("-inf")


def _fmt_num_one(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.1f}"
    return str(v)


def format_ou_alert(game_data: dict) -> str:
    """Customer-facing O/U Telegram message (CSV remains full detail)."""
    t = _alert_formatted_time(game_data)
    conf, emoji = _ou_confidence_display(game_data)
    proj = game_data.get("Projected_Total", "?")
    edge_val = game_data.get("Edge", "?")
    if isinstance(edge_val, (int, float)):
        edge_str = f"{edge_val:+.1f}"
    else:
        edge_str = str(edge_val)
    return (
        f"📊 *O/U · Over Gang*\n"
        f"🏟️ *{game_data['Game']}*\n"
        f"📍 {game_data.get('Venue', 'Unknown')} | 🕒 {t}\n\n"
        f"🎯 {game_data['Pitchers']}\n"
        f"📊 xERA {game_data['xERA']} · WHIP {game_data['WHIP']}\n\n"
        f"Projection: {_fmt_num_one(proj)} | Edge: {edge_str}\n"
        f"🧠 *Pick*: {game_data['Prediction']}\n"
        f"💪 *Confidence*: {conf}{f' {emoji}' if emoji else ''} · *Units*: {game_data.get('Units', '-')}\n\n"
        "18+ only. For informational purposes only. Past performance does not guarantee future results. Bet responsibly."
    )


def format_ml_alert(game_data: dict) -> str:
    """Customer-facing ML Telegram message (distinct from O/U; CSV remains full detail)."""
    t = _alert_formatted_time(game_data)
    conf, emoji = _ml_confidence_display(game_data)
    return (
        f"💵 *ML · Over Gang*\n"
        f"🏟️ *{game_data['Game']}*\n"
        f"📍 {game_data.get('Venue', 'Unknown')} | 🕒 {t}\n\n"
        f"🎯 {game_data['Pitchers']}\n"
        f"📊 xERA {game_data['xERA']} · WHIP {game_data['WHIP']}\n\n"
        f"🏆 *Pick*: {game_data.get('ML_Pick', '-')}\n"
        f"💪 *Confidence*: {conf}{f' {emoji}' if emoji else ''} · *Kelly*: {game_data.get('ML_Kelly_Units', '-')}\n\n"
        "18+ only. For informational purposes only. Past performance does not guarantee future results. Bet responsibly."
    )

def send_telegram_file(file_path, caption="📊 Over Gang Predictions"):
    try:
        with open(file_path, 'rb') as doc:
            response = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption
                },
                files={"document": doc},
                timeout=15
            )
        response.raise_for_status()
        print(f"📤 CSV file sent to Telegram: {file_path}")
    except Exception as e:
        print(f"❌ Failed to send file to Telegram: {e}")

def safe_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

# --- Safe lineup impact helper (ALWAYS returns values) ---
def safe_lineup_impacts(lineup_obj, away_lineup, home_lineup, away_pitcher_hand, home_pitcher_hand, logger=None):
    """
    Returns: (away_impact, home_impact, away_scope, home_scope)
    Impacts are floats (default 0.0). Scopes: 'lineup'|'team'|'none'
    """
    away_impact = 0.0
    home_impact = 0.0
    away_scope = "none"
    home_scope = "none"
    try:
        # AWAY hitters vs HOME starter's hand
        away_impact, away_scope = lineup_obj.score_lineup(
            away_lineup,
            pitcher_hand=(home_pitcher_hand or 'R')
        )
        # HOME hitters vs AWAY starter's hand
        home_impact, home_scope = lineup_obj.score_lineup(
            home_lineup,
            pitcher_hand=(away_pitcher_hand or 'R')
        )
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Lineup impact error (safe): {e}")
    # clamp to a sane range
    away_impact = max(-0.30, min(0.30, float(away_impact)))
    home_impact = max(-0.30, min(0.30, float(home_impact)))
    return away_impact, home_impact, away_scope, home_scope

# ================================
# 🔍 PREFLIGHT (validation scaffold)
# ================================
def _preflight_count_games_with_real_totals(games, odds_map):
    """
    Count slate games where VegasLines.get_vegas_line reports _has_real_total (same rule as per-game logs).
    Does not treat mere odds_map keys or default 8.5 as real market coverage.
    Uses emit_live_total_diagnostics=False so preflight does not print per-game live-total diagnostics.
    """
    if not games:
        return 0
    n = 0
    for g in games:
        try:
            home_team = safe_get(g, "home_name", "")
            away_team = safe_get(g, "away_name", "")
            _, info = VegasLines.get_vegas_line(
                home_team, away_team, odds_map, emit_live_total_diagnostics=False
            )
            if bool(info.get("_has_real_total")):
                n += 1
        except Exception:
            continue
    return n


def _vegas_line_info_is_manual_trusted_total(info):
    """
    True when this VegasLines info dict is a manual-totals trusted line.
    Same rules as _result_row_is_manual_trusted_total: _source manual_totals_csv or book Manual.
    """
    if not isinstance(info, dict):
        return False
    if str(info.get("_source") or "").strip() == "manual_totals_csv":
        return True
    if str(info.get("book") or "").strip() == "Manual":
        return True
    return False


def _result_row_is_manual_trusted_total(r):
    """
    True when this prediction result row used a manual-totals line. Uses Total_Line_Source and
    Odds_Book (propagated from VegasLines odds_info), not Market_Source (global fetch label).
    Delegates to _vegas_line_info_is_manual_trusted_total for one canonical definition.
    """
    if not isinstance(r, dict):
        return False
    return _vegas_line_info_is_manual_trusted_total(
        {"_source": r.get("Total_Line_Source"), "book": r.get("Odds_Book")}
    )


def _preflight_count_games_with_non_manual_real_totals(games, odds_map):
    """
    Count slate games with a real O/U from VegasLines that is NOT manual_totals.csv / Manual book.
    Used for full_auto gating (aligned with api_real_n: market-trusted totals only).
    """
    if not games:
        return 0
    n = 0
    for g in games:
        try:
            home_team = safe_get(g, "home_name", "")
            away_team = safe_get(g, "away_name", "")
            _, info = VegasLines.get_vegas_line(
                home_team, away_team, odds_map, emit_live_total_diagnostics=False
            )
            if bool(info.get("_has_real_total")) and not _vegas_line_info_is_manual_trusted_total(info):
                n += 1
        except Exception:
            continue
    return n


def run_preflight_checks(stats_df, games, public_betting_data, odds_map):
    """
    Inspect already-loaded data and print a preflight summary.
    Returns dict: ok, mode, warnings, issues.
    Does not block execution.
    Modes: full_auto | manual_test | projection_only | stop
    """
    warnings = []
    issues = []
    try:
        pitcher_ok = stats_df is not None and not (isinstance(stats_df, pd.DataFrame) and stats_df.empty)
        pitcher_status = "ok" if pitcher_ok else "missing or empty"
        pitcher_n = len(stats_df) if isinstance(stats_df, pd.DataFrame) else 0

        batter_ok = BATTER_DF is not None and not (isinstance(BATTER_DF, pd.DataFrame) and BATTER_DF.empty)
        batter_status = "ok" if batter_ok else "missing or empty"
        batter_n = len(BATTER_DF) if isinstance(BATTER_DF, pd.DataFrame) else 0

        bullpen_path = getattr(BullpenManager, "BULLPEN_CSV", os.path.join("data", "bullpen_stats.csv"))
        bullpen_exists = os.path.exists(bullpen_path) and os.path.getsize(bullpen_path) > 0
        try:
            bf = pd.read_csv(bullpen_path) if bullpen_exists else pd.DataFrame()
            bullpen_n = len(bf) if not bf.empty else 0
        except Exception:
            bullpen_n = 0
        bullpen_ok = bullpen_exists and bullpen_n > 0
        bullpen_status = "ok" if bullpen_ok else "missing or empty"

        games_n = len(games) if games is not None else 0
        games_status = f"{games_n} games" if games_n else "no games"

        public_ok = public_betting_data is not None and isinstance(public_betting_data, dict) and len(public_betting_data) > 0
        public_status = "loaded" if public_ok else "empty or missing"
        public_n = len(public_betting_data) if isinstance(public_betting_data, dict) else 0

        odds_map_n = len(odds_map) if (odds_map is not None and isinstance(odds_map, dict)) else 0
        odds_real_n = _preflight_count_games_with_real_totals(games, odds_map)
        odds_non_manual_n = _preflight_count_games_with_non_manual_real_totals(games, odds_map)
        # full_auto requires market (non-manual) trusted totals; manual CSV alone uses manual_test
        odds_ok = odds_non_manual_n > 0
        odds_coverage_ok = games_n > 0 and odds_non_manual_n >= max(1, games_n - 1)
        odds_status = (
            f"real O/U {odds_real_n}/{games_n} (all sources); non-manual (market) {odds_non_manual_n}/{games_n} (odds_map rows={odds_map_n})"
            if games_n
            else "no games"
        )

        manual_totals = _load_manual_totals()
        manual_loaded = isinstance(manual_totals, dict) and len(manual_totals) >= 1
        slate_keys_pf = set()
        if games:
            for g in games:
                _ak = normalize_team_name(safe_get(g, "away_name", ""))
                _hk = normalize_team_name(safe_get(g, "home_name", ""))
                slate_keys_pf.add(f"{_ak} @ {_hk}")
        manual_keys_pf = set(manual_totals.keys()) if isinstance(manual_totals, dict) else set()
        matched_manual_keys_pf = slate_keys_pf.intersection(manual_keys_pf)
        manual_matched_today = len(matched_manual_keys_pf) >= 1

        if not pitcher_ok:
            issues.append("pitcher data missing or empty")
        if not batter_ok:
            warnings.append("batter data missing or empty")
        if not bullpen_ok:
            warnings.append("bullpen data missing or empty")
        if games_n == 0:
            issues.append("no games for slate")
        # Public betting is optional overlay; do not warn or gate modes on it.

        if not pitcher_ok or games_n == 0:
            mode = "stop"
            if not pitcher_ok:
                issues.append("run mode stop: missing pitcher data")
            if games_n == 0:
                issues.append("run mode stop: no games for slate")
        elif (
            pitcher_ok
            and bullpen_ok
            and odds_ok
            and odds_coverage_ok
        ):
            mode = "full_auto"
            warnings.append("all systems go; full auto mode")
        elif (
            pitcher_ok
            and bullpen_ok
            and manual_matched_today
            and (not odds_ok or not odds_coverage_ok)
        ):
            mode = "manual_test"
            warnings.append(
                "manual totals matched today's slate; non-manual market totals incomplete — manual_test mode"
            )
        else:
            mode = "projection_only"
            warnings.append("odds weak/fallback or no manual rows for today's slate — projection_only mode")

        ok = mode != "stop"

        print("\n--- PREFLIGHT ---")
        print(f"  Pitcher data:   {pitcher_status} (n={pitcher_n})")
        print(f"  Batter data:   {batter_status} (n={batter_n})")
        print(f"  Bullpen data:  {bullpen_status} (n={bullpen_n})")
        print(f"  Games found:   {games_status}")
        print(f"  Public betting: {public_status} (n={public_n})")
        print(f"  Odds status:   {odds_status} (real_totals_n={odds_real_n}, non_manual_n={odds_non_manual_n})")
        print(
            f"  Manual totals: {'loaded' if manual_loaded else 'none'} "
            f"({len(manual_totals) if isinstance(manual_totals, dict) else 0} rows) | matched today: {len(matched_manual_keys_pf)}"
        )
        print(f"  Proceed mode:  {mode}")
        print("-----------------\n")

        return {"ok": ok, "mode": mode, "warnings": warnings, "issues": issues}
    except Exception as e:
        warnings.append(f"preflight error: {e}")
        return {"ok": True, "mode": "projection_only", "warnings": warnings, "issues": issues}


# ================================
# 🔍 CORE LOGIC
# ================================
def run_predictions():
    print(f"🔮 OVER GANG PREDICTOR v4.0 (projection model) | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*50 + "\n")
    _LIVE_TOTAL_BLOCKER_DIAG_KEYS_EMITTED.clear()

    if AUTO_UPDATE_DATA:
        print("🔄 AUTO UPDATE MODE: Fetching latest data")
        stats_df = DataManager.update_pitcher_stats()
        BullpenManager.update_bullpen_data()
    else:
        print("⏩ MANUAL MODE: Using existing data")
        stats_df = DataManager.load_pitcher_stats()

    velocity_tracker = VelocityTracker()
    lineups = LineupImpact()
    try:
        # --- Active slate date (America/Denver, 04:00 MT rollover); OVERGANG_TARGET_DATE overrides — see active_slate_date_mt()
        today_mt, _ = active_slate_date_mt()

        # Pull MLB schedule for that calendar day (UTC-based API)
        games = schedule(
            start_date=today_mt.strftime('%Y-%m-%d'),
            end_date=today_mt.strftime('%Y-%m-%d')
        )

        try:
            from core.public_betting_scraper import scrape_public_betting

            _pb_ok = scrape_public_betting(target_date=today_mt)
            if not _pb_ok:
                print(
                    "⚠️ Public betting CSV not refreshed (scraper returned False); "
                    "continuing with existing data/public_betting.csv if present."
                )
        except Exception as _e_pb:
            print(
                f"⚠️ Public betting scrape error ({_e_pb!r}); "
                "continuing with existing data/public_betting.csv if present."
            )

        public_betting_data = load_public_betting_data()
        target_date_str = today_mt.strftime("%Y-%m-%d")
        print("[ODDS] Trying SportsDataIO first...")
        sdio_map = fetch_mlb_odds_by_date_allow_empty_book(target_date_str)
        print(f"[ODDS] SportsDataIO odds_map size: {len(sdio_map)}")
        odds_source = "none"
        odds_api_map = {}
        if sdio_map:
            odds_source = "SportsDataIO"
            print("[ODDS] Fetching Odds API for trusted-total lane...")
            odds_api_map = fetch_mlb_odds(target_date=target_date_str) or {}
        else:
            print("[ODDS] Falling back to Odds API...")
            odds_api_map = fetch_mlb_odds(target_date=target_date_str) or {}
            if odds_api_map:
                odds_source = "Odds API"

        def _is_trusted_total_row(row):
            if not isinstance(row, dict):
                return False
            raw = row.get("total_line")
            try:
                total = float(raw)
            except (TypeError, ValueError):
                return False
            if total < 5 or total > 15:
                return False
            book = str(row.get("book") or "").strip()
            if book.lower().strip() == "scrambled":
                return False
            # Empty book is OK (e.g. SportsDataIO PregameOdds with OverUnder + SportsbookId only).
            return True

        # Trusted real-total lane (Odds API preferred; SportsDataIO only if truly trusted)
        trusted_total_source_map = {}
        for k, v in (odds_api_map or {}).items():
            if _is_trusted_total_row(v):
                trusted_total_source_map[k] = dict(v)
                trusted_total_source_map[k]["_trusted_source"] = "Odds API"

        for k, v in (sdio_map or {}).items():
            if k in trusted_total_source_map:
                continue
            if _is_trusted_total_row(v):
                trusted_total_source_map[k] = dict(v)
                trusted_total_source_map[k]["_trusted_source"] = "SportsDataIO"

        # SportsDataIO fallback metadata lane (keep ml/juice metadata; remove weak totals)
        odds_map = dict(sdio_map or {})
        for k in list(odds_map.keys()):
            if k not in trusted_total_source_map:
                row = dict(odds_map.get(k, {}))
                row["total_line"] = None
                row["book"] = ""
                odds_map[k] = row

        # Overlay trusted totals on top of fallback metadata
        for k, v in trusted_total_source_map.items():
            base = dict(odds_map.get(k, {}))
            base.update({
                "total_line": v.get("total_line"),
                "over_juice": v.get("over_juice"),
                "under_juice": v.get("under_juice"),
                "book": v.get("book") or "",
                "sportsbook_id": v.get("sportsbook_id", ""),
                "sportsbook_url": v.get("sportsbook_url", ""),
                "odd_type": v.get("odd_type", ""),
            })
            if base.get("ml_home") is None and v.get("ml_home") is not None:
                base["ml_home"] = v.get("ml_home")
            if base.get("ml_away") is None and v.get("ml_away") is not None:
                base["ml_away"] = v.get("ml_away")
            odds_map[k] = base

        print(f"[ODDS] Final odds source: {odds_source}")

        # Keep only games whose local MT calendar date matches the target slate (today_mt)
        def game_mt_date(g):
            dt_utc = datetime.strptime(
                g["game_datetime"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=ZoneInfo("UTC"))
            return dt_utc.astimezone(ZoneInfo("America/Denver")).date()

        games = [g for g in games if game_mt_date(g) == today_mt]

        print(f"✅ Found {len(games)} games for {today_mt} MT")
        for g in games:
            print("•", f"{g['away_name']} @ {g['home_name']}")

        # Trusted total lane visibility (per slate game)
        for g in games:
            away_norm = normalize_team_name(safe_get(g, "away_name", ""))
            home_norm = normalize_team_name(safe_get(g, "home_name", ""))
            game_key = f"{away_norm} @ {home_norm}"
            trow = trusted_total_source_map.get(game_key)
            if trow is not None and trow != {}:
                t_source = trow.get("_trusted_source", "trusted")
                t_total = trow.get("total_line")
                t_book = trow.get("book") or ""
                t_ok = True
            else:
                t_source = "none"
                t_total = None
                t_book = ""
                t_ok = False
            # One compact line per slate game
            print(
                "[TRUSTED TOTAL SOURCE] "
                f"key={game_key} | source={t_source} | total={t_total} | book={t_book} | trusted={t_ok}"
            )

        # Manual totals freshness check (slate matching only; logging visibility)
        manual_totals = _load_manual_totals()
        manual_n = len(manual_totals) if isinstance(manual_totals, dict) else 0
        slate_keys = {
            f"{normalize_team_name(safe_get(g, 'away_name', ''))} @ {normalize_team_name(safe_get(g, 'home_name', ''))}"
            for g in games
        }
        manual_keys = set(manual_totals.keys()) if isinstance(manual_totals, dict) else set()
        matched_keys = sorted(slate_keys.intersection(manual_keys))
        matched_n = len(matched_keys)
        unmatched_n = manual_n - matched_n
        print(
            f"[MANUAL TOTALS CHECK] manual rows loaded: {manual_n} | matched today: {matched_n} | unmatched manual: {unmatched_n}"
        )
        if matched_n > 0:
            print(f"[MANUAL TOTALS MATCHES] " + "; ".join(matched_keys))

        if not games:
            print("⚠️ No games scheduled today")
            return

        preflight = run_preflight_checks(stats_df, games, public_betting_data, odds_map)
        mode = preflight.get("mode", "projection_only")
        if mode == "full_auto":
            print("[PREFLIGHT] Mode = FULL AUTO | live market conditions look ready")
        elif mode == "manual_test":
            print("[PREFLIGHT] Mode = MANUAL TEST | trusted manual totals active; public/odds incomplete")
        elif mode == "projection_only":
            print("[PREFLIGHT] Mode = PROJECTION ONLY | fallback environment, no trusted market path")
        elif mode == "stop":
            print("[PREFLIGHT] Mode = STOP | critical inputs missing")
        if mode != "stop":
            warnings = preflight.get("warnings", [])
            if warnings:
                print("[PREFLIGHT] Warnings:", "; ".join(warnings))
            print(
                "[READINESS] OU: "
                f"trusted_total_games={len(trusted_total_source_map)} | slate_games={len(games)}"
            )
            print(
                "[READINESS] Public: "
                f"loaded_game_rows={len(public_betting_data)} "
                "(empty file is OK — degrades confidence; not a hard stop)"
            )
            print("[READINESS] ML: per-game ML block runs regardless of trusted O/U (see ML_Fired after slate)")
        if mode == "stop":
            issues = preflight.get("issues", [])
            if issues:
                print("[PREFLIGHT] Stop reason:", "; ".join(issues))
            print("[PREFLIGHT] STOP engaged | exiting before game processing")
            return

    except Exception as e:
        print(f"❌ Schedule API error: {e}")
        return

    results = []
    alerts = []
    ml_alerts = []
    unmatched_pitchers = set()
    alias_log = []

    # --- Local pitcher_stats preflight: who is missing before game processing (no scraper) ---
    RUNTIME_TARGETED_LOCAL_RECOVERY_ALIASES.clear()
    _alias_audit = _build_pitcher_alias_reversed_dict()
    _probable_pitchers = set()
    for _g in games:
        for _pk in ("away_probable_pitcher", "home_probable_pitcher"):
            _raw = safe_get(_g, _pk, "TBD")
            _s = str(_raw) if _raw is not None else ""
            if (not _s.strip()) or _s.strip() == "TBD":
                continue
            _probable_pitchers.add(_s.strip())
    _probable_sorted = sorted(_probable_pitchers, key=lambda x: x.lower())
    _resolved_names = []
    _unresolved_names = []
    for _nm in _probable_sorted:
        if _pitcher_resolvable_locally_without_league_average(
            _nm, stats_df, manual_fallback_df, _alias_audit
        ):
            _resolved_names.append(_nm)
        else:
            _unresolved_names.append(_nm)
    print("\n--- LOCAL PITCHER STATS PREFLIGHT ---")
    print(f"  Probable pitchers checked: {len(_probable_sorted)}")
    print(f"  Resolved locally (CSV / alias / fuzzy / manual): {len(_resolved_names)}")
    print(f"  Unresolved before targeted MLB backfill: {len(_unresolved_names)}")
    if _unresolved_names:
        print("  Unresolved names (pre-backfill):")
        for _u in _unresolved_names:
            print(f"    - {_u}")
        print(
            "  Note: next step tries MLB id + per-player reg-season stats only; "
            "logs label no-id vs no-MLB-season-stats (spring/no-debut) vs upsert success."
        )
    print("-------------------------------------\n")

    # Targeted MLB StatsAPI backfill for today's unresolved probables (no FanGraphs; no full-file replace).
    _unresolved_before = list(_unresolved_names)
    if _unresolved_before:
        _n_attempt = len(_unresolved_before)
        print("--- TARGETED PROBABLE-PITCHER BACKFILL (independent MLB resolver) ---")
        print(f"  Targeted resolver attempted {_n_attempt} name(s): {', '.join(_unresolved_before)}")
        print(
            "  [Per-player lane] No global season leaderboard; each name → MLB id → "
            "/people/{id}/stats (current year, else prior year)."
        )
        try:
            _yr = datetime.now().year
            _mo = datetime.now().month
            if _mo < 6:
                _min_ip_tb = MIN_PITCHER_IP_EARLY
            elif _mo < 8:
                _min_ip_tb = MIN_PITCHER_IP_MID
            else:
                _min_ip_tb = MIN_PITCHER_IP_LATE
            try:
                _savant_tb = DataManager._savant_xera_by_id(_yr)
            except Exception as _se:
                print(f"[Targeted backfill] Savant xERA unavailable ({_se}); using ERA for xERA.")
                _savant_tb = pd.DataFrame(columns=["player_id", "xERA"])
                _savant_tb["player_id"] = pd.Series(dtype=int)
                _savant_tb["xERA"] = pd.Series(dtype=float)
            _rows_to_file = []
            _seen_norm = set()
            _id_ok = 0
            _stat_ok = 0
            _tb_no_mlb_pitcher_id = []
            _tb_id_no_reg_season_stats = []  # (probable_name, mlb_id, api_full_name)
            for _pn in _unresolved_before:
                print(f"  [Targeted backfill] Lookup MLB id for probable: '{_pn}'")
                _pid, _pfull = _mlb_targeted_resolve_pitcher_id(_pn)
                if _pid is None:
                    _tb_no_mlb_pitcher_id.append(_pn)
                    print(
                        f"    → CLASS: no_MLB_pitcher_id — MLB search did not yield a P match for '{_pn}' "
                        "(bad string / non-roster / lookup miss; not a scraper failure)"
                    )
                    continue
                _id_ok += 1
                print(f"    → MLB id {_pid} ({_pfull})")
                _raw_row = None
                _used_season = None
                for _try_y in (_yr, _yr - 1):
                    _raw_row = _mlb_targeted_fetch_player_season_pitching(_pid, _try_y)
                    if _raw_row is not None:
                        _used_season = _try_y
                        break
                if _raw_row is None:
                    _tb_id_no_reg_season_stats.append((_pn, _pid, _pfull or ""))
                    print(
                        f"    → CLASS: no_MLB_reg_season_pitching_stats — id={_pid} "
                        f"(tried seasons {_yr}, {_yr - 1}); "
                        "will try local pitcher_stats row next (API name / schedule name); else league average"
                    )
                    continue
                _stat_ok += 1
                print(
                    f"    → season pitching OK (season={_used_season}, "
                    f"IP={_raw_row.get('IP')}, ERA={_raw_row.get('ERA')}, WHIP={_raw_row.get('WHIP')})"
                )
                _csv_r = _targeted_pitcher_csv_row_from_mlb_row(_raw_row, _savant_tb, _min_ip_tb)
                _nk = str(_csv_r.get("Name") or "").strip()
                if not _nk or _nk in _seen_norm:
                    if _nk and _nk in _seen_norm:
                        print(
                            f"    → CLASS: stats_ok_duplicate_pitcher — skip upsert (norm '{_nk}' already queued); "
                            "if name still fails local match, check alias/schedule spelling"
                        )
                    continue
                _seen_norm.add(_nk)
                _rows_to_file.append(_csv_r)
            if _rows_to_file:
                _upsert_pitcher_stats_rows(_rows_to_file)
                print(f"  Upserted {len(_rows_to_file)} pitcher row(s) into {STATS_FILE}")
                try:
                    stats_df = DataManager.load_pitcher_stats()
                except Exception:
                    pass
            # Final local tier: MLB id + no reg-season stats — try existing pitcher_stats row via API/schedule name.
            _tb_local_recovered = []
            for _pn_lr, _pid_lr, _pfull_lr in _tb_id_no_reg_season_stats:
                if _pitcher_resolvable_locally_without_league_average(
                    _pn_lr, stats_df, manual_fallback_df, _alias_audit
                ):
                    continue
                _hit_key = None
                _hit_via = None
                for _cand, _via in ((_pfull_lr, "mlb_api_fullName"), (_pn_lr, "schedule_name")):
                    if not _cand or not str(_cand).strip():
                        continue
                    _k2 = _local_pitcher_stats_index_key(_cand, stats_df, _alias_audit)
                    if _k2 is not None:
                        _hit_key, _hit_via = _k2, _via
                        break
                if _hit_key is not None:
                    RUNTIME_TARGETED_LOCAL_RECOVERY_ALIASES[_pn_lr.strip().lower()] = _hit_key
                    _tb_local_recovered.append((_pn_lr, _hit_key, _hit_via, _pid_lr))
                    print(
                        f"  [Targeted backfill] CLASS: local_existing_row_recovery — "
                        f"schedule '{_pn_lr}' → pitcher_stats index '{_hit_key}' "
                        f"(via {_hit_via}; MLB id {_pid_lr}; not an MLB stat upsert)"
                    )
            _tb_id_no_reg_still = [
                t for t in _tb_id_no_reg_season_stats if t[0] not in {x[0] for x in _tb_local_recovered}
            ]
            print("  --- Targeted backfill classification (attempted probables) ---")
            print(
                f"  No MLB pitcher id: {len(_tb_no_mlb_pitcher_id)}/{_n_attempt} — "
                f"{', '.join(_tb_no_mlb_pitcher_id) if _tb_no_mlb_pitcher_id else '(none)'}"
            )
            print(
                f"  MLB id OK, no reg-season pitching stats ({_yr}/{_yr - 1}) — pre-local check: "
                f"{len(_tb_id_no_reg_season_stats)}/{_n_attempt} — "
                f"{', '.join(f'{a} (id {b})' for a, b, _c in _tb_id_no_reg_season_stats) if _tb_id_no_reg_season_stats else '(none)'}"
            )
            print(
                f"  Local CSV recovery (existing pitcher_stats row; runtime alias only): "
                f"{len(_tb_local_recovered)}/{_n_attempt} — "
                f"{', '.join(f'{a}→{b}' for a, b, _v, _p in _tb_local_recovered) if _tb_local_recovered else '(none)'}"
            )
            print(
                f"  After local recovery, still no pitcher_stats row (league average): "
                f"{len(_tb_id_no_reg_still)}/{_n_attempt} — "
                f"{', '.join(f'{a} (id {b})' for a, b, _c in _tb_id_no_reg_still) if _tb_id_no_reg_still else '(none)'}"
            )
            print(
                f"  Per-player counts: id resolved {_id_ok}/{_n_attempt}, "
                f"stat pulls OK {_stat_ok}/{_n_attempt}, rows queued for upsert {len(_rows_to_file)}"
            )
            # Names that were unresolved pre-backfill but now resolve locally (ground truth after CSV upsert).
            _backfilled_probables = [
                _pn
                for _pn in _unresolved_before
                if _pitcher_resolvable_locally_without_league_average(
                    _pn, stats_df, manual_fallback_df, _alias_audit
                )
            ]
            _m_ok = len(_backfilled_probables)
            print(
                f"  Resolved locally after targeted lane (MLB stat upsert + local CSV recovery): {_m_ok}/{_n_attempt} — "
                f"{', '.join(_backfilled_probables) if _backfilled_probables else '(none)'}"
            )
            _still_unresolved = [
                _nm
                for _nm in _probable_sorted
                if not _pitcher_resolvable_locally_without_league_average(
                    _nm, stats_df, manual_fallback_df, _alias_audit
                )
            ]
            _k_remain = len(_still_unresolved)
            print(f"  Slate still unresolved (league-average at match time): {_k_remain} — "
                  f"{', '.join(_still_unresolved) if _still_unresolved else '(none)'}")
            if _still_unresolved:
                print("  Still unresolved names:")
                for _su in _still_unresolved:
                    print(f"    - {_su}")
            _no_id_set = set(_tb_no_mlb_pitcher_id)
            _no_stats_still_set = {t[0] for t in _tb_id_no_reg_still}
            _local_rec_set = {t[0] for t in _tb_local_recovered}
            _still_from_tb = [n for n in _still_unresolved if n in _unresolved_before]
            if _still_from_tb:
                print("  Reasons for targeted-list names still on league-average path:")
                for _sn in _still_from_tb:
                    if _sn in _no_id_set:
                        print(f"    - {_sn}: no_MLB_pitcher_id")
                    elif _sn in _local_rec_set:
                        print(f"    - {_sn}: internal error (local recovery should have resolved; check logs)")
                    elif _sn in _no_stats_still_set:
                        print(
                            f"    - {_sn}: no_MLB_reg_season_pitching_stats_and_no_local_row "
                            f"(prospect / spring — league average expected)"
                        )
                    else:
                        print(f"    - {_sn}: other (see per-name lines above; e.g. duplicate norm / edge match)")
        except Exception as _tbe:
            print(f"  Targeted resolver error (independent path; full-file SAFE MODE unchanged): {_tbe}")
            print(f"  Successfully resolved 0 of {_n_attempt} attempted: (none)")
            print(
                f"  Unresolved {_n_attempt} name(s) remain: {', '.join(_unresolved_before)}"
            )
            if _unresolved_before:
                print("  Still unresolved names:")
                for _su in _unresolved_before:
                    print(f"    - {_su}")
        print("---------------------------------------------------------\n")
    else:
        print("--- TARGETED PROBABLE-PITCHER BACKFILL (independent MLB resolver) ---\n"
              "  Skipped: no unresolved probable pitchers.\n"
              "---------------------------------------------------------\n")

    # ================================
    # 📅 OPENING DAY PREFLIGHT (readiness)
    # ================================
    trusted_total_source_map = locals().get("trusted_total_source_map", {}) or {}
    if not isinstance(trusted_total_source_map, dict):
        trusted_total_source_map = {}

    def _is_missing_probable_pitcher(v):
        s = str(v) if v is not None else ""
        return (not s.strip()) or s.strip() == "TBD"

    # Build alias reverse map once (variation -> official), for deterministic preflight classification.
    _alias_reverse = {}
    _alias_path = os.path.join(DATA_DIR, "pitcher_aliases.json")
    if os.path.exists(_alias_path):
        try:
            with open(_alias_path, "r") as f:
                _alias_map = json.load(f)
            for _official, _variations in (_alias_map or {}).items():
                if isinstance(_variations, list):
                    for _a in _variations:
                        if _a:
                            _alias_reverse[str(_a).strip().lower()] = str(_official).strip().lower()
                elif isinstance(_variations, str) and _variations.strip():
                    _alias_reverse[_variations.strip().lower()] = str(_official).strip().lower()
        except Exception:
            _alias_reverse = {}

    _league_avg_set = {"League Avg Away", "League Avg Home"}

    def _would_fuzzy_match(pitcher_name_used: str) -> bool:
        if stats_df is None or not isinstance(stats_df, pd.DataFrame) or stats_df.empty:
            return False
        if not pitcher_name_used:
            return False
        clean_name = DataManager.normalize_name(pitcher_name_used)
        if not clean_name:
            return False
        choices = stats_df.index.tolist()
        result = process.extractOne(clean_name, choices, scorer=fuzz.WRatio, score_cutoff=NAME_MATCH_THRESHOLD)
        if not result:
            return False
        best_match, score = result[0], result[1]
        try:
            return clean_name.split()[-1] == best_match.split()[-1]
        except Exception:
            return False

    def _classify_pitcher_used(pitcher_name_used: str) -> str:
        """
        Deterministic preflight classification (logging only):
        - exact: found in pitcher_stats.csv index (including fuzzy-success as "exact CSV found")
        - alias: matched via pitcher_aliases.json to an index entry
        - league_average: would use league-average dict fallback (no live scrape)
        - league_avg: league-average sentinel pitcher (or equivalent)
        """
        if pitcher_name_used in _league_avg_set:
            return "league_avg"

        if stats_df is not None and isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
            clean = DataManager.normalize_name(pitcher_name_used)
            if clean in stats_df.index:
                return "exact"

        # alias?
        alias_official = _alias_reverse.get(str(pitcher_name_used).strip().lower())
        if alias_official:
            alias_clean = DataManager.normalize_name(alias_official)
            if stats_df is not None and isinstance(stats_df, pd.DataFrame) and not stats_df.empty and alias_clean in stats_df.index:
                return "alias"

        # manual fallback pitchers?
        if manual_fallback_df is not None and isinstance(manual_fallback_df, pd.DataFrame) and not manual_fallback_df.empty:
            if str(pitcher_name_used).strip().lower() in manual_fallback_df.index:
                return "manual_fallback"

        # fuzzy match success?
        if _would_fuzzy_match(pitcher_name_used):
            return "exact"

        return "league_average"

    opening_total_games = len(games) if games is not None else 0
    opening_games_missing_probable_pitchers = 0
    opening_exact_pitchers_found = 0
    opening_alias_pitchers_found = 0
    opening_scrape_fallback_pitchers = 0
    opening_league_avg_fallback_pitchers = 0
    opening_missing_public_betting = 0
    opening_missing_trusted_totals = 0
    opening_fully_ready_games = 0
    opening_degraded_games = 0

    opening_missing_pitchers_by_name = set()
    opening_fallback_pitchers_by_name = set()
    opening_games_missing_required_inputs = []

    for game in games:
        away_team = safe_get(game, "away_name", "Away Team")
        home_team = safe_get(game, "home_name", "Home Team")
        away_norm = normalize_team_name(away_team)
        home_norm = normalize_team_name(home_team)
        game_key = f"{away_norm} @ {home_norm}"

        away_raw = safe_get(game, "away_probable_pitcher", "TBD")
        home_raw = safe_get(game, "home_probable_pitcher", "TBD")
        away_missing = _is_missing_probable_pitcher(away_raw)
        home_missing = _is_missing_probable_pitcher(home_raw)

        away_used = "League Avg Away" if away_missing else str(away_raw).strip()
        home_used = "League Avg Home" if home_missing else str(home_raw).strip()

        if away_missing or home_missing:
            opening_games_missing_probable_pitchers += 1
            if away_missing:
                opening_missing_pitchers_by_name.add("League Avg Away")
            if home_missing:
                opening_missing_pitchers_by_name.add("League Avg Home")

        away_type = _classify_pitcher_used(away_used)
        home_type = _classify_pitcher_used(home_used)

        if away_type == "exact":
            opening_exact_pitchers_found += 1
        elif away_type == "alias":
            opening_alias_pitchers_found += 1
        elif away_type == "manual_fallback":
            pass
        elif away_type == "league_average":
            opening_league_avg_fallback_pitchers += 1
            opening_fallback_pitchers_by_name.add(away_used)
        elif away_type == "league_avg":
            opening_league_avg_fallback_pitchers += 1
            opening_fallback_pitchers_by_name.add(away_used)

        if home_type == "exact":
            opening_exact_pitchers_found += 1
        elif home_type == "alias":
            opening_alias_pitchers_found += 1
        elif home_type == "manual_fallback":
            pass
        elif home_type == "league_average":
            opening_league_avg_fallback_pitchers += 1
            opening_fallback_pitchers_by_name.add(home_used)
        elif home_type == "league_avg":
            opening_league_avg_fallback_pitchers += 1
            opening_fallback_pitchers_by_name.add(home_used)

        public = public_betting_data.get(game_key) if isinstance(public_betting_data, dict) else None
        public_missing = public is None or public == {}
        if public_missing:
            opening_missing_public_betting += 1

        trow = trusted_total_source_map.get(game_key) if isinstance(trusted_total_source_map, dict) else None
        trusted_exists = isinstance(trow, dict) and trow.get("total_line") is not None
        if not trusted_exists:
            opening_missing_trusted_totals += 1

        pitchers_ready = (away_type in {"exact", "alias"}) and (home_type in {"exact", "alias"})
        totals_ready = trusted_exists
        # Public betting is optional; readiness = pitchers + trusted totals only.
        if pitchers_ready and totals_ready:
            opening_fully_ready_games += 1
        else:
            opening_degraded_games += 1
            if not totals_ready or not pitchers_ready:
                opening_games_missing_required_inputs.append(game_key)

    # Print compact preflight block
    print("\n--- OPENING DAY PREFLIGHT ---")
    print(f"  Total games found: {opening_total_games}")
    print(f"  Games missing probable pitchers: {opening_games_missing_probable_pitchers}")
    print(f"  Pitchers found by exact match: {opening_exact_pitchers_found}")
    print(f"  Pitchers found by alias match: {opening_alias_pitchers_found}")
    print(f"  Live scrape fallback pitchers: {opening_scrape_fallback_pitchers} (disabled)")
    print(f"  Pitchers using league-average fallback: {opening_league_avg_fallback_pitchers}")
    print(f"  Missing public betting: {opening_missing_public_betting}")
    print(f"  Missing trusted totals: {opening_missing_trusted_totals}")
    print(f"  Games fully ready: {opening_fully_ready_games}")
    print(f"  Games degraded: {opening_degraded_games}")
    print("-------------------------------\n")

    # LIVE TOTAL BLOCKERS tracking (logging only)
    live_total_scrambled_book = 0
    live_total_empty_book = 0
    live_total_unrealistic_total = 0
    live_total_missing_total_line = 0
    live_total_real_total_pass = 0
    live_total_scrambled_book_keys = []
    live_total_empty_book_keys = []

    # Runtime pitcher-resolution outcome logging (logging only)
    rt_exact = 0
    rt_alias = 0
    rt_manual_fallback = 0
    rt_fuzzy = 0
    rt_scrape_success = 0
    rt_scrape_fail = 0
    rt_league_average = 0

    _alias_reverse_runtime = locals().get("_alias_reverse", {}) or {}
    _league_avg_clean = {"league avg away", "league avg home"}

    def _low_ip_from_stats(s):
        try:
            return bool(s.get("LowIP", False)) if isinstance(s, dict) else bool(s["LowIP"])
        except Exception:
            return False

    def _to_dict(s):
        if s is None:
            return {}
        if isinstance(s, dict):
            return s
        try:
            if isinstance(s, pd.Series):
                return s.to_dict()
        except Exception:
            pass
        try:
            if hasattr(s, "to_dict"):
                return s.to_dict()
        except Exception:
            pass
        return {}

    def _num_close(a, b, tol=1e-2):
        try:
            return abs(float(a) - float(b)) <= tol
        except Exception:
            return False

    def _stats_equal(candidate, resolved):
        cd = _to_dict(candidate)
        rd = _to_dict(resolved)
        if not rd:
            return False
        if "xERA" in cd and "xERA" in rd and not _num_close(cd.get("xERA"), rd.get("xERA"), tol=0.01):
            return False
        if "WHIP" in cd and "WHIP" in rd and not _num_close(cd.get("WHIP"), rd.get("WHIP"), tol=0.01):
            return False
        if "LowIP" in cd and "LowIP" in rd and bool(cd.get("LowIP")) != bool(rd.get("LowIP")):
            return False
        if "IP" in cd and "IP" in rd and cd.get("IP") is not None and rd.get("IP") is not None:
            try:
                if not _num_close(cd.get("IP"), rd.get("IP"), tol=0.5):
                    return False
            except Exception:
                pass
        return True

    def _runtime_pitcher_path(pitcher_name_used: str, resolved_obj, alias_used: bool = False):
        """
        Runtime classification of the *actual* resolution lane, based on:
        - whether alias mapping triggered for this specific call (alias_used flag)
        - which lane would be chosen by the same DataManager.match_pitcher_row decision order
        - only using returned-object metadata to detect scrape_success (scraped dict includes 'Name')
        """
        clean = DataManager.normalize_name(pitcher_name_used or "")
        resolved_dict = _to_dict(resolved_obj)

        # League-average sentinel shortcut (direct support in DataManager)
        if clean in _league_avg_clean:
            return "league_average"

        stats_ok = stats_df is not None and isinstance(stats_df, pd.DataFrame) and not stats_df.empty
        manual_ok = manual_fallback_df is not None and isinstance(manual_fallback_df, pd.DataFrame) and not manual_fallback_df.empty

        pitcher_key = str(pitcher_name_used or "").strip().lower()
        alias_official = _alias_reverse_runtime.get(pitcher_key)
        alias_clean = DataManager.normalize_name(alias_official) if alias_official else ""

        # DataManager applies alias mapping by swapping clean_name before direct/fuzzy matches.
        effective_clean = alias_clean if (alias_used and alias_clean) else clean

        exact_possible = stats_ok and effective_clean in stats_df.index
        alias_possible = bool(alias_used and alias_clean) and stats_ok and alias_clean in stats_df.index

        # Manual fallback key is based on the original input name (DataManager uses pitcher_name.lower()).
        manual_possible = manual_ok and pitcher_key in manual_fallback_df.index

        # Fuzzy match attempt outcome prediction (same last-name guard).
        fuzzy_possible = False
        if stats_ok and not exact_possible:
            try:
                choices = stats_df.index.tolist()
                res = process.extractOne(effective_clean, choices, scorer=fuzz.WRatio, score_cutoff=NAME_MATCH_THRESHOLD)
                if res:
                    best_match, score = res[0], res[1]
                    if effective_clean.split()[-1] == best_match.split()[-1]:
                        fuzzy_possible = True
            except Exception:
                fuzzy_possible = False

        # Now decide the lane:
        if alias_possible:
            return "alias"
        if exact_possible:
            return "exact"
        if manual_possible:
            return "manual_fallback"
        if fuzzy_possible:
            return "fuzzy"

        # Live predictor: no scrape in match_pitcher_row; unresolved -> league-average dict (no Name key).
        # OG_TEST_SCRAPER direct scrape returns dict with Name; would not appear from match_pitcher_row in live.
        if isinstance(resolved_dict, dict) and resolved_dict.get("Name"):
            return "scrape_success"
        return "league_average"

    def _rt_inc(path: str):
        nonlocal rt_exact, rt_alias, rt_manual_fallback, rt_fuzzy, rt_scrape_success, rt_scrape_fail, rt_league_average
        if path == "exact":
            rt_exact += 1
        elif path == "alias":
            rt_alias += 1
        elif path == "manual_fallback":
            rt_manual_fallback += 1
        elif path == "fuzzy":
            rt_fuzzy += 1
        elif path == "scrape_success":
            rt_scrape_success += 1
        elif path == "scrape_fail":
            rt_scrape_fail += 1
        elif path == "league_average":
            rt_league_average += 1

    def _fallback_used_from_path(path: str) -> bool:
        return path in {"manual_fallback", "league_average"}

    for game in games:
        try:
            home_team = safe_get(game, 'home_name', 'Home Team')
            away_team = safe_get(game, 'away_name', 'Away Team')
            vegas_line, odds_info = VegasLines.get_vegas_line(home_team, away_team, odds_map)
            print(f"[ODDS] Game: {away_team} @ {home_team}")
            print(f"[ODDS]   Lookup key: {odds_info.get('_lookup_key', '?')}")
            print(f"[ODDS]   Match found in odds_map: {odds_info.get('_match_found', '?')}")
            print(f"[ODDS]   odds_info: total_line={odds_info.get('total_line')}, over_juice={odds_info.get('over_juice')}, under_juice={odds_info.get('under_juice')}, book={repr(odds_info.get('book'))}")
            print(f"[ODDS]   Source: {odds_info.get('_source', '?')}")
            print(f"[ODDS]   Total status: {'REAL sportsbook total' if odds_info.get('_has_real_total', False) else 'FALLBACK total (missing market totals)'}")

            # LIVE TOTAL BLOCKERS tracking: only count odds_map/API rows (exclude manual_totals_csv + no-match default fallback)
            _src = str(odds_info.get("_source", ""))
            if odds_info.get("_match_found", False) and _src in {"Odds API", "8.5 fallback", "fallback (scrambled book)"}:
                if odds_info.get("_blocker_scrambled_book"):
                    live_total_scrambled_book += 1
                    k = odds_info.get("_lookup_key")
                    if k:
                        live_total_scrambled_book_keys.append(k)
                elif odds_info.get("_blocker_empty_book"):
                    live_total_empty_book += 1
                    k = odds_info.get("_lookup_key")
                    if k:
                        live_total_empty_book_keys.append(k)
                elif odds_info.get("_blocker_unrealistic_total"):
                    live_total_unrealistic_total += 1
                elif odds_info.get("_blocker_missing_total_line"):
                    live_total_missing_total_line += 1
                elif odds_info.get("_blocker_real_total_pass"):
                    live_total_real_total_pass += 1

            # --- starters must be defined BEFORE we score lineups
            away_pitcher = safe_get(game, 'away_probable_pitcher', 'TBD')
            home_pitcher = safe_get(game, 'home_probable_pitcher', 'TBD')

            # ✅ Use fallback league average pitchers instead of skipping
            if not away_pitcher.strip() or away_pitcher == "TBD":
                print(f"⚠️ Missing away pitcher for {away_team} — using League Avg Away")
                away_pitcher = "League Avg Away"
            if not home_pitcher.strip() or home_pitcher == "TBD":
                print(f"⚠️ Missing home pitcher for {home_team} — using League Avg Home")
                home_pitcher = "League Avg Home"

            print(f"🎯 Matchup: {away_team} ({away_pitcher}) vs {home_team} ({home_pitcher})")

            venue = game.get("venue_name", "Unknown")
            park_factors = PARK_FACTORS.get(venue, PARK_FACTORS['Unknown'])

            print(f"\n🔍 Processing: {away_team} @ {home_team}")
            print(f"🧪 Matching: Away = {away_pitcher} | Home = {home_pitcher}")

            away_impact = 0.0
            home_impact = 0.0
            away_scope = "none"
            home_scope = "none"
            lineup_delta = 0.0

            # --- score lineups vs the real opposing starter hand (safe) ---
            try:
                # teams must be full, lowercase names to match the CSV
                away_best9 = lineups.get_team_best9(away_team.lower())
                home_best9 = lineups.get_team_best9(home_team.lower())

                # Determine the hand each offense will face
                try:
                    home_starter_hand = Batters.get_pitcher_hand(home_pitcher)  # away lineup faces HOME starter
                except Exception:
                    home_starter_hand = None
                try:
                    away_starter_hand = Batters.get_pitcher_hand(away_pitcher)  # home lineup faces AWAY starter
                except Exception:
                    away_starter_hand = None

                away_impact, home_impact, away_scope, home_scope = safe_lineup_impacts(
                    lineup_obj=lineups,
                    away_lineup=away_best9,
                    home_lineup=home_best9,
                    away_pitcher_hand=away_starter_hand,
                    home_pitcher_hand=home_starter_hand,
                    logger=None
                )

                # positive means home lineup projects stronger than away
                lineup_delta = float(home_impact) - float(away_impact)

                # small, controlled nudge to the total (tune 0.1–0.5 after backtests)
                vegas_line_adj = float(vegas_line) + 0.3 * lineup_delta

                print(
                    f"🧮 Lineup impacts → AWAY {away_team}: {away_impact:.3f} ({away_scope}) | "
                    f"HOME {home_team}: {home_impact:.3f} ({home_scope}) | Δ={lineup_delta:+.3f}"
                )
            except Exception as e:
                print(f"⚠️ Lineup impact error: {e}")
                lineup_delta = 0.0
                vegas_line_adj = float(vegas_line)

            # Runtime game key for pitcher-resolution logging (no logic change)
            away_norm_rt = normalize_team_name(away_team)
            home_norm_rt = normalize_team_name(home_team)
            game_key_rt = f"{away_norm_rt} @ {home_norm_rt}"

            _alias_len_before_away = len(alias_log)
            away_stats = DataManager.match_pitcher_row(stats_df, away_pitcher, alias_log=alias_log)
            away_alias_used = len(alias_log) > _alias_len_before_away

            _alias_len_before_home = len(alias_log)
            home_stats = DataManager.match_pitcher_row(stats_df, home_pitcher, alias_log=alias_log)
            home_alias_used = len(alias_log) > _alias_len_before_home

            # 🧠 Log fallback usage if LowIP
            for name, stats in [(away_pitcher, away_stats), (home_pitcher, home_stats)]:
                if isinstance(stats, dict) and stats.get("LowIP", False):
                    print(f"⚠️ Using fallback stats for: {name}")

            # Runtime pitcher-resolution outcome logging (logging only)
            away_path = _runtime_pitcher_path(away_pitcher, away_stats, alias_used=away_alias_used)
            away_low = _low_ip_from_stats(away_stats)
            _rt_inc(away_path)
            print(
                "[RUNTIME PITCHER RESOLUTION] "
                f"game={game_key_rt} | pitcher={away_pitcher} | path={away_path} | "
                f"fallback_used={_fallback_used_from_path(away_path)} | low_ip={away_low}"
            )

            home_path = _runtime_pitcher_path(home_pitcher, home_stats, alias_used=home_alias_used)
            home_low = _low_ip_from_stats(home_stats)
            _rt_inc(home_path)
            print(
                "[RUNTIME PITCHER RESOLUTION] "
                f"game={game_key_rt} | pitcher={home_pitcher} | path={home_path} | "
                f"fallback_used={_fallback_used_from_path(home_path)} | low_ip={home_low}"
            )

            # which hand each starter throws
            try:
                away_hand = Batters.get_pitcher_hand(away_pitcher)  # hand of AWAY starter (faces HOME offense)
            except Exception:
                away_hand = None
            try:
                home_hand = Batters.get_pitcher_hand(home_pitcher)  # hand of HOME starter (faces AWAY offense)
            except Exception:
                home_hand = None

            # defaults in case batter table isn't loaded or hand unknown
            home_off = {"mult": 1.0, "pop": "none"}
            away_off = {"mult": 1.0, "pop": "none"}

            if not BATTER_DF.empty and (away_hand or home_hand):
                try:
                    # HOME offense vs AWAY pitcher's hand
                    if away_hand:
                        home_off = Batters.offense_vs_hand_dict(
                            BATTER_DF, home_team, away_hand, lineup_names=None
                        )
                    # AWAY offense vs HOME pitcher's hand
                    if home_hand:
                        away_off = Batters.offense_vs_hand_dict(
                            BATTER_DF, away_team, home_hand, lineup_names=None
                        )

                    print(
                        f"🧮 Offense vs hand → HOME {home_team} vs {away_hand or '?'}: "
                        f"x{home_off.get('mult',1.0):.3f} ({home_off.get('pop','none')}); "
                        f"AWAY {away_team} vs {home_hand or '?'}: "
                        f"x{away_off.get('mult',1.0):.3f} ({away_off.get('pop','none')})"
                    )
                except Exception as e:
                    print(f"⚠️ Batter split adjustment failed: {e}")

            # Per-team offense mult for project_team_runs: Batters.offense_vs_hand_dict "mult" when available, else 1.0
            away_offense_mult = max(OFFENSE_MULT_MIN, min(OFFENSE_MULT_MAX, float(away_off.get("mult", 1.0))))
            home_offense_mult = max(OFFENSE_MULT_MIN, min(OFFENSE_MULT_MAX, float(home_off.get("mult", 1.0))))

            if isinstance(away_stats, dict) and away_stats.get('LowIP', False):
                unmatched_pitchers.add(away_pitcher)
            if isinstance(home_stats, dict) and home_stats.get('LowIP', False):
                unmatched_pitchers.add(home_pitcher)

            print(f"🧠 Pitcher Check: {away_pitcher} → {away_stats if away_stats is not None else 'None'}")
            print(f"🧠 Pitcher Check: {home_pitcher} → {home_stats if home_stats is not None else 'None'}")

            # ✅ Check pitcher data validity
            if (away_stats is None or not isinstance(away_stats, (dict, pd.Series)) or
                home_stats is None or not isinstance(home_stats, (dict, pd.Series))):
                print(f"⚠️ Skipping - missing or invalid pitcher data")
                print(f"❓ away_stats = {away_stats}")
                print(f"❓ home_stats = {home_stats}")
                continue

            bullpen_home = BullpenManager.get_bullpen_stats(home_team)
            bullpen_away = BullpenManager.get_bullpen_stats(away_team)
            velo_drop_away = velocity_tracker.get_velocity_drop(away_pitcher)
            velo_drop_home = velocity_tracker.get_velocity_drop(home_pitcher)
            print(f"🚀 Velo Drop → Away: {velo_drop_away}, Home: {velo_drop_home}")

            # 🧠 Normalize game name and look up public betting data
            away_norm = normalize_team_name(away_team)
            home_norm = normalize_team_name(home_team)
            game_key = f"{away_norm} @ {home_norm}"

            print(f"🧩 Game Key: {game_key}")
            print(f"📊 Public Data Found: {game_key in public_betting_data}")

            public = public_betting_data.get(game_key)
            if public is None:
                print(f"⚠️ No public betting data for: {game_key}")
                public = {}

            game_name = f"{away_team} @ {home_team}"

            game_data = {
                'Game': game_name,
                'Venue': venue,
                'Pitchers': f"{away_pitcher} vs {home_pitcher}",
                'Datetime': safe_get(game, 'game_datetime', datetime.utcnow().isoformat()),
                'vegas_line': vegas_line,
                'Total_Is_Real': bool(odds_info.get('_has_real_total', False)),
                'Odds_Line': odds_info.get('total_line', 8.5) if odds_info.get('_has_real_total', False) else '',
                'Over_Juice': odds_info.get('over_juice', -110),
                'Under_Juice': odds_info.get('under_juice', -110),
                'Odds_Book': odds_info.get('book', ''),
                'Total_Line_Source': str(odds_info.get("_source", "") or ""),
                'Odds_ML_Home': odds_info.get('ml_home'),
                'Odds_ML_Away': odds_info.get('ml_away'),
                'Market_Source': odds_source if 'odds_source' in locals() else '',
                'Captured_Book': odds_info.get('book', ''),
                'Captured_Total': odds_info.get('total_line', 8.5) if odds_info.get('_has_real_total', False) else '',
                'Captured_ML_Home': odds_info.get('ml_home'),
                'Captured_ML_Away': odds_info.get('ml_away'),
                'Fired_Play': False,
                'OU_Fired': False,
                'ML_Fired': False,
                'Trigger_Tags': '',
                'No_Fire_Reason': '',
                'No_Fire_OU_Reason': '',
                'No_Fire_ML_Reason': '',
                'ML_Quality_Flag': '',
                'Model_Notes': '',
                'Confidence_Tier': '',
                'Edge_Tier': '',
                'Bet_Type': 'total',
                'Side': '',
                'Play_Status': '',
                'Bettable': False,
                'Line_Status': '',
                'Fallback_Used': False,
                'Data_Quality_Flag': '',
                'Bet_Line': odds_info.get('total_line', 8.5),
                'Closing_Line': '',
                'CLV': '',
                'CLV_Result': '',
                'OU_Result': 'PENDING',
                'ML_Result': 'PENDING',
            }

            # 🔮 Run prediction (compare projection to actual Vegas line; do not pass lineup-adjusted line)
            data_quality_degraded = (
                "League Avg" in (away_pitcher or "")
                or "League Avg" in (home_pitcher or "")
                or bool(safe_get(away_stats, "LowIP", False))
                or bool(safe_get(home_stats, "LowIP", False))
            )
            proj = generate_prediction(
                away_stats=away_stats,
                home_stats=home_stats,
                bullpen_home=bullpen_home,
                bullpen_away=bullpen_away,
                velo_drop_away=velo_drop_away,
                velo_drop_home=velo_drop_home,
                park_factors=park_factors,
                vegas_data=vegas_line,
                public_data=public,
                away_lineup_impact=float(away_impact),
                home_lineup_impact=float(home_impact),
                away_offense_mult=away_offense_mult,
                home_offense_mult=home_offense_mult,
                has_real_total=bool(odds_info.get("_has_real_total", False)),
                data_quality_degraded=data_quality_degraded,
            )

            if proj.get("skip"):
                print(f"⏭️ Skipping {game_name} due to bad public data")
                continue

            prediction = proj["prediction"]
            confidence = proj["confidence"]
            total_open = proj["total_open"]
            total_current = proj["total_current"]
            projected_total = proj["projected_total"]
            away_runs = proj["away_runs"]
            home_runs = proj["home_runs"]
            edge = proj["edge"]
            recommended_units = proj["recommended_units"]

            # Offense strength already in projection via away_offense_mult / home_offense_mult; no post-hoc bat_mult

            confidence = max(0.0, min(1.0, confidence))
            has_real_total = bool(odds_info.get("_has_real_total", False))
            if not has_real_total:
                prediction = "NO BET (fallback total only)"

            # Optional: scale units by lineup conviction
            try:
                units_mult = 1.0 + 0.25 * abs(float(lineup_delta))
                units_mult = min(units_mult, 1.50)
                sized_units = round(recommended_units * units_mult, 2)
            except Exception:
                sized_units = recommended_units

            game_data.update({
                "Projected_Total": projected_total,
                "Away_Runs": away_runs,
                "Home_Runs": home_runs,
                "Vegas_Line": total_current if (total_current is not None and total_current != 0) else vegas_line,
                "Edge": edge,
                "Prediction": prediction,
                "Confidence": f"{confidence:.0%}",
                "Confidence_Value": confidence,
                "Units": sized_units,
                "Line_Open": total_open,
                "Line_Current": total_current,
                "xERA": f"{safe_get(away_stats, 'xERA', 'N/A')}/{safe_get(home_stats, 'xERA', 'N/A')}",
                "WHIP": f"{safe_get(away_stats, 'WHIP', 'N/A')}/{safe_get(home_stats, 'WHIP', 'N/A')}",
                "Bullpen": f"{safe_float(safe_get(bullpen_away, 'ERA', 0)):.2f}/{safe_float(safe_get(bullpen_home, 'ERA', 0)):.2f}",
                "Velo Drops": f"{safe_float(velo_drop_away):.1f}/{safe_float(velo_drop_home):.1f}",
                "Velo": round((safe_float(velo_drop_home) + safe_float(velo_drop_away)) / 2, 2),
                "Park Factor": f"{park_factors[0]}/{park_factors[1]}",
                "Relievers": f"{safe_get(bullpen_away, 'Relievers', '?')}/{safe_get(bullpen_home, 'Relievers', '?')}",
                "VeloDrop_Away": round(safe_float(velo_drop_away), 1),
                "VeloDrop_Home": round(safe_float(velo_drop_home), 1),
            })

            # 💵 MONEYLINE PREDICTION (independent of trusted O/U — uses starter/bullpen model + optional public ML line)
            home_ml_data = get_team_ml_data(home_team, home_pitcher)
            away_ml_data = get_team_ml_data(away_team, away_pitcher)

            home_win_prob, away_win_prob = calculate_team_win_probability(home_ml_data, away_ml_data)

            try:
                odds_str = public.get("ML_Home", "-130") if isinstance(public, dict) else "-130"
                odds_value = float(odds_str) if isinstance(odds_str, (int, float)) else float(str(odds_str).strip())
                if odds_value < 0:
                    implied_home = abs(odds_value) / (100 + abs(odds_value))
                else:
                    implied_home = 100 / (100 + odds_value)
            except Exception:
                implied_home = 0.53

            implied_away = 1 - implied_home

            home_kelly = calculate_kelly_units(home_win_prob, implied_home)
            away_kelly = calculate_kelly_units(away_win_prob, implied_away)

            if home_win_prob > away_win_prob:
                ml_pick = f"{home_team.upper()} ML"
                ml_conf = f"{round(home_win_prob * 100)}%"
                ml_value = f"{round((home_win_prob - implied_home) * 100)}%"
                ml_kelly = f"{round(home_kelly, 2)}u"
            else:
                ml_pick = f"{away_team.upper()} ML"
                ml_conf = f"{round(away_win_prob * 100)}%"
                ml_value = f"{round((away_win_prob - implied_away) * 100)}%"
                ml_kelly = f"{round(away_kelly, 2)}u"

            _league_avg_pitcher = (
                "League Avg" in (away_pitcher or "") or "League Avg" in (home_pitcher or "")
            )
            ml_win_max = max(home_win_prob, away_win_prob)
            # Fire on probability only; League Avg probable = degraded tag, not a hard ML block.
            ml_fired = bool(ml_win_max >= MIN_ML_WIN_PROB_FOR_FIRE)
            ml_quality_flag = "league_avg_pitcher_fallback" if _league_avg_pitcher else ""
            if ml_fired:
                no_fire_ml = ""
            else:
                no_fire_ml = "ml_win_prob_below_threshold"

            game_data.update({
                "ML_Pick": ml_pick,
                "ML_Confidence": ml_conf,
                "ML_Value": ml_value,
                "ML_Kelly_Units": ml_kelly,
                "ML_Fired": ml_fired,
                "No_Fire_ML_Reason": no_fire_ml,
                "ML_Quality_Flag": ml_quality_flag,
            })

            if isinstance(public, dict):
                print(f"Public keys found: {list(public.keys())}")
            else:
                print("Public betting data is missing (NoneType)")

            if public:
                game_data["ou_bets_pct_over"] = public.get("ou_bets_pct_over", "?")
                game_data["ou_bets_pct_under"] = public.get("ou_bets_pct_under", "?")
                game_data["ml_bets_pct_home"] = public.get("ml_bets_pct_home", "?")
                game_data["ml_bets_pct_away"] = public.get("ml_bets_pct_away", "?")
                game_data["total_open"] = public.get("total_open", "?")
                game_data["total_current"] = public.get("total_current", "?")

            is_manual_trusted = (odds_info.get("_source") == "manual_totals_csv") and has_real_total
            fire_threshold = 0.79 if is_manual_trusted else MIN_CONFIDENCE_ALERT
            ou_fired = (confidence >= fire_threshold) and has_real_total
            trigger_tags = "|".join(filter(None, [
                "ou_high_confidence" if ou_fired else None,
                "ml_high_signal" if ml_fired else None,
                "ml_degraded_league_avg_pitcher" if (_league_avg_pitcher and ml_fired) else None,
                "sportsdataio" if (odds_source == "SportsDataIO") else None,
                "odds_api" if (odds_source == "Odds API") else None,
                "fallback_line" if (not has_real_total) else None,
            ]))
            game_data["Fired_Play"] = ou_fired
            game_data["OU_Fired"] = ou_fired
            game_data["Trigger_Tags"] = trigger_tags
            if ou_fired:
                no_fire_ou = ""
            else:
                if not has_real_total:
                    no_fire_ou = "fallback_line_used"
                elif abs(edge) < 1.0:
                    no_fire_ou = "edge_too_small"
                elif confidence < fire_threshold:
                    no_fire_ou = "confidence_below_alert_threshold"
                elif "League Avg" in (away_pitcher or "") or "League Avg" in (home_pitcher or ""):
                    no_fire_ou = "data_quality_degraded"
                else:
                    no_fire_ou = "manual_review"
            game_data["No_Fire_Reason"] = no_fire_ou
            game_data["No_Fire_OU_Reason"] = no_fire_ou
            game_data["Play_Status"] = "BETTABLE" if has_real_total else "PROJECTION_ONLY"
            game_data["Bettable"] = bool(has_real_total)
            game_data["Model_Notes"] = f"edge={edge:.2f}|conf={confidence:.2f}|book={odds_info.get('book', '')}"
            game_data["Confidence_Tier"] = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.60 else "low")
            game_data["Edge_Tier"] = "strong" if abs(edge) >= 2.0 else ("medium" if abs(edge) >= 1.0 else "thin")
            game_data["Bet_Type"] = "total"
            game_data["Side"] = "over" if "OVER" in (prediction or "").upper() else ("under" if "UNDER" in (prediction or "").upper() else "")
            line_status = "market" if bool(odds_info.get("_has_real_total", False)) else "fallback"
            game_data["Line_Status"] = line_status
            game_data["Fallback_Used"] = not bool(odds_info.get("_has_real_total", False))
            dq_parts = []
            if line_status == "fallback":
                dq_parts.append("fallback_line")
            _away_low = bool(safe_get(away_stats, "LowIP", False))
            _home_low = bool(safe_get(home_stats, "LowIP", False))
            if (
                "League Avg" in (away_pitcher or "")
                or "League Avg" in (home_pitcher or "")
                or _away_low
                or _home_low
            ):
                dq_parts.append("fallback_pitcher")
            if _away_low or _home_low:
                dq_parts.append("low_ip")
            game_data["Data_Quality_Flag"] = "|".join(dq_parts)

            results.append(game_data)
            print(f"✅ Prediction: {prediction} | Confidence: {confidence:.0%} | OU_fired={ou_fired} | ML_fired={ml_fired}")
            if ou_fired:
                alerts.append(game_data)
            if ml_fired:
                ml_alerts.append(game_data)

        except Exception as e:
            print(f"❌ Game processing error: {e}")
            traceback.print_exc()
            continue

    # Export: one combined CSV — trusted-total O/U and/or ML_Fired rows (one row per game in `results`).
    export_cols = [
        "Game", "Projected_Total", "Away_Runs", "Home_Runs", "Vegas_Line", "Edge",
        "Prediction", "Confidence", "Units", "Line_Open", "Line_Current",
        "Total_Is_Real", "Odds_Line", "Over_Juice", "Under_Juice", "Odds_Book",
        "Total_Line_Source", "Market_Source", "Captured_Book", "Captured_Total", "Captured_ML_Home", "Captured_ML_Away",
        "Fired_Play", "OU_Fired", "ML_Fired", "Trigger_Tags", "No_Fire_Reason", "No_Fire_OU_Reason", "No_Fire_ML_Reason",
        "Model_Notes",
        "Confidence_Tier", "Edge_Tier", "Bet_Type", "Side", "Play_Status", "Bettable",
        "Line_Status", "Fallback_Used", "Data_Quality_Flag",
        "Bet_Line", "Closing_Line", "CLV", "CLV_Result",
        "OU_Result", "ML_Result",
        "ML_Pick", "ML_Confidence", "ML_Value", "ML_Kelly_Units", "ML_Quality_Flag",
    ]
    eligible_export = [
        r for r in results
        if r.get("Total_Is_Real", False) or r.get("ML_Fired", False)
    ]
    if eligible_export:
        archive_date = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        combined_path = f"{ARCHIVE_DIR}/predictions_{archive_date}.csv"
        combined_df = pd.DataFrame(eligible_export, columns=export_cols)
        combined_df.to_csv(combined_path, index=False)
        print(f"\n💾 Saved {len(eligible_export)} combined row(s) → {combined_path}")
        send_telegram_file(
            combined_path,
            caption=f"📊 Over Gang predictions — {datetime.now().strftime('%b %d')}",
        )
    elif results:
        print("\nℹ️ No export rows (no trusted O/U games and no ML_Fired games).")

    _ml_fired_n = sum(1 for r in results if r.get("ML_Fired"))
    print(f"[READINESS] ML: slate_ml_fired_games={_ml_fired_n} / {len(results)} games processed")

    # Telegram: O/U and ML use separate send loops and distinct message bodies (format_ou_alert vs format_ml_alert).
    # Gating (thresholds TELEGRAM_*_MIN_*): CSV keeps all rows; only high-confidence rows are messaged.
    telegram_ou_alerts = [
        a
        for a in alerts
        if _ou_confidence_percent_for_telegram_gate(a) >= TELEGRAM_OU_MIN_CONFIDENCE_PCT
    ]
    telegram_ml_alerts = [
        a
        for a in ml_alerts
        if _ml_confidence_percent_for_telegram_gate(a) >= TELEGRAM_ML_MIN_CONFIDENCE_PCT
    ]
    # The same game can appear in both lists when both fire, producing two clearly different messages.
    if telegram_ou_alerts:
        print(f"\n🚨 Sending {len(telegram_ou_alerts)} O/U Telegram alert(s) (≥{TELEGRAM_OU_MIN_CONFIDENCE_PCT:.0f}% conf)...")
        for alert in telegram_ou_alerts:
            message = format_ou_alert(alert)
            if send_telegram_alert(message):
                print(f"📤 O/U alert sent for {alert['Game']}")
                time.sleep(1)

    if telegram_ml_alerts:
        print(f"\n🚨 Sending {len(telegram_ml_alerts)} ML Telegram alert(s) (≥{TELEGRAM_ML_MIN_CONFIDENCE_PCT:.0f}% ML conf)...")
        for alert in telegram_ml_alerts:
            message = format_ml_alert(alert)
            if send_telegram_alert(message):
                print(f"📤 ML alert sent for {alert['Game']}")
                time.sleep(1)

    # Log unmatched
    if unmatched_pitchers:
        alias_path = os.path.join(DATA_DIR, "pitcher_aliases.json")
        existing_aliases = {}
        if os.path.exists(alias_path):
            try:
                with open(alias_path, 'r') as f:
                    existing_aliases = json.load(f)
            except:
                existing_aliases = {}

            new_aliases = {
                DataManager.normalize_name(p): [DataManager.normalize_name(p)]
                for p in unmatched_pitchers
                if DataManager.normalize_name(p) not in existing_aliases
                and DataManager.normalize_name(p) not in stats_df.index
            }

        if new_aliases:
            existing_aliases.update(new_aliases)
            with open(alias_path, 'w') as f:
                json.dump(existing_aliases, f, indent=2)

            alert_msg = "🚨 *Over Gang Alert*: Unmatched Pitchers Detected\n\n"
            alert_msg += "\n".join(f"• {p}" for p in unmatched_pitchers)
            alert_msg += "\n\n🔧 Add aliases to `/data/pitcher_aliases.json`"
            send_telegram_alert(alert_msg)
    else:
        print("\nℹ️ No unmatched pitchers detected")

    if alias_log:
        print("\n🔁 Alias matches used:")
        for match in alias_log:
            print(f"• {match}")

    # Data quality summary
    real_totals = sum(1 for r in results if r.get("Total_Is_Real"))
    fallback_totals = sum(1 for r in results if not r.get("Total_Is_Real"))
    projection_only = sum(1 for r in results if r.get("Play_Status") == "PROJECTION_ONLY")
    degraded = sum(1 for r in results if (str(r.get("Data_Quality_Flag") or "").strip() != "" or r.get("No_Fire_Reason") == "data_quality_degraded"))
    print("\n--- DATA QUALITY ---")
    print(f"  Real totals used: {real_totals}")
    print(f"  Fallback totals used: {fallback_totals}")
    print(f"  Projection-only games: {projection_only}")
    print(f"  Degraded-data games: {degraded}")
    print("--------------------")

    # Degraded reasons summary
    dq_flag = lambda r, key: (str(r.get("Data_Quality_Flag") or "") or "").find(key) >= 0
    missing_public = sum(1 for r in results if dq_flag(r, "missing_public_data"))
    low_ip_flags = sum(1 for r in results if dq_flag(r, "low_ip"))
    fallback_pitcher = sum(1 for r in results if dq_flag(r, "fallback_pitcher"))
    print("\n--- DEGRADED REASONS ---")
    print(f"  Missing public betting: {missing_public}")
    print(f"  Low-IP starter flags: {low_ip_flags}")
    print(f"  Fallback pitcher stats used: {fallback_pitcher}")
    print("------------------------")

    # Market path summary (manual vs market: same classifier as VegasLines via Total_Line_Source)
    manual_trusted_n = sum(1 for r in results if _result_row_is_manual_trusted_total(r))
    api_real_n = sum(1 for r in results if r.get("Total_Is_Real") and not _result_row_is_manual_trusted_total(r))
    fallback_n = sum(1 for r in results if not r.get("Total_Is_Real"))
    print("\n--- MARKET PATH ---")
    print(f"  Trusted totals (manual CSV): {manual_trusted_n}")
    print(f"  Trusted totals (API/SDIO, non-manual): {api_real_n}")
    print(f"  No trusted line (fallback): {fallback_n}")
    print("  (Non-manual count excludes manual rows: Total_Line_Source=manual_totals_csv or Odds_Book=Manual.)")
    print("-------------------")

    # LIVE TOTAL BLOCKERS summary
    print("\n--- LIVE TOTAL BLOCKERS ---")
    print(f"  scrambled_book: {live_total_scrambled_book}")
    print(f"  empty_book: {live_total_empty_book}")
    print(f"  unrealistic_total: {live_total_unrealistic_total}")
    print(f"  missing_total_line: {live_total_missing_total_line}")
    print(f"  real_total_pass: {live_total_real_total_pass}")
    print("-----------------------------")

    # LIVE TOTAL BLOCKER GAMES (compact listing)
    def _fmt_keys(keys_list):
        _n = len(keys_list)
        if _n <= 8:
            return ", ".join(keys_list)
        return ", ".join(keys_list[:8]) + f"... (+{_n - 8} more)"

    print("\n--- LIVE TOTAL BLOCKER GAMES ---")
    print(f"  Empty book: {_fmt_keys(live_total_empty_book_keys)}")
    print(f"  Scrambled book: {_fmt_keys(live_total_scrambled_book_keys)}")
    print("-------------------------------")

    # Auto fire status (one line) — aligned with preflight mode, not api_real_n (result rows can
    # diverge from slate-level non-manual coverage when lines are manual CSV / Book=Manual).
    _pf_mode = preflight.get("mode", "projection_only")
    if _pf_mode == "full_auto":
        print("[AUTO FIRE STATUS] LIVE AUTO READY")
    elif _pf_mode == "manual_test":
        print("[AUTO FIRE STATUS] MANUAL TEST ONLY")
    elif _pf_mode == "projection_only":
        print("[AUTO FIRE STATUS] PROJECTION ONLY")
    elif _pf_mode == "stop":
        print("[AUTO FIRE STATUS] STOP")
    else:
        print("[AUTO FIRE STATUS] NOT READY")

    # Full auto check (one line)
    if _pf_mode == "full_auto":
        print("[FULL AUTO CHECK] All requirements satisfied")
    else:
        _fa_missing = []
        _g_n = len(games) if games else 0
        _non_manual_ou_n = _preflight_count_games_with_non_manual_real_totals(games, odds_map)
        if stats_df is None or (isinstance(stats_df, pd.DataFrame) and stats_df.empty):
            _fa_missing.append("pitcher data missing")
        if _g_n == 0:
            _fa_missing.append("no games")
        _bp = getattr(BullpenManager, "BULLPEN_CSV", os.path.join("data", "bullpen_stats.csv"))
        try:
            _bp_ok = os.path.exists(_bp) and os.path.getsize(_bp) > 0
            _bp_n = len(pd.read_csv(_bp)) if _bp_ok else 0
        except Exception:
            _bp_ok, _bp_n = False, 0
        if not _bp_ok or _bp_n == 0:
            _fa_missing.append("bullpen missing")
        if _g_n > 0 and (_non_manual_ou_n == 0 or _non_manual_ou_n < max(1, _g_n - 1)):
            _fa_missing.append("odds coverage incomplete (non-manual market O/U totals)")
        print("[FULL AUTO CHECK]", "; ".join(_fa_missing) if _fa_missing else "see preflight mode")

    # Opening Day readiness end-of-run summary
    def _fmt_keys_preview(keys_list, limit=12):
        _keys = list(keys_list or [])
        _n = len(_keys)
        if _n == 0:
            return ""
        if _n <= limit:
            return ", ".join(_keys)
        return ", ".join(_keys[:limit]) + f"... (+{_n - limit} more)"

    print("\n--- OPENING DAY READINESS ---")
    print(f"  Missing pitchers: {_fmt_keys_preview(sorted(opening_missing_pitchers_by_name))}")
    print(f"  Fallback pitchers: {_fmt_keys_preview(sorted(opening_fallback_pitchers_by_name))}")
    print(f"  Games missing required inputs: {_fmt_keys_preview(sorted(set(opening_games_missing_required_inputs)))}")
    print("----------------------------------")

    # Runtime pitcher resolution outcome totals
    print("\n--- RUNTIME PITCHER RESOLUTION SUMMARY ---")
    print(f"  exact: {rt_exact}")
    print(f"  alias: {rt_alias}")
    print(f"  manual_fallback: {rt_manual_fallback}")
    print(f"  fuzzy: {rt_fuzzy}")
    print(f"  scrape_success: {rt_scrape_success}")
    print(f"  scrape_fail: {rt_scrape_fail}")
    print(f"  league_average: {rt_league_average}")
    print("-------------------------------------------")

    # Final run summary
    _manual = _load_manual_totals()
    print("\n--- RUN SUMMARY ---")
    print(f"  Mode: {preflight.get('mode', 'projection_only')}")
    print(f"  Games processed: {len(results)}")
    print(f"  Games with trusted O/U (Total_Is_Real): {sum(1 for r in results if r.get('Total_Is_Real'))}")
    print(f"  Games with ML_Fired: {sum(1 for r in results if r.get('ML_Fired'))}")
    print(f"  O/U Telegram alerts sent: {len(telegram_ou_alerts)} (candidates: {len(alerts)})")
    print(f"  ML Telegram alerts sent: {len(telegram_ml_alerts)} (candidates: {len(ml_alerts)})")
    print(f"  Manual totals loaded: {len(_manual)}")
    print("-------------------")

# ================================
# 🚀 ENTRY POINT
# ================================
if __name__ == "__main__":
    if os.getenv("OG_TEST_SCRAPER") == "1":
        # Temporary local test: scraper only for Gerrit Cole, Martin Perez.
        # Prints lookup URL(s), HTTP status, response shape, player id found, final stats, exact return.
        for test_name in ["Gerrit Cole", "Martin Perez"]:
            print("\n" + "=" * 60)
            print(f"TEST SCRAPER: {test_name}")
            print("=" * 60)
            out = DataManager.scrape_fangraphs_pitcher(test_name)
            player_id_found = out is not None
            final_stats_returned = out is not None
            print(f"[TEST] Player id found: {player_id_found}")
            print(f"[TEST] Final stats returned: {final_stats_returned}")
            print(f"[TEST] Exact returned object: {out}")
        print("\n" + "=" * 60)
        print("OG_TEST_SCRAPER done.")
        print("=" * 60)
    else:
        run_predictions()
