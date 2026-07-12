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
from core.odds_api import fetch_mlb_odds, fetch_ou_sharp_totals, get_game_odds
from core.the_odds_api import (
    fetch_f5_totals_by_game,
    fetch_full_game_odds_map,
    fetch_ou_totals_by_game,
)
from core.batters import Batters, LineupImpact, BATTER_DF
from core.lineups import fetch_confirmed_lineups
from core.weather_adjustment import (
    WEATHER_RUNS_MULT_MAX,
    WEATHER_RUNS_MULT_MIN,
    compute_weather_runs_mult,
)
from core.starter_fatigue import xera_delta_for_pitcher_days_rest
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


def _export_era_minus_xera(era_val, xera_val):
    """Archive export only: ERA − xERA when both parse as finite floats; else ''."""
    try:
        e = float(era_val)
        x = float(xera_val)
        if not np.isfinite(e) or not np.isfinite(x):
            return ""
        return round(e - x, 4)
    except (TypeError, ValueError):
        return ""


# ================================
# ⚙️ CONFIGURATION
# ================================
TELEGRAM_BOT_TOKEN = '7660295294:AAHakWClywbZP9hdgC5DomgT8EyBa14w-wU'
TELEGRAM_CHAT_ID = '1821580164'
MIN_CONFIDENCE_ALERT = 0.85
# ML side-signal fire: max(home_win_prob, away_win_prob) from calculate_team_win_probability (not gated on O/U totals).
MIN_ML_WIN_PROB_FOR_FIRE = 0.55
# Customer Telegram only (CSV/export unchanged): min confidence to send a message.
TELEGRAM_OU_MIN_CONFIDENCE_PCT = 75.0
TELEGRAM_ML_MIN_CONFIDENCE_PCT = 70.0
# ML calibration: global shrink toward 50%, then hard ceiling on displayed/fired/Kelly probs.
ml_compression_k = 0.75
max_ml_cap = 0.80
# ML fire: both moneylines required (de-vig), no League Avg pitcher shell, penalty must clear floor.
MIN_ML_QUALITY_PENALTY_FOR_FIRE = 0.90
# Picked-side edge (adjusted prob minus de-vig implied); blocks thin ML_Fired when win prob alone clears.
MIN_ML_EDGE_FOR_FIRE = 0.05
# ML sharpness / market-truth parallel fire gate: max allowed absolute gap (fraction, 0.10 = 10pp)
# between Novig-exchange and Pinnacle-sharp de-vigged implied probability on the picked side.
# Fail-open when sharpness inputs are absent/invalid — the gate can only block a fire that
# would otherwise pass; it never changes confidence, Kelly, Telegram send, or O/U behavior.
MAX_EXCHANGE_VS_SHARP_GAP = 0.10
DATA_DIR = "data"


def _american_odds_to_implied(american):
    """Implied win probability from American odds; None if missing or invalid."""
    if american is None:
        return None
    try:
        v = float(american)
    except (TypeError, ValueError):
        return None
    if v == 0:
        return None
    if v < 0:
        return abs(v) / (100.0 + abs(v))
    return 100.0 / (100.0 + v)


def _ml_pair_devig_implied(ml_home_am, ml_away_am) -> tuple:
    """
    Proportional de-vig from both sides' American moneylines (Odds API / book).
    Returns (implied_home, implied_away, status) with status 'ok' | 'missing_market' | 'invalid_odds'.
    """
    ih = _american_odds_to_implied(ml_home_am)
    ia = _american_odds_to_implied(ml_away_am)
    if ih is None or ia is None:
        return None, None, "missing_market"
    s = ih + ia
    if s <= 0:
        return None, None, "invalid_odds"
    return ih / s, ia / s, "ok"


def _clamp_unit_interval(x, lo=0.0, hi=1.0):
    """Clamp x into [lo, hi]; None if not numeric or NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return max(lo, min(hi, v))


ARCHIVE_DIR = "archive"


def archive_output_path(stem: str, archive_date: str) -> str:
    """
    Build archive CSV path.

    Normal runs keep existing canonical names:
      archive/predictions_YYYYMMDD_HHMM.csv

    Targeted refresh runs can set:
      OVERGANG_ARCHIVE_PREFIX=starter_refresh

    which produces:
      archive/starter_refresh_predictions_YYYYMMDD_HHMM.csv

    This prevents partial targeted refresh output from being selected as the
    latest full-slate predictions file by downstream scripts.
    """
    prefix = str(os.environ.get("OVERGANG_ARCHIVE_PREFIX") or "").strip()
    safe_prefix = "".join(
        ch if (ch.isalnum() or ch in "_-") else "_"
        for ch in prefix
    ).strip("_")
    filename = (
        f"{stem}_{archive_date}.csv"
        if not safe_prefix
        else f"{safe_prefix}_{stem}_{archive_date}.csv"
    )
    return f"{ARCHIVE_DIR}/{filename}"


def model_telegram_suppressed() -> bool:
    return str(os.environ.get("OVERGANG_SUPPRESS_MODEL_TELEGRAM") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

STATS_FILE = os.path.join(DATA_DIR, "pitcher_stats.csv")
AUTO_UPDATE_DATA = True

# Dynamic IP thresholds
MIN_PITCHER_IP_EARLY = 10
MIN_PITCHER_IP_MID = 20
MIN_PITCHER_IP_LATE = 15

# Thresholds
NAME_MATCH_THRESHOLD = 85
FATIGUE_THRESHOLD = 4.25
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


_MISSING_PROBABLE_RECOVERY_CACHE = {}


def _recover_missing_probable_pitcher_from_game_feed(game: dict, side: str) -> str:
    """
    Best-effort recovery for blank/TBD probable starters before falling back to
    League Avg Away/Home.

    Uses MLB StatsAPI live game feed only when a probable starter is missing.
    This is not an odds/paid-credit API call.

    Expected side: "away" or "home".
    """
    side = str(side or "").strip().lower()
    if side not in {"away", "home"}:
        return ""

    try:
        game_pk = (
            game.get("game_id")
            or game.get("gamePk")
            or game.get("game_pk")
            or game.get("Game_ID")
        )
    except Exception:
        game_pk = None

    if not game_pk:
        return ""

    cache_key = (str(game_pk), side)
    try:
        if cache_key in _MISSING_PROBABLE_RECOVERY_CACHE:
            return _MISSING_PROBABLE_RECOVERY_CACHE[cache_key]
    except Exception:
        pass

    recovered = ""
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if resp.status_code != 200:
            recovered = ""
        else:
            data = resp.json() or {}

            candidates = []

            # Pregame/daily feed path when MLB publishes probable pitchers.
            try:
                pp = data.get("gameData", {}).get("probablePitchers", {}) or {}
                node = pp.get(side) or {}
                if isinstance(node, dict):
                    candidates.extend([
                        node.get("fullName"),
                        node.get("name"),
                        node.get("boxscoreName"),
                    ])
            except Exception:
                pass

            # Some feeds expose probable info under gameData.teams side nodes.
            try:
                team_node = data.get("gameData", {}).get("teams", {}).get(side, {}) or {}
                node = team_node.get("probablePitcher") or {}
                if isinstance(node, dict):
                    candidates.extend([
                        node.get("fullName"),
                        node.get("name"),
                        node.get("boxscoreName"),
                    ])
                elif isinstance(node, str):
                    candidates.append(node)
            except Exception:
                pass

            # Post-lineup/live fallback: if the first listed pitcher is available
            # pregame/near-start, use its person name. Safe best-effort only.
            try:
                box_team = data.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}) or {}
                pitchers = box_team.get("pitchers") or []
                players = box_team.get("players") or {}
                if pitchers:
                    pid = str(pitchers[0])
                    pnode = players.get(f"ID{pid}") or players.get(pid) or {}
                    person = pnode.get("person") or {}
                    candidates.extend([
                        person.get("fullName"),
                        pnode.get("fullName"),
                    ])
            except Exception:
                pass

            for cand in candidates:
                s = str(cand or "").strip()
                if s and s.upper() != "TBD" and "league avg" not in s.lower():
                    recovered = s
                    break

    except Exception as e:
        print(f"⚠️ Missing probable recovery failed for gamePk={game_pk} side={side}: {e}")
        recovered = ""

    try:
        _MISSING_PROBABLE_RECOVERY_CACHE[cache_key] = recovered
    except Exception:
        pass

    return recovered


def _mlb_targeted_resolve_pitcher_id(probable_name: str):
    """
    Resolve probable display name → (mlb_id, full_name) via MLB StatsAPI people/search?names=...
    Does not use FanGraphs or full-file pitcher refresh.
    """
    if not probable_name or not str(probable_name).strip():
        return None, None
    norm_t = DataManager.normalize_name(probable_name)
    if not norm_t:
        return None, None
    # Two-way players: search hits may list primaryPosition as DH/OF, so no _is_mlb_pitcher_person match.
    if norm_t == "shohei ohtani":
        return 660271, "Shohei Ohtani"
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
STARTER_IP_SHARE = 0.55      # ~55% of IP from starter, 45% bullpen
BULLPEN_IP_SHARE = 0.45
WHIP_LEAGUE = 1.30            # baseline WHIP for modifier
EDGE_THRESHOLD = 0.25        # min |edge| to recommend OVER/UNDER (runs); tune up (e.g. 0.35) if too aggressive
EDGE_FOR_FULL_UNIT = 0.5     # edge >= this gets 1.0 unit; scale below; tune up (e.g. 0.6) for conservative sizing
# O/U confidence (generate_prediction only): multiplicative trims on pitcher primitives — not derived CSV labels.
LEAGUE_AVG_OU_CONFIDENCE_MULT = 0.75  # "League Avg" pitcher name shell
LOW_IP_OU_CONFIDENCE_MULT = 0.78      # severe LowIP tier for very small real starter samples


def _ou_low_ip_confidence_multiplier(starter_ip, is_low_ip) -> float:
    """
    Confidence trim for real matched low-IP starters.

    Do not use this for League Avg fallback shells. League Avg has its own
    fallback penalty and should not stack with the real-starter LowIP penalty.

    Sportsbook-clarity principle:
    3 IP and 45 IP are not the same uncertainty.
    """
    if not bool(is_low_ip):
        return 1.0

    try:
        ip = float(starter_ip)
        if ip != ip:
            ip = 0.0
    except (TypeError, ValueError):
        ip = 0.0

    if ip < 10.0:
        return LOW_IP_OU_CONFIDENCE_MULT
    if ip < 25.0:
        return 0.84
    if ip < 45.0:
        return 0.90
    if ip < 60.0:
        return 0.95
    return 1.0

_OU_WORKLOAD_MAP_CACHE = None


def _ou_safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _ou_workload_shares_from_expected_ip(expected_starter_ip):
    """Return starter/bullpen shares from proven Expected_Starter_IP, else fixed 55/45."""
    ip = _ou_safe_float(expected_starter_ip)
    if ip is None:
        return float(STARTER_IP_SHARE), float(BULLPEN_IP_SHARE), ""

    ip = max(4.0, min(6.5, ip))
    starter_share = max(0.44, min(0.72, ip / 9.0))
    bullpen_share = 1.0 - starter_share
    return float(starter_share), float(bullpen_share), round(float(ip), 3)


def _load_ou_workload_map():
    """Load canonical pitcher workload fields from data/pitcher_k_stats.csv once."""
    global _OU_WORKLOAD_MAP_CACHE
    if _OU_WORKLOAD_MAP_CACHE is not None:
        return _OU_WORKLOAD_MAP_CACHE

    out = {}
    path = os.path.join(DATA_DIR, "pitcher_k_stats.csv")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ O/U workload map unavailable: {e}")
        _OU_WORKLOAD_MAP_CACHE = out
        return out

    required = {"Name", "Expected_Starter_IP", "Starter_Workload_Profile", "Workload_Eligible"}
    if df.empty or not required.issubset(set(df.columns)):
        print("⚠️ O/U workload map unavailable: pitcher_k_stats.csv missing workload columns")
        _OU_WORKLOAD_MAP_CACHE = out
        return out

    for _, row in df.iterrows():
        name = str(row.get("Name") or "").strip()
        key = DataManager.normalize_name(name)
        if not key or "league avg" in key:
            continue

        profile = str(row.get("Starter_Workload_Profile") or "").strip()
        workload_eligible = str(row.get("Workload_Eligible")).strip().lower() in {"true", "1", "yes", "y"}
        expected_ip = _ou_safe_float(row.get("Expected_Starter_IP"))

        if workload_eligible and profile == "clean_starter" and expected_ip is not None:
            starter_share, bullpen_share, expected_ip_used = _ou_workload_shares_from_expected_ip(expected_ip)
            out[key] = {
                "profile": profile,
                "eligible": True,
                "expected_ip": expected_ip_used,
                "starter_share": starter_share,
                "bullpen_share": bullpen_share,
                "source": "expected_starter_ip",
            }
        else:
            out[key] = {
                "profile": profile or "workload_not_available",
                "eligible": False,
                "expected_ip": "",
                "starter_share": float(STARTER_IP_SHARE),
                "bullpen_share": float(BULLPEN_IP_SHARE),
                "source": "fixed_55_45",
            }

    _OU_WORKLOAD_MAP_CACHE = out
    print(f"✅ Loaded O/U workload map: {len(out)} pitcher(s)")
    return out


def _ou_workload_for_pitcher(pitcher_name):
    """Return source-owned O/U workload context; unknown profiles preserve fixed 55/45."""
    name = str(pitcher_name or "").strip()
    if not name or "league avg" in name.lower():
        return {
            "profile": "league_avg_shell" if name else "missing_pitcher_name",
            "eligible": False,
            "expected_ip": "",
            "starter_share": float(STARTER_IP_SHARE),
            "bullpen_share": float(BULLPEN_IP_SHARE),
            "source": "fixed_55_45",
        }

    key = DataManager.normalize_name(name)
    row = _load_ou_workload_map().get(key)
    if isinstance(row, dict):
        return row

    return {
        "profile": "missing_k_stats",
        "eligible": False,
        "expected_ip": "",
        "starter_share": float(STARTER_IP_SHARE),
        "bullpen_share": float(BULLPEN_IP_SHARE),
        "source": "fixed_55_45",
    }
# Unified bullpen workload fatigue (single run multiplier — do not stack with a second IP rule):
# expected_weekly_ip ≈ reliever_count * BULLPEN_EXPECTED_IP_PER_RELIEVER_WEEK
# fatigue_ratio = IP_Week / expected_weekly_ip — symmetric around neutral: tired pen (+runs),
# fresh pen (−runs), with a gentler slope/cap on the fresh side.
# Baseline matches observed IP_Week/Relievers (~2.6–2.7) in data/bullpen_stats.csv so routine usage sits near ratio 1.0.
BULLPEN_EXPECTED_IP_PER_RELIEVER_WEEK = 2.65
BULLPEN_FATIGUE_RATIO_NEUTRAL = 1.0
BULLPEN_FATIGUE_RUNS_PER_EXCESS_RATIO = 0.32
BULLPEN_FATIGUE_RUNS_PER_DEFICIT_RATIO = 0.07
BULLPEN_FATIGUE_RUNS_MULT_MAX = 1.08
BULLPEN_FATIGUE_FRESH_RUNS_MULT_MIN = 0.98


def _bullpen_workload_fatigue_multiplier(fatigue_ratio: float) -> float:
    """
    Maps workload ratio (IP_Week / expected weekly IP) to a runs multiplier for the
    offense facing that bullpen. Used by project_team_runs and mirrored in telemetry.
    """
    try:
        r = float(fatigue_ratio)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(r):
        return 1.0
    deviation = r - BULLPEN_FATIGUE_RATIO_NEUTRAL
    if deviation > 0:
        bump = min(
            BULLPEN_FATIGUE_RUNS_MULT_MAX - 1.0,
            BULLPEN_FATIGUE_RUNS_PER_EXCESS_RATIO * deviation,
        )
        return 1.0 + bump
    if deviation < 0:
        cut = min(
            1.0 - BULLPEN_FATIGUE_FRESH_RUNS_MULT_MIN,
            BULLPEN_FATIGUE_RUNS_PER_DEFICIT_RATIO * (-deviation),
        )
        return 1.0 - cut
    return 1.0


# Velocity (Season_Velo - Recent_Velo from VelocityTracker): positive = lost velo vs season baseline.
# Ignore small scatter; scale conservatively and cap so we do not stack aggressively with starter_fatigue xERA or LowIP.
VELO_DROP_NOISE_MPH = 0.50
VELO_DROP_RUNS_PER_MPH = 0.012
VELO_DROP_RUNS_MULT_MAX = 1.05
VELO_DROP_CONF_LOSS_MPH = 1.05  # meaningful loss: slight confidence trim only (separate from xERA bumps)
LINEUP_IMPACT_CAP = 0.15     # cap (1 + lineup_impact) to ±15%; kept smaller than offense_mult
# Stage C: lineup × bullpen cross-term in project_team_runs only (after lineup_mult).
# Normalized boost/cut × small coef; does not replace lineup_mult or bullpen blend.
LINEUP_BULLPEN_INTERACTION_COEF = 0.035
LINEUP_BULLPEN_INTERACTION_BP_ERA_ARM = 1.5  # ERA pts vs league for full normalized arm
LOW_IP_XERA_PENALTY = 0.75   # add to xERA when pitcher has low IP (unreliable)
OFFENSE_MULT_MIN = 0.90      # clamp offense_mult (team offense strength vs pitcher hand)
OFFENSE_MULT_MAX = 1.10
# Per-team run ceiling after all multipliers: base 6.5, raised modestly in weak-pitching / hitter-friendly contexts.
DYNAMIC_TEAM_RUN_CAP_BASE = 6.5
DYNAMIC_TEAM_RUN_CAP_MAX = 8.5

# Park Factors
PARK_FACTORS = {
    "Coors Field": (1.25, 0.80),
    "Fenway Park": (1.15, 0.90),
    "Globe Life Field": (1.15, 0.90),
    "Oriole Park at Camden Yards": (1.10, 0.92),
    "Great American Ball Park": (1.12, 0.91),
    "Wrigley Field": (1.08, 0.95),
    "Petco Park": (0.92, 1.10),
    "Oracle Park": (0.85, 1.15),
    "Citi Field": (0.90, 1.12),
    "UNIQLO Field at Dodger Stadium": (0.95, 1.08),
    "T-Mobile Park": (0.88, 1.14),
    "American Family Field": (0.99, 1.01),
    "Angel Stadium": (1.01, 0.99),
    "Busch Stadium": (0.98, 1.02),
    "Chase Field": (1.01, 0.99),
    "Citizens Bank Park": (1.01, 0.99),
    "Comerica Park": (1.00, 1.00),
    "Kauffman Stadium": (1.03, 0.97),
    "Nationals Park": (1.00, 1.00),
    "PNC Park": (1.02, 0.98),
    "Progressive Field": (0.99, 1.01),
    "Rate Field": (1.00, 1.00),
    "Rogers Centre": (0.99, 1.01),
    "Sutter Health Park": (1.03, 0.97),
    "Target Field": (1.01, 0.99),
    "Tropicana Field": (1.01, 0.99),
    "Truist Park": (1.00, 1.00),
    "Yankee Stadium": (0.99, 1.01),
    "loanDepot park": (1.01, 0.99),
    "Unknown": (1.0, 1.0)
}

# MLB Stats API venue_name strings; outdoor weather may not apply when roof is closed (no roof-state source yet).
RETRACTABLE_ROOF_VENUES = frozenset({
    "American Family Field",
    "Chase Field",
    "Globe Life Field",
    "loanDepot park",
    "Rogers Centre",
    "T-Mobile Park",
})


def _opponent_velocity_run_multiplier(opponent_velo_drop: float) -> float:
    """
    opponent_velo_drop = Season - Recent (mph). Positive => recent fastball slower than season (fatigue signal).
    Below noise => 1.0. Otherwise bounded increase in expected runs vs that starter (capped).
    """
    try:
        d = float(opponent_velo_drop)
    except (TypeError, ValueError):
        return 1.0
    if d <= VELO_DROP_NOISE_MPH:
        return 1.0
    excess = d - VELO_DROP_NOISE_MPH
    bump = min(VELO_DROP_RUNS_MULT_MAX - 1.0, VELO_DROP_RUNS_PER_MPH * excess)
    return min(VELO_DROP_RUNS_MULT_MAX, 1.0 + bump)


def _confidence_trim_for_velocity_loss(velo_drop: float) -> float:
    """Tiny multiplicative trim when this starter has clear velocity loss (kept mild vs rest/LowIP)."""
    try:
        d = float(velo_drop)
    except (TypeError, ValueError):
        return 1.0
    if d < VELO_DROP_CONF_LOSS_MPH:
        return 1.0
    return 0.99


def _dynamic_team_run_cap(
    opponent_starter_xera: float,
    opponent_bullpen_era: float,
    park_runs_factor: float,
    opponent_low_ip: bool,
    opponent_starter_whip: float,
    lineup_impact: float,
) -> float:
    """
    Context-aware per-team run ceiling (not a run multiplier). Base matches legacy 6.5; increases only when
    opponent pitching context or park favors scoring; bounded by DYNAMIC_TEAM_RUN_CAP_MAX.
    lineup_impact is accepted for API stability; kept neutral here (lineup already in runs path).
    """
    _ = lineup_impact
    cap = float(DYNAMIC_TEAM_RUN_CAP_BASE)
    try:
        x = float(opponent_starter_xera)
    except (TypeError, ValueError):
        x = LEAGUE_ERA
    if x > 4.5:
        cap += min(0.58, (x - 4.5) * 0.19)
    try:
        bp = float(opponent_bullpen_era)
    except (TypeError, ValueError):
        bp = LEAGUE_ERA
    if bp > 4.5:
        cap += min(0.45, (bp - 4.5) * 0.14)
    try:
        pk = float(park_runs_factor)
    except (TypeError, ValueError):
        pk = 1.0
    if pk > 1.0:
        cap += min(0.48, (pk - 1.0) * 0.98)
    try:
        w = float(opponent_starter_whip)
    except (TypeError, ValueError):
        w = WHIP_LEAGUE
    if w > WHIP_LEAGUE:
        cap += min(0.15, (w - WHIP_LEAGUE) * 0.55)
    if opponent_low_ip:
        cap += 0.08
    return max(DYNAMIC_TEAM_RUN_CAP_BASE, min(float(DYNAMIC_TEAM_RUN_CAP_MAX), cap))


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
    opponent_bullpen_xera: float = None,  # None -> falls back to ERA-only (current behavior)
    opponent_late_run_pressure_score: float = 0.0,
    opponent_late_run_suppression_score: float = 0.0,
    opponent_bullpen_clarity_score: float = 0.0,
    opponent_reliever_depth_risk: str = "",
    opponent_bullpen_quality_tier: str = "",
    weather_runs_mult: float = 1.0,
    opponent_expected_starter_ip: float = None,  # None -> preserves fixed 55/45 behavior
) -> tuple:
    """
    Project expected runs scored by one team in the game (they face opponent starter + bullpen).

    offense_mult: team offense strength vs opposing pitcher hand (Batters.offense_vs_hand_dict "mult");
      clamped to OFFENSE_MULT_MIN..OFFENSE_MULT_MAX. 1.0 when batter data unavailable.
    lineup_impact: smaller adjustment from LineupImpact.score_lineup (capped by LINEUP_IMPACT_CAP).
    opponent_bullpen_relievers: active reliever count for expected weekly IP baseline (workload fatigue ratio).

    Returns (analytical_runs, safety_capped_runs): analytical_runs is the uncapped model output;
    safety_capped_runs is min(runs, dynamic cap) for risk control only (not the O/U analytical path).
    """
    if opponent_low_ip:
        opponent_starter_xera = min(opponent_starter_xera + LOW_IP_XERA_PENALTY, 6.0)

    # Stage A: bullpen-quality input for the O/U projection blends raw ERA
    # with xERA (75/25) when xERA is available and finite. xERA remains in the
    # path for signal; ERA is weighted higher to avoid over-cold bullpen xERA
    # vs realized runs when ERA−xERA gaps are large and positive.
    # Safe fallback: if xERA is missing or non-finite, use raw ERA so the
    # projection degrades exactly to current behavior.
    _bp_era = opponent_bullpen_era
    _bp_xera = opponent_bullpen_xera
    if _bp_xera is None or not np.isfinite(_bp_xera):
        bullpen_quality = _bp_era
    else:
        bullpen_quality = 0.75 * _bp_era + 0.25 * _bp_xera

    starter_ip_share, bullpen_ip_share, _expected_ip_used = _ou_workload_shares_from_expected_ip(
        opponent_expected_starter_ip
    )

    def _ptr_float(v, default=0.0):
        try:
            f = float(v)
            return f if np.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    # Whole-picture run pressure belongs inside the run projection, not only as a
    # tiny post-total overlay. Early pressure reprices the starter component;
    # late pressure reprices only the bullpen component. This keeps the model
    # baseball-structured instead of forcing blanket OVER flips.
    # Early repricing is owned only by starter xERA. WHIP, offense,
    # park, and weather are each applied once later in the canonical stack.
    _starter_xera_for_pressure = _ptr_float(
        opponent_starter_xera,
        LEAGUE_ERA,
    )

    _early_mult = 1.0
    if _starter_xera_for_pressure >= 5.00:
        _early_mult += 0.050
    elif _starter_xera_for_pressure >= 4.40:
        _early_mult += 0.030

    _early_mult = max(0.96, min(1.12, _early_mult))
    adjusted_starter_xera = opponent_starter_xera * _early_mult

    _late_pressure = max(0.0, min(1.25, _ptr_float(opponent_late_run_pressure_score, 0.0)))
    _late_suppression = max(0.0, min(1.25, _ptr_float(opponent_late_run_suppression_score, 0.0)))
    _late_net = max(0.0, _late_pressure - _late_suppression)
    _late_supp_net = max(0.0, _late_suppression - _late_pressure)
    _clarity = _ptr_float(opponent_bullpen_clarity_score, 0.0)
    _depth = str(opponent_reliever_depth_risk or "").strip().lower()
    _tier = str(opponent_bullpen_quality_tier or "").strip().lower()

    _late_mult = 1.0
    _late_mult += min(0.100, _late_net * 0.075)
    _late_mult -= min(0.070, _late_supp_net * 0.050)

    if _depth == "high":
        _late_mult += 0.045
    elif _depth == "medium":
        _late_mult += 0.018
    elif _depth == "low":
        _late_mult -= 0.012

    if _tier == "danger":
        _late_mult += 0.040
    elif _tier == "risky":
        _late_mult += 0.030
    elif _tier in {"strong", "elite"}:
        _late_mult -= 0.020

    if _clarity <= -15.0:
        _late_mult += 0.050
    elif _clarity <= -8.0:
        _late_mult += 0.035
    elif _clarity <= -5.0:
        _late_mult += 0.020
    elif _clarity >= 10.0:
        _late_mult -= 0.030
    elif _clarity >= 5.0:
        _late_mult -= 0.018

    # Normal late-pressure games stay tightly bounded. Truly extreme late-run
    # profiles get a wider ceiling because the prior 1.22 cap still underpriced
    # High/High + late_over_path bullpen-chaos games.
    _late_cap = 1.22

    # Preserve the existing extreme-profile eligibility gate without
    # adding offense back into the late-pressure multiplier itself.
    _offense_for_extreme_gate = max(
        OFFENSE_MULT_MIN,
        min(
            OFFENSE_MULT_MAX,
            _ptr_float(offense_mult, 1.0),
        ),
    )
    _extreme_late_profile = (
        _late_pressure >= 1.15
        and _late_net >= 0.90
        and _depth == "high"
        and _clarity <= -5.0
        and _tier in {"risky", "danger"}
        and _offense_for_extreme_gate >= 0.98
    )
    if _extreme_late_profile:
        _late_cap = 1.34
        _late_mult += 0.080

    _late_mult = max(0.94, min(_late_cap, _late_mult))
    adjusted_bullpen_quality = bullpen_quality * _late_mult

    effective_era = (
        starter_ip_share * adjusted_starter_xera
        + bullpen_ip_share * adjusted_bullpen_quality
    )
    effective_era = max(2.5, min(7.0, effective_era))
    base_era = 0.85 * effective_era + 0.15 * LEAGUE_ERA
    runs = LEAGUE_RUNS_PER_TEAM * (base_era / LEAGUE_ERA)

    runs *= park_runs_factor

    offense_mult = max(OFFENSE_MULT_MIN, min(OFFENSE_MULT_MAX, float(offense_mult)))
    runs *= offense_mult

    lineup_mult = 1.0 + max(-LINEUP_IMPACT_CAP, min(LINEUP_IMPACT_CAP, lineup_impact))
    runs *= lineup_mult

    # Stage C: lineup × bullpen interaction (O/U projection). Bounded asymmetric
    # cross-term: strong lineup + weak bullpen (ERA/xERA blend above league) →
    # modest +runs; weak lineup + strong bullpen → modest −runs. Uses the same
    # clamped lineup signal as lineup_mult and bullpen_quality from Stage A.
    # Telemetry hook: lineup_bullpen_interaction_mult (and boost/cut parts).
    try:
        _lu_c = max(-LINEUP_IMPACT_CAP, min(LINEUP_IMPACT_CAP, float(lineup_impact)))
        _bq = float(bullpen_quality)
    except (TypeError, ValueError):
        _lu_c = 0.0
        _bq = float(LEAGUE_ERA)
    _bp_weak_arm = max(0.0, _bq - LEAGUE_ERA)
    _bp_strong_arm = max(0.0, LEAGUE_ERA - _bq)
    _norm_weak = min(1.0, _bp_weak_arm / LINEUP_BULLPEN_INTERACTION_BP_ERA_ARM)
    _norm_strong = min(1.0, _bp_strong_arm / LINEUP_BULLPEN_INTERACTION_BP_ERA_ARM)
    _lu_pos = max(0.0, _lu_c) / LINEUP_IMPACT_CAP
    _lu_neg = max(0.0, -_lu_c) / LINEUP_IMPACT_CAP
    _lineup_bullpen_inter_boost = _lu_pos * _norm_weak
    _lineup_bullpen_inter_cut = _lu_neg * _norm_strong
    lineup_bullpen_interaction_mult = 1.0 + LINEUP_BULLPEN_INTERACTION_COEF * (
        _lineup_bullpen_inter_boost - _lineup_bullpen_inter_cut
    )
    runs *= lineup_bullpen_interaction_mult

    whip_mult = opponent_starter_whip / WHIP_LEAGUE
    whip_mult = max(0.92, min(1.15, whip_mult))
    runs *= whip_mult

    runs *= _opponent_velocity_run_multiplier(opponent_velo_drop)

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
            fatigue_mult = _bullpen_workload_fatigue_multiplier(fatigue_ratio)
            runs *= fatigue_mult

    raw_runs = runs
    _cap = _dynamic_team_run_cap(
        opponent_starter_xera,
        opponent_bullpen_era,
        park_runs_factor,
        opponent_low_ip,
        opponent_starter_whip,
        lineup_impact,
    )
    capped_runs = min(runs, _cap)
    return round(raw_runs, 2), round(capped_runs, 2)


def project_team_f5_runs(
    opponent_starter_xera: float,
    opponent_starter_whip: float,
    park_runs_factor: float,
    lineup_impact: float,
    opponent_velo_drop: float,
    opponent_low_ip: bool = False,
    offense_mult: float = 1.0,
) -> float:
    """
    Project expected runs scored by one team in the FIRST 5 INNINGS only.
    F5 telemetry helper. Starter-driven. Drops bullpen blend, bullpen workload
    fatigue, lineup x bullpen interaction, and dynamic safety cap.
    Used ONLY for archive export telemetry. Never called from any fire, edge,
    Kelly, confidence, or Telegram path.
    """
    if opponent_low_ip:
        opponent_starter_xera = min(opponent_starter_xera + LOW_IP_XERA_PENALTY, 6.0)

    f5_effective_era = max(2.5, min(7.0, opponent_starter_xera))
    f5_base_era = 0.85 * f5_effective_era + 0.15 * LEAGUE_ERA
    f5_runs = LEAGUE_RUNS_PER_TEAM * (f5_base_era / LEAGUE_ERA) * (5.0 / 9.0)

    f5_runs *= park_runs_factor

    offense_mult = max(OFFENSE_MULT_MIN, min(OFFENSE_MULT_MAX, float(offense_mult)))
    f5_runs *= offense_mult

    lineup_mult = 1.0 + max(-LINEUP_IMPACT_CAP, min(LINEUP_IMPACT_CAP, lineup_impact))
    f5_runs *= lineup_mult

    whip_mult = max(0.92, min(1.15, opponent_starter_whip / WHIP_LEAGUE))
    f5_runs *= whip_mult

    f5_runs *= _opponent_velocity_run_multiplier(opponent_velo_drop)

    return round(f5_runs, 2)


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
        """Return dict with ERA, xERA, IP_Week, Relievers for a team; league-average fallback.

        xERA is read from the same bullpen CSV row that ERA already comes from
        (column already present; previously unread on the O/U path). When the
        CSV lacks a usable xERA value, xERA falls back to ERA so the blended
        bullpen-quality input in project_team_runs degrades to the current
        ERA-only behavior (safe no-op fallback).
        """
        # Fix short/common names to MLB API full names
        team_fixed = BullpenManager.TEAM_NAME_FIXES.get(team, team)

        df = BullpenManager._safe_read_csv(BullpenManager.BULLPEN_CSV)
        if df.empty:
            return {'ERA': 4.25, 'xERA': 4.25, 'IP_Week': 12.0, 'Relievers': 7, 'source': 'League Avg'}

        # Normalize
        if "Team" not in df.columns:
            # unexpected schema — fail safe
            return {'ERA': 4.25, 'xERA': 4.25, 'IP_Week': 12.0, 'Relievers': 7, 'source': 'League Avg'}

        df["Team_norm"] = df["Team"].astype(str).str.strip().str.lower()
        lookup = team_fixed.strip().lower()

        # Try exact
        hit = df[df["Team_norm"] == lookup]
        if hit.empty:
            # try original (in case caller already had the full name)
            hit = df[df["Team_norm"] == team.strip().lower()]

        if hit.empty:
            print(f"⚠️ Team not found in bullpen file: {team} (resolved: {team_fixed})")
            return {'ERA': 4.25, 'xERA': 4.25, 'IP_Week': 12.0, 'Relievers': 7, 'source': 'League Avg'}

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
        xera = _finite_float(row.get("xERA"), era)
        ipw = _finite_float(row.get("IP_Week"), 12.0)
        rel = int(_finite_float(row.get("Relievers"), 7.0))

        return {
            "ERA": era,
            "xERA": xera,
            "IP_Week": ipw,
            "Relievers": rel,
            "source": "MLB Stats API",
        }


# -----------------------------
# Reliever-depth telemetry (data/reliever_stats.csv; export-only)
# -----------------------------
RELIEVER_STATS_CSV = os.path.join("data", "reliever_stats.csv")


def load_reliever_stats(path: str | None = None) -> pd.DataFrame:
    """Load reliever_stats.csv; empty DataFrame if missing or unreadable."""
    p = path or RELIEVER_STATS_CSV
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def normalize_team_name_for_reliever_csv(team_name: str) -> str:
    """
    Resolve schedule/short team labels to the same convention as bullpen_stats /
    reliever_stats Team column where possible (fixes + strip).
    """
    if not isinstance(team_name, str):
        s = str(team_name or "").strip()
    else:
        s = team_name.strip()
    if not s:
        return ""
    return BullpenManager.TEAM_NAME_FIXES.get(s, s).strip()


def get_reliever_depth_metrics(team_name: str, reliever_df: pd.DataFrame) -> dict:
    """
    Per-team reliever-depth counts from data/reliever_stats.csv (export telemetry only).
    """
    out = {
        "bad_xera_count": 0,
        "bad_whip_count": 0,
        "recent_bad_arm_count": 0,
        "depth_risk": "low",
        "reliever_metrics_source": "neutral",

        # Bullpen clarity v1: grade actual arm quality + availability.
        "reliever_total_count": 0,
        "shutdown_arm_count": 0,
        "good_arm_count": 0,
        "average_arm_count": 0,
        "risky_arm_count": 0,
        "disaster_arm_count": 0,
        "recent_used_count": 0,
        "taxed_arm_count": 0,
        "heavy_week_count": 0,
        "good_available_count": 0,
        "bad_recent_count": 0,
        "disaster_recent_count": 0,
        "bullpen_quality_tier": "unknown",
        "bullpen_availability_tier": "unknown",
        "bullpen_clarity_score": 0.0,
        "late_run_pressure_score": 0.0,
        "late_run_suppression_score": 0.0,
    }
    if reliever_df is None or not isinstance(reliever_df, pd.DataFrame):
        out["depth_risk"] = "unknown"
        out["reliever_metrics_source"] = "missing_or_empty"
        return out
    if reliever_df.empty or "Team" not in reliever_df.columns:
        out["depth_risk"] = "unknown"
        out["reliever_metrics_source"] = "missing_or_empty"
        return out

    team_fixed = normalize_team_name_for_reliever_csv(team_name)
    df = reliever_df.copy()
    df["_team_norm"] = df["Team"].astype(str).str.strip().str.lower()
    lookup = team_fixed.strip().lower()
    sub = df[df["_team_norm"] == lookup] if lookup else df.iloc[0:0]
    if sub.empty and isinstance(team_name, str):
        alt = team_name.strip().lower()
        if alt:
            sub = df[df["_team_norm"] == alt]
    if sub.empty:
        canon_key = normalize_team_name(team_name).strip().lower() if team_name else ""
        if canon_key:
            sub = df[df["_team_norm"] == canon_key]
    if sub.empty:
        out["depth_risk"] = "unknown"
        out["reliever_metrics_source"] = "team_not_found"
        return out

    out["reliever_metrics_source"] = "csv"

    def _as_float(v):
        try:
            x = float(v)
            if not np.isfinite(x):
                return None
            return x
        except (TypeError, ValueError):
            return None

    def _as_int_nonneg(v):
        try:
            x = float(v)
            if not np.isfinite(x):
                return 0
            return int(max(0, round(x)))
        except (TypeError, ValueError):
            return 0

    def _as_timestamp(v):
        try:
            ts = pd.to_datetime(v, errors="coerce")
            if pd.isna(ts):
                return None
            return ts
        except Exception:
            return None

    # Availability should be judged against the freshest reliever activity in
    # the file, not the wall-clock date. This keeps zero-API validation stable
    # and avoids penalizing off-days, while preventing April/May stale arms from
    # being counted as today's available bullpen support.
    ref_last_game_date = None
    if "Last_Game_Date" in df.columns:
        _all_dates = pd.to_datetime(df["Last_Game_Date"], errors="coerce")
        if _all_dates.notna().any():
            ref_last_game_date = _all_dates.max()

    active_recent_window_days = 21

    bad_xera = bad_whip = recent_bad = 0
    active_bad_xera = active_bad_whip = 0

    total = shutdown = good = average = risky = disaster = 0
    active_shutdown = active_good = active_average = active_risky = active_disaster = 0
    recent_used = taxed = heavy_week = 0
    good_available = bad_recent = disaster_recent = 0

    for _, row in sub.iterrows():
        xera = _as_float(row.get("xERA"))
        whip = _as_float(row.get("WHIP"))
        ip_week = _as_float(row.get("IP_Week"))
        a3 = _as_int_nonneg(row.get("Appearances_3D"))
        last_game_date = _as_timestamp(row.get("Last_Game_Date"))

        total += 1

        if ref_last_game_date is None or last_game_date is None:
            active_recent = True
        else:
            active_recent = (
                0 <= int((ref_last_game_date - last_game_date).days) <= active_recent_window_days
            )

        if xera is not None and xera >= 4.50:
            bad_xera += 1
            if active_recent:
                active_bad_xera += 1
        if whip is not None and whip >= 1.50:
            bad_whip += 1
            if active_recent:
                active_bad_whip += 1

        is_recent = a3 > 0
        is_taxed = a3 >= 2 or (ip_week is not None and ip_week >= 3.0)
        is_heavy_week = ip_week is not None and ip_week >= 4.0

        if active_recent and is_recent:
            recent_used += 1
        if active_recent and is_taxed:
            taxed += 1
        if active_recent and is_heavy_week:
            heavy_week += 1

        # Arm grade v1.
        # Use xERA + WHIP together when available. Unknown values do not force
        # a bad grade; they fall to average/risky only when one known metric is poor.
        if (
            xera is not None and whip is not None
            and xera <= 3.20 and whip <= 1.15
        ):
            grade = "shutdown"
            shutdown += 1
        elif (
            xera is not None and whip is not None
            and xera <= 3.80 and whip <= 1.25
        ):
            grade = "good"
            good += 1
        elif (
            (xera is not None and xera >= 5.50)
            or (whip is not None and whip >= 1.70)
        ):
            grade = "disaster"
            disaster += 1
        elif (
            (xera is not None and xera >= 4.50)
            or (whip is not None and whip >= 1.50)
        ):
            grade = "risky"
            risky += 1
        elif (
            (xera is not None and xera <= 4.50)
            and (whip is None or whip <= 1.40)
        ):
            grade = "average"
            average += 1
        else:
            grade = "average"
            average += 1

        if active_recent:
            if grade == "shutdown":
                active_shutdown += 1
            elif grade == "good":
                active_good += 1
            elif grade == "average":
                active_average += 1
            elif grade == "risky":
                active_risky += 1
            elif grade == "disaster":
                active_disaster += 1

        bad_enough = grade in {"risky", "disaster"}
        good_enough = grade in {"shutdown", "good"}

        if active_recent and is_recent and bad_enough:
            recent_bad += 1
            bad_recent += 1
        if active_recent and is_recent and grade == "disaster":
            disaster_recent += 1

        # Good available means a quality arm exists, is recently active, and
        # is not clearly taxed. Stale April/May arms do not count as available.
        if active_recent and good_enough and not is_taxed and not is_heavy_week:
            good_available += 1

    out["bad_xera_count"] = int(bad_xera)
    out["bad_whip_count"] = int(bad_whip)
    out["recent_bad_arm_count"] = int(recent_bad)

    out["reliever_total_count"] = int(total)
    out["shutdown_arm_count"] = int(shutdown)
    out["good_arm_count"] = int(good)
    out["average_arm_count"] = int(average)
    out["risky_arm_count"] = int(risky)
    out["disaster_arm_count"] = int(disaster)
    out["recent_used_count"] = int(recent_used)
    out["taxed_arm_count"] = int(taxed)
    out["heavy_week_count"] = int(heavy_week)
    out["good_available_count"] = int(good_available)
    out["bad_recent_count"] = int(bad_recent)
    out["disaster_recent_count"] = int(disaster_recent)

    if active_bad_xera >= 4 or active_bad_whip >= 4 or recent_bad >= 4 or active_disaster >= 3:
        out["depth_risk"] = "high"
    elif (
        active_bad_xera >= 2
        or active_bad_whip >= 2
        or recent_bad >= 2
        or active_disaster >= 1
        or active_risky >= 4
    ):
        out["depth_risk"] = "medium"
    else:
        out["depth_risk"] = "low"

    quality_score = (
        2.0 * active_shutdown
        + 1.25 * active_good
        + 0.25 * active_average
        - 1.0 * active_risky
        - 2.0 * active_disaster
    )

    # Availability score penalizes taxed/heavy usage and rewards fresh quality.
    availability_score = (
        1.25 * good_available
        - 0.75 * taxed
        - 1.00 * heavy_week
        - 0.75 * bad_recent
        - 1.25 * disaster_recent
    )

    if active_shutdown + active_good >= 4 and active_disaster == 0 and active_risky <= 2:
        quality_tier = "strong"
    elif active_shutdown + active_good >= 3 and active_disaster <= 1 and active_risky <= 3:
        quality_tier = "solid"
    elif active_disaster >= 3 or active_risky >= 6:
        quality_tier = "danger"
    elif active_disaster >= 1 or active_risky >= 4:
        quality_tier = "risky"
    else:
        quality_tier = "average"

    if good_available >= 3 and taxed <= 1 and heavy_week == 0:
        availability_tier = "fresh"
    elif good_available >= 2 and taxed <= 3:
        availability_tier = "usable"
    elif taxed >= 5 or heavy_week >= 3 or good_available <= 1:
        availability_tier = "thin"
    else:
        availability_tier = "strained"

    # Scores are intentionally bounded because they will be consumed by lane
    # separation later. These are not direct run projections yet.
    late_pressure = (
        0.12 * active_risky
        + 0.22 * active_disaster
        + 0.08 * bad_recent
        + 0.14 * disaster_recent
        + 0.05 * taxed
        + 0.05 * heavy_week
        - 0.05 * good_available
    )
    late_suppression = (
        0.12 * active_shutdown
        + 0.08 * good_available
        + 0.04 * active_good
        - 0.05 * bad_recent
        - 0.08 * active_disaster
    )

    late_pressure = max(0.0, min(1.25, late_pressure))
    late_suppression = max(0.0, min(1.25, late_suppression))

    out["bullpen_quality_tier"] = quality_tier
    out["bullpen_availability_tier"] = availability_tier
    out["bullpen_clarity_score"] = round(float(quality_score + availability_score), 3)
    out["late_run_pressure_score"] = round(float(late_pressure), 3)
    out["late_run_suppression_score"] = round(float(late_suppression), 3)

    return out


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
    def get_vegas_line(home_team, away_team, odds_map=None, game_datetime=None, *, emit_live_total_diagnostics=True):
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
            row = get_game_odds(away_team, home_team, odds_map, commence_time=game_datetime)
            match_found = bool(row.get("_match_found", False))
            raw_line = row.get("_raw_total_line")
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
                and (not book_empty)
                and (not book_scrambled)
                and line_realistic
            )
            is_fallback_line = not has_real_total
            if match_found:
                source = (
                    "fallback (scrambled book)"
                    if book_scrambled
                    else ("8.5 fallback" if is_fallback_line else (row.get("_source") or "parlay_api"))
                )
            else:
                source = "8.5 fallback (no match in odds_map)"
            info["_source"] = source
            info["_lookup_key"] = row.get("_lookup_key", lookup_key)
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
                                _raw_detail = odds_map.get(info.get("_lookup_key", lookup_key)) or {}
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
def _calibrate_ou_edge(raw_edge):
    """
    Calibrate final O/U edge toward the market when raw model edge is large.

    This preserves the model's raw baseball projection separately, while using
    calibrated edge for O/U decision projection, confidence, fire logic, and
    downstream profile boards.

    Based on 2026 archive residual audit:
      - tail edges improved MAE after shrinkage
      - small/sweet-spot edges should be protected
    """
    try:
        edge_f = float(raw_edge)
    except (TypeError, ValueError):
        return 0.0, 1.0, "uncalibrated_invalid_edge"

    abs_edge = abs(edge_f)

    if abs_edge >= 2.0:
        factor = 0.45
        tag = "calibrated_tail_ge_2_00"
    elif abs_edge >= 1.5:
        factor = 0.60
        tag = "calibrated_tail_1_50_to_2_00"
    elif abs_edge >= 1.0:
        factor = 0.75
        tag = "calibrated_mid_1_00_to_1_50"
    elif abs_edge >= 0.75:
        factor = 0.90
        tag = "calibrated_sweet_0_75_to_1_00"
    else:
        factor = 1.00
        tag = "uncalibrated_small_edge"

    calibrated_edge = round(edge_f * factor, 2)
    return calibrated_edge, factor, tag


def _fp_safe_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        f = float(value)
        if f != f:
            return default
        return f
    except Exception:
        return default


def _fp_get(obj, key, default=0.0):
    try:
        if obj is None:
            return default
        if hasattr(obj, "get"):
            return _fp_safe_float(obj.get(key, default), default)
        return _fp_safe_float(obj[key], default)
    except Exception:
        return default


def _fp_metric(metrics, keys, default=0.0):
    try:
        if metrics is None:
            return default
        for key in keys:
            if hasattr(metrics, "get") and metrics.get(key, None) not in ("", None):
                return _fp_safe_float(metrics.get(key), default)
        return default
    except Exception:
        return default


def _fp_depth_level(value):
    s = str(value or "").strip().lower()
    if s in {"high", "true", "1", "yes"} or "high" in s:
        return "high"
    if s in {"medium", "med"} or "medium" in s:
        return "medium"
    if s in {"low"} or "low" in s:
        return "low"
    return ""


def _fp_offense_component(offense_mult, lineup_impact):
    pressure = 0.0
    suppression = 0.0
    reasons = []

    off = _fp_safe_float(offense_mult, 1.0)
    li = _fp_safe_float(lineup_impact, 0.0)

    if off >= 1.08:
        pressure += 0.18
        reasons.append("elite_offense_mult_1.08+")
    elif off >= 1.04:
        pressure += 0.12
        reasons.append("strong_offense_mult_1.04+")
    elif off >= 1.02:
        pressure += 0.06
        reasons.append("mild_offense_mult_1.02+")

    if off <= 0.92:
        suppression += 0.18
        reasons.append("very_weak_offense_mult_0.92-")
    elif off <= 0.96:
        suppression += 0.12
        reasons.append("weak_offense_mult_0.96-")
    elif off <= 0.98:
        suppression += 0.06
        reasons.append("mild_weak_offense_mult_0.98-")

    if li >= 0.08:
        pressure += 0.10
        reasons.append("strong_lineup_boost_0.08+")
    elif li >= 0.04:
        pressure += 0.06
        reasons.append("lineup_boost_0.04+")
    elif li >= 0.02:
        pressure += 0.03
        reasons.append("mild_lineup_boost_0.02+")

    if li <= -0.08:
        suppression += 0.10
        reasons.append("strong_lineup_suppression_-0.08")
    elif li <= -0.04:
        suppression += 0.06
        reasons.append("lineup_suppression_-0.04")
    elif li <= -0.02:
        suppression += 0.03
        reasons.append("mild_lineup_suppression_-0.02")

    return pressure, suppression, reasons


def _fp_starter_prevention_component(opp_stats):
    weak = 0.0
    strong = 0.0
    reasons = []

    xera = _fp_get(opp_stats, "xERA", 4.20)
    whip = _fp_get(opp_stats, "WHIP", 1.30)
    era = _fp_get(opp_stats, "ERA", xera)
    gap = era - xera

    if xera >= 5.50:
        weak += 0.24
        reasons.append("opp_starter_xera_5.50+")
    elif xera >= 5.00:
        weak += 0.18
        reasons.append("opp_starter_xera_5.00+")
    elif xera >= 4.60:
        weak += 0.10
        reasons.append("opp_starter_xera_4.60+")

    if xera <= 3.10:
        strong += 0.14
        reasons.append("opp_starter_xera_3.10-")
    elif xera <= 3.50:
        strong += 0.09
        reasons.append("opp_starter_xera_3.50-")
    elif xera <= 3.80:
        strong += 0.05
        reasons.append("opp_starter_xera_3.80-")

    if whip >= 1.50:
        weak += 0.16
        reasons.append("opp_starter_whip_1.50+")
    elif whip >= 1.38:
        weak += 0.09
        reasons.append("opp_starter_whip_1.38+")

    if whip <= 1.00:
        strong += 0.10
        reasons.append("opp_starter_whip_1.00-")
    elif whip <= 1.12:
        strong += 0.06
        reasons.append("opp_starter_whip_1.12-")

    # Negative ERA-xERA gap means ERA looks better than xERA: regression/run risk.
    if gap <= -1.00:
        weak += 0.14
        reasons.append("opp_starter_lucky_era_gap_-1.00")
    elif gap <= -0.50:
        weak += 0.08
        reasons.append("opp_starter_lucky_era_gap_-0.50")

    if gap >= 1.00:
        strong += 0.07
        reasons.append("opp_starter_unlucky_era_gap_1.00+")
    elif gap >= 0.50:
        strong += 0.04
        reasons.append("opp_starter_unlucky_era_gap_0.50+")

    return weak, strong, reasons


def _fp_bullpen_prevention_component(metrics):
    weak = 0.0
    strong = 0.0
    reasons = []

    bad_xera = _fp_metric(metrics, ["bad_xera_count", "Bad_xERA_Count", "bad_xera"], 0.0)
    bad_whip = _fp_metric(metrics, ["bad_whip_count", "Bad_WHIP_Count", "bad_whip"], 0.0)
    recent_bad = _fp_metric(metrics, ["recent_bad_arm_count", "Recent_Bad_Arm_Count", "recent_bad"], 0.0)
    total_bad = bad_xera + bad_whip + recent_bad

    depth_raw = ""
    try:
        if metrics is not None and hasattr(metrics, "get"):
            depth_raw = metrics.get("depth_risk", metrics.get("Reliever_Depth_Risk", ""))
    except Exception:
        depth_raw = ""
    depth = _fp_depth_level(depth_raw)

    if total_bad >= 8:
        weak += 0.34
        reasons.append("pen_bad_arms_8+")
    elif total_bad >= 6:
        weak += 0.26
        reasons.append("pen_bad_arms_6+")
    elif total_bad >= 4:
        weak += 0.18
        reasons.append("pen_bad_arms_4+")
    elif total_bad >= 2:
        weak += 0.08
        reasons.append("pen_bad_arms_2+")

    if recent_bad >= 4:
        weak += 0.16
        reasons.append("recent_bad_arms_4+")
    elif recent_bad >= 2:
        weak += 0.10
        reasons.append("recent_bad_arms_2+")
    elif recent_bad >= 1:
        weak += 0.05
        reasons.append("recent_bad_arms_1+")

    if depth == "high":
        weak += 0.18
        reasons.append("reliever_depth_high")
    elif depth == "medium":
        weak += 0.07
        reasons.append("reliever_depth_medium")

    if total_bad <= 1 and depth == "low":
        strong += 0.10
        reasons.append("clean_late_pen_profile")
    elif total_bad <= 2 and depth in {"low", ""}:
        strong += 0.05
        reasons.append("mostly_clean_late_pen_profile")

    return weak, strong, reasons


def _fp_team_pressure(
    *,
    team_prefix,
    offense_mult,
    lineup_impact,
    opposing_starter_stats,
    opposing_reliever_metrics,
):
    offense_pressure, offense_suppression, offense_reasons = _fp_offense_component(
        offense_mult,
        lineup_impact,
    )
    starter_weak, starter_strong, starter_reasons = _fp_starter_prevention_component(
        opposing_starter_stats,
    )
    pen_weak, pen_strong, pen_reasons = _fp_bullpen_prevention_component(
        opposing_reliever_metrics,
    )

    early_pressure = starter_weak + offense_pressure + 0.60 * min(starter_weak, offense_pressure)
    early_suppression = starter_strong + offense_suppression + 0.60 * min(starter_strong, offense_suppression)
    late_pressure = pen_weak + 0.55 * min(pen_weak, offense_pressure)
    late_suppression = pen_strong + 0.55 * min(pen_strong, offense_suppression)

    reasons = []
    reasons.extend([f"{team_prefix.lower()}_bat_{r}" for r in offense_reasons])
    reasons.extend([f"{team_prefix.lower()}_early_{r}" for r in starter_reasons])
    reasons.extend([f"{team_prefix.lower()}_late_{r}" for r in pen_reasons])

    return {
        f"{team_prefix}_Early_Run_Pressure": round(early_pressure, 3),
        f"{team_prefix}_Early_Run_Suppression": round(early_suppression, 3),
        f"{team_prefix}_Late_Run_Pressure": round(late_pressure, 3),
        f"{team_prefix}_Late_Run_Suppression": round(late_suppression, 3),
        f"{team_prefix}_Run_Pressure_Reasons": "|".join(reasons),
    }


def _calculate_full_picture_run_pressure(
    *,
    base_edge,
    has_real_total,
    away_stats,
    home_stats,
    away_lineup_impact,
    home_lineup_impact,
    away_offense_mult,
    home_offense_mult,
    weather_runs_mult,
    away_reliever_metrics=None,
    home_reliever_metrics=None,
):
    away = _fp_team_pressure(
        team_prefix="Away",
        offense_mult=away_offense_mult,
        lineup_impact=away_lineup_impact,
        opposing_starter_stats=home_stats,
        opposing_reliever_metrics=home_reliever_metrics,
    )
    home = _fp_team_pressure(
        team_prefix="Home",
        offense_mult=home_offense_mult,
        lineup_impact=home_lineup_impact,
        opposing_starter_stats=away_stats,
        opposing_reliever_metrics=away_reliever_metrics,
    )

    def _apply_bullpen_clarity_to_late_path(team_blob, team_prefix, opponent_reliever_metrics):
        """
        Bullpen clarity v1 -> real late-run math.

        Mapping:
        - Away offense late pressure/suppression is driven by HOME bullpen clarity.
        - Home offense late pressure/suppression is driven by AWAY bullpen clarity.

        This intentionally uses max(existing, clarity_score) instead of blindly adding,
        because the old late path already includes bad-arm/depth signals. The clarity
        score is the sharper per-arm/availability read and should upgrade weak old
        late reads without double-counting the same bullpen risk.
        """
        if not isinstance(team_blob, dict):
            return team_blob

        pressure_key = f"{team_prefix}_Late_Run_Pressure"
        suppression_key = f"{team_prefix}_Late_Run_Suppression"
        reasons_key = f"{team_prefix}_Run_Pressure_Reasons"

        old_pressure = _fp_safe_float(team_blob.get(pressure_key), 0.0)
        old_suppression = _fp_safe_float(team_blob.get(suppression_key), 0.0)

        clarity_pressure = _fp_metric(
            opponent_reliever_metrics,
            ["late_run_pressure_score"],
            0.0,
        )
        clarity_suppression = _fp_metric(
            opponent_reliever_metrics,
            ["late_run_suppression_score"],
            0.0,
        )

        new_pressure = max(old_pressure, clarity_pressure)
        new_suppression = max(old_suppression, clarity_suppression)

        new_pressure = max(0.0, min(1.25, float(new_pressure)))
        new_suppression = max(0.0, min(1.25, float(new_suppression)))

        tags = []
        if new_pressure > old_pressure + 0.001:
            tags.append(
                f"{team_prefix.lower()}_late_bullpen_clarity_pressure_{new_pressure:.2f}"
            )
        if new_suppression > old_suppression + 0.001:
            tags.append(
                f"{team_prefix.lower()}_late_bullpen_clarity_suppression_{new_suppression:.2f}"
            )

        team_blob[pressure_key] = round(new_pressure, 3)
        team_blob[suppression_key] = round(new_suppression, 3)

        if tags:
            existing = str(team_blob.get(reasons_key, "") or "").strip()
            add = "|".join(tags)
            team_blob[reasons_key] = f"{existing}|{add}" if existing else add

        return team_blob

    away = _apply_bullpen_clarity_to_late_path(
        away,
        "Away",
        home_reliever_metrics,
    )
    home = _apply_bullpen_clarity_to_late_path(
        home,
        "Home",
        away_reliever_metrics,
    )

    weather = _fp_safe_float(weather_runs_mult, 1.0)
    weather_adj = 0.0
    weather_reason = ""

    # Weather is already in base projection, so this is a small residual interaction only.
    if weather >= 1.015:
        weather_adj = 0.08
        weather_reason = "weather_boost_1.015+"
    elif weather <= 0.990:
        weather_adj = -0.08
        weather_reason = "weather_suppression_0.990-"

    away_raw = (
        away["Away_Early_Run_Pressure"]
        - away["Away_Early_Run_Suppression"]
        + away["Away_Late_Run_Pressure"]
        - away["Away_Late_Run_Suppression"]
    )
    home_raw = (
        home["Home_Early_Run_Pressure"]
        - home["Home_Early_Run_Suppression"]
        + home["Home_Late_Run_Pressure"]
        - home["Home_Late_Run_Suppression"]
    )
    raw = away_raw + home_raw + weather_adj

    # TEAM_RUN_ENVIRONMENT v3:
    # The prior v2 compressed all pressure into one total-only overlay. That hid
    # which team owned the scoring path. Keep normal profiles conservative, but
    # price extreme late-over paths on the side facing the bad bullpen.
    def _side_run_pressure_adjust(side_raw, late_pressure, late_suppression, reason_text, weather_half):
        try:
            sr = float(side_raw) + float(weather_half)
            lp = float(late_pressure)
            ls = float(late_suppression)
        except (TypeError, ValueError):
            return 0.0, False

        if not np.isfinite(sr) or not np.isfinite(lp) or not np.isfinite(ls):
            return 0.0, False

        reasons_l = str(reason_text or "").lower()
        extreme_late = (
            sr >= 1.00
            and lp >= 1.15
            and ls <= 0.10
            and "late_reliever_depth_high" in reasons_l
            and "late_bullpen_clarity_pressure" in reasons_l
            and (
                "late_recent_bad_arms_" in reasons_l
                or "late_pen_bad_arms_" in reasons_l
            )
        )

        # Extreme late-run pressure is already priced inside project_team_runs()
        # through the expanded bullpen-quality multiplier. Return the normal
        # side adjustment here only for classification; the caller suppresses
        # the entire second overlay when either side is extreme.
        normal_adj = max(-0.15, min(0.15, sr * 0.10))
        return normal_adj, extreme_late

    if not has_real_total:
        away_adj = 0.0
        home_adj = 0.0
        adj = 0.0
        mode = "no_real_total_no_adjust"
    else:
        weather_half = weather_adj / 2.0
        away_adj, away_extreme = _side_run_pressure_adjust(
            away_raw,
            away["Away_Late_Run_Pressure"],
            away["Away_Late_Run_Suppression"],
            away["Away_Run_Pressure_Reasons"],
            weather_half,
        )
        home_adj, home_extreme = _side_run_pressure_adjust(
            home_raw,
            home["Home_Late_Run_Pressure"],
            home["Home_Late_Run_Suppression"],
            home["Home_Run_Pressure_Reasons"],
            weather_half,
        )
        if away_extreme or home_extreme:
            # Core team-run math has already expanded the affected bullpen
            # exposure. Do not apply a second game-level pressure overlay.
            away_adj = 0.0
            home_adj = 0.0
            adj = 0.0
            mode = "team_run_environment_extreme_core_only_no_overlay"
        else:
            adj = away_adj + home_adj
            mode = "team_run_environment_coef_0_10_sidecap_0_15"

    reasons = []
    if away["Away_Run_Pressure_Reasons"]:
        reasons.append(away["Away_Run_Pressure_Reasons"])
    if home["Home_Run_Pressure_Reasons"]:
        reasons.append(home["Home_Run_Pressure_Reasons"])
    if weather_reason:
        reasons.append(weather_reason)
    reasons.append(mode)

    out = {}
    out.update(away)
    out.update(home)
    out["Weather_Interaction_Adjustment"] = round(weather_adj, 3)
    out["Full_Picture_Adjustment_Raw"] = round(raw, 3)
    out["Away_Run_Pressure_Adjustment"] = round(away_adj, 3)
    out["Home_Run_Pressure_Adjustment"] = round(home_adj, 3)
    out["Run_Pressure_Adjustment"] = round(adj, 3)
    out["Run_Pressure_Mode"] = mode
    out["Run_Pressure_Reasons"] = "|".join(reasons)
    return out


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
    weather_runs_mult=1.0,
    away_pitcher_name="",
    home_pitcher_name="",
    game_datetime=None,
    schedule_game_date=None,
    odds_info=None,
    away_reliever_metrics=None,
    home_reliever_metrics=None,
):
    """
    Project expected runs for each team, sum to projected total, then compare to Vegas.

    weather_runs_mult: bounded daily environment overlay (temp/wind) on top of static park factor;
      combined as effective_park_runs_factor = park_runs_factor * weather_runs_mult.

    away_pitcher_name / home_pitcher_name / game_datetime / schedule_game_date: optional inputs
      for days-rest v1 starter xERA bump (see core.starter_fatigue); schedule_game_date should
      be statsapi schedule game_date (YYYY-MM-DD) when available; defaults leave behavior unchanged.

    Returns dict with: projected_total, away_runs, home_runs, edge, pick, prediction (str),
    confidence, total_open, total_current, recommended_units, skip (bool).
    """
    if public_data is None:
        public_data = {}
    if odds_info is None:
        odds_info = {}
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

    away_xera = safe_get(away_stats, "xERA", 4.50) + xera_delta_for_pitcher_days_rest(
        away_pitcher_name, game_datetime, schedule_game_date
    )
    home_xera = safe_get(home_stats, "xERA", 4.50) + xera_delta_for_pitcher_days_rest(
        home_pitcher_name, game_datetime, schedule_game_date
    )
    away_whip = safe_get(away_stats, "WHIP", 1.30)
    home_whip = safe_get(home_stats, "WHIP", 1.30)
    bullpen_home_era = safe_get(bullpen_home, "ERA", 4.25)
    bullpen_away_era = safe_get(bullpen_away, "ERA", 4.25)
    # Stage A: pull xERA alongside ERA for the O/U bullpen-quality blend in
    # project_team_runs. Default (None) preserves current ERA-only behavior
    # when the bullpen dict lacks xERA (pre-Stage-A loader, test stubs, etc.).
    bullpen_home_xera = safe_get(bullpen_home, "xERA", None)
    bullpen_away_xera = safe_get(bullpen_away, "xERA", None)
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
    try:
        _wm = float(weather_runs_mult)
    except (TypeError, ValueError):
        _wm = 1.0
    if not np.isfinite(_wm) or _wm <= 0:
        _wm = 1.0
    _wm = max(WEATHER_RUNS_MULT_MIN, min(WEATHER_RUNS_MULT_MAX, _wm))
    effective_park_runs_factor = over_boost * _wm

    away_workload = _ou_workload_for_pitcher(away_pitcher_name)
    home_workload = _ou_workload_for_pitcher(home_pitcher_name)
    away_expected_starter_ip = away_workload.get("expected_ip", "")
    home_expected_starter_ip = home_workload.get("expected_ip", "")
    away_starter_ip_share = float(away_workload.get("starter_share", STARTER_IP_SHARE))
    home_starter_ip_share = float(home_workload.get("starter_share", STARTER_IP_SHARE))
    away_bullpen_ip_share = float(away_workload.get("bullpen_share", BULLPEN_IP_SHARE))
    home_bullpen_ip_share = float(home_workload.get("bullpen_share", BULLPEN_IP_SHARE))

    # ---------- Project runs for each team ----------
    # Away offense faces home pitcher + home bullpen; away_offense_mult from Batters.offense_vs_hand_dict(away_team vs home_hand)
    away_runs, away_runs_safety = project_team_runs(
        opponent_starter_xera=home_xera,
        opponent_starter_whip=home_whip,
        opponent_bullpen_era=bullpen_home_era,
        opponent_bullpen_ip_week=bullpen_home_ip_week,
        opponent_bullpen_relievers=bullpen_home_rel,
        park_runs_factor=effective_park_runs_factor,
        lineup_impact=away_lineup_impact,
        opponent_velo_drop=velo_drop_home,
        opponent_low_ip=safe_get(home_stats, "LowIP", False),
        offense_mult=away_offense_mult,
        opponent_bullpen_xera=bullpen_home_xera,
        opponent_late_run_pressure_score=_fp_metric(home_reliever_metrics, ["late_run_pressure_score"], 0.0),
        opponent_late_run_suppression_score=_fp_metric(home_reliever_metrics, ["late_run_suppression_score"], 0.0),
        opponent_bullpen_clarity_score=_fp_metric(home_reliever_metrics, ["bullpen_clarity_score"], 0.0),
        opponent_reliever_depth_risk=(home_reliever_metrics or {}).get("depth_risk", ""),
        opponent_bullpen_quality_tier=(home_reliever_metrics or {}).get("bullpen_quality_tier", ""),
        weather_runs_mult=_wm,
        opponent_expected_starter_ip=home_expected_starter_ip,
    )
    # Home offense faces away pitcher + away bullpen; home_offense_mult from Batters.offense_vs_hand_dict(home_team vs away_hand)
    home_runs, home_runs_safety = project_team_runs(
        opponent_starter_xera=away_xera,
        opponent_starter_whip=away_whip,
        opponent_bullpen_era=bullpen_away_era,
        opponent_bullpen_ip_week=bullpen_away_ip_week,
        opponent_bullpen_relievers=bullpen_away_rel,
        park_runs_factor=effective_park_runs_factor,
        lineup_impact=home_lineup_impact,
        opponent_velo_drop=velo_drop_away,
        opponent_low_ip=safe_get(away_stats, "LowIP", False),
        offense_mult=home_offense_mult,
        opponent_bullpen_xera=bullpen_away_xera,
        opponent_late_run_pressure_score=_fp_metric(away_reliever_metrics, ["late_run_pressure_score"], 0.0),
        opponent_late_run_suppression_score=_fp_metric(away_reliever_metrics, ["late_run_suppression_score"], 0.0),
        opponent_bullpen_clarity_score=_fp_metric(away_reliever_metrics, ["bullpen_clarity_score"], 0.0),
        opponent_reliever_depth_risk=(away_reliever_metrics or {}).get("depth_risk", ""),
        opponent_bullpen_quality_tier=(away_reliever_metrics or {}).get("bullpen_quality_tier", ""),
        weather_runs_mult=_wm,
        opponent_expected_starter_ip=away_expected_starter_ip,
    )

    # ---------- F5 telemetry projection (export-only, no market line) ----------
    # Starter-only, 5-inning-scaled. Never feeds OU_Fired / ML_Fired / OU_Edge /
    # Confidence / Kelly / Telegram. Parlay does not expose F5 lines yet.
    f5_away_runs = project_team_f5_runs(
        opponent_starter_xera=home_xera,
        opponent_starter_whip=home_whip,
        park_runs_factor=effective_park_runs_factor,
        lineup_impact=away_lineup_impact,
        opponent_velo_drop=velo_drop_home,
        opponent_low_ip=safe_get(home_stats, "LowIP", False),
        offense_mult=away_offense_mult,
    )
    f5_home_runs = project_team_f5_runs(
        opponent_starter_xera=away_xera,
        opponent_starter_whip=away_whip,
        park_runs_factor=effective_park_runs_factor,
        lineup_impact=home_lineup_impact,
        opponent_velo_drop=velo_drop_away,
        opponent_low_ip=safe_get(away_stats, "LowIP", False),
        offense_mult=home_offense_mult,
    )
    f5_projected_total = round(f5_away_runs + f5_home_runs, 2)

    _away_xera_raw = safe_get(away_stats, "xERA", None)
    _home_xera_raw = safe_get(home_stats, "xERA", None)

    def _is_real_xera(v):
        try:
            return v is not None and np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    f5_eligible = bool(
        _is_real_xera(_away_xera_raw)
        and _is_real_xera(_home_xera_raw)
        and not safe_get(away_stats, "LowIP", False)
        and not safe_get(home_stats, "LowIP", False)
    )

    _xera_gap = float(home_xera) - float(away_xera)
    if _xera_gap >= 0.40:
        f5_starter_lean = "AWAY"
    elif _xera_gap <= -0.40:
        f5_starter_lean = "HOME"
    else:
        f5_starter_lean = "EVEN"

    _f5_baseline = LEAGUE_RUNS_PER_TEAM * 2.0 * (5.0 / 9.0)
    if f5_projected_total - _f5_baseline >= 0.30:
        f5_model_side = "OVER"
    elif f5_projected_total - _f5_baseline <= -0.30:
        f5_model_side = "UNDER"
    else:
        f5_model_side = "EVEN"

    f5_market_line = None
    for bm in odds_info.get("bookmakers", []):
        try:
            if (bm.get("key") or "").lower() != "pinnacle":
                continue
            for market in bm.get("markets", []):
                if market.get("key") != "totals_h1":
                    continue
                outcomes = market.get("outcomes", [])
                if outcomes:
                    f5_market_line = outcomes[0].get("point")
                    break
            if f5_market_line is not None:
                break
        except (AttributeError, TypeError, IndexError):
            continue
    f5_no_line_reason = None if f5_market_line is not None else "no_f5_market_source"

    # Analytical truth for O/U (uncapped); cap is parallel safety only.
    away_runs_raw = away_runs
    home_runs_raw = home_runs

    # Dynamic per-team cap: flag when analytical projection exceeds safety-capped projection (O/U customer-fire gating).
    projection_cap_hit = (float(away_runs) > float(away_runs_safety)) or (
        float(home_runs) > float(home_runs_safety)
    )

    base_projected_total = round(away_runs + home_runs, 2)
    base_ou_edge = round(base_projected_total - vegas_line, 2)

    run_pressure = _calculate_full_picture_run_pressure(
        base_edge=base_ou_edge,
        has_real_total=bool(odds_info.get("_has_real_total")),
        away_stats=away_stats,
        home_stats=home_stats,
        away_lineup_impact=away_lineup_impact,
        home_lineup_impact=home_lineup_impact,
        away_offense_mult=away_offense_mult,
        home_offense_mult=home_offense_mult,
        weather_runs_mult=weather_runs_mult,
        away_reliever_metrics=away_reliever_metrics,
        home_reliever_metrics=home_reliever_metrics,
    )
    away_run_pressure_adjustment = _fp_safe_float(
        run_pressure.get("Away_Run_Pressure_Adjustment"),
        0.0,
    )
    home_run_pressure_adjustment = _fp_safe_float(
        run_pressure.get("Home_Run_Pressure_Adjustment"),
        0.0,
    )
    run_pressure_adjustment = round(
        float(away_run_pressure_adjustment) + float(home_run_pressure_adjustment),
        3,
    )

    # Apply run pressure to the team that owns the scoring path. This preserves
    # Base_Projected_Total as the pre-pressure baseline and makes Away_Runs /
    # Home_Runs match the adjusted total path.
    if bool(odds_info.get("_has_real_total")):
        away_runs = round(max(0.0, float(away_runs) + float(away_run_pressure_adjustment)), 2)
        home_runs = round(max(0.0, float(home_runs) + float(home_run_pressure_adjustment)), 2)
        away_runs_safety = round(max(0.0, float(away_runs_safety) + float(away_run_pressure_adjustment)), 2)
        home_runs_safety = round(max(0.0, float(home_runs_safety) + float(home_run_pressure_adjustment)), 2)
        away_runs_raw = away_runs
        home_runs_raw = home_runs

    raw_projected_total = round(float(away_runs) + float(home_runs), 2)
    raw_edge = round(raw_projected_total - vegas_line, 2)

    # Only calibrate against a real/trusted total. Do not pull projection toward
    # default/fallback lines because that would hide data-quality problems.
    if bool(odds_info.get("_has_real_total")):
        edge, ou_edge_calibration_factor, ou_edge_calibration_tag = _calibrate_ou_edge(raw_edge)
        projected_total = round(vegas_line + edge, 2)
    else:
        edge = raw_edge
        projected_total = raw_projected_total
        ou_edge_calibration_factor = 1.0
        ou_edge_calibration_tag = "uncalibrated_no_real_total"

    # ---------- Pick: OVER / UNDER based on edge vs threshold ----------
    if edge >= EDGE_THRESHOLD:
        pick = "OVER"
        prediction_str = f"OVER {vegas_line:.1f}"
    elif edge <= -EDGE_THRESHOLD:
        pick = "UNDER"
        prediction_str = f"UNDER {vegas_line:.1f}"
    else:
        pick = "LEAN_OVER" if edge > 0 else "LEAN_UNDER"
        prediction_str = f"LEAN {'OVER' if edge > 0 else 'UNDER'} {vegas_line:.1f} (proj {projected_total:.1f})"

    # ---------- O/U confidence: edge strength × data quality × market trust (additive nudges) ----------
    # base_confidence = edge strength (projection vs line); data_quality_factor = uncertainty / completeness;
    # market_trust_factor = reserved multiplicative market trust (1.0); public + line = small additive market nudges.
    abs_edge = abs(edge)
    if abs_edge >= 1.0:
        # Top bucket: slight lift vs 0.75 so |edge|>=1 rows with routine velo
        # trims (~0.99) and aligned sharp (+0.04) can reach OU_Fired (0.79)
        # without lowering the gate or trimming other buckets.
        base_confidence = 0.76
    elif abs_edge >= 0.5:
        base_confidence = 0.67
    elif abs_edge >= EDGE_THRESHOLD:
        base_confidence = 0.58
    else:
        base_confidence = 0.45

    # (Bullpen/park already in projection; avoid re-using for confidence to prevent bias)

    # Meaningful velocity loss on either starter: modest multiplicative penalty (same as prior 0.99 trims).
    _velo_trust = (
        _confidence_trim_for_velocity_loss(velo_drop_away) * _confidence_trim_for_velocity_loss(velo_drop_home)
    )
    # Reliever depth used to apply a pick-dependent multiplier (OVER vs UNDER),
    # giving typical UNDER rows +5% confidence vs OVER at full bullpen depth.
    # Neutralized: bullpen workload/quality stay in project_team_runs only;
    # O/U confidence no longer gets a structural UNDER advantage at the gate.
    reliever_mult = 1.0

    away_pitcher = away_pitcher_name or ""
    home_pitcher = home_pitcher_name or ""

    league_avg_penalty = 1.0
    low_ip_penalty = 1.0

    has_league_avg_starter = (
        "League Avg" in away_pitcher
        or "League Avg" in home_pitcher
    )

    if has_league_avg_starter:
        league_avg_penalty = LEAGUE_AVG_OU_CONFIDENCE_MULT

    # Real low-IP starters get a tiered confidence trim by actual IP.
    # League Avg fallback shells may carry LowIP=True internally, but they should
    # be handled by the league-average penalty, not double-stacked as real LowIP.
    if not has_league_avg_starter:
        away_low_ip_mult = _ou_low_ip_confidence_multiplier(
            safe_get(away_stats, "IP", 0.0),
            safe_get(away_stats, "LowIP", False),
        )
        home_low_ip_mult = _ou_low_ip_confidence_multiplier(
            safe_get(home_stats, "IP", 0.0),
            safe_get(home_stats, "LowIP", False),
        )
        low_ip_penalty = min(float(away_low_ip_mult), float(home_low_ip_mult))

    data_quality_factor = _velo_trust * reliever_mult
    data_quality_factor *= league_avg_penalty * low_ip_penalty

    market_trust_factor = 1.0

    confidence = base_confidence * data_quality_factor * market_trust_factor

    # Loader uses total_open / total_current; accept legacy Total_Open / Total_Current.
    _raw_to = public_data.get("total_open")
    if _raw_to is None or _raw_to == "":
        _raw_to = public_data.get("Total_Open")
    _raw_tc = public_data.get("total_current")
    if _raw_tc is None or _raw_tc == "":
        _raw_tc = public_data.get("Total_Current")
    total_open = safe_float(_raw_to, default=vegas_line)
    total_current = safe_float(_raw_tc, default=total_open)

    # Public betting: fade heavy public (contrarian); additive market-trust nudges (unchanged behavior).
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

    if pick in ("OVER", "UNDER"):
        line_sign = 1 if pick == "OVER" else -1
        if total_current > total_open:
            confidence += line_sign * 0.02
        elif total_current < total_open:
            confidence -= line_sign * 0.02

    confidence = max(0.01, min(confidence, 0.99))
    if not has_real_total:
        confidence = min(0.59, confidence * 0.65)
        prediction_str = "NO BET (fallback total only)"

    # Recommended units from edge size (only bet when |edge| >= threshold)
    if abs_edge < EDGE_THRESHOLD:
        recommended_units = 0.0
    elif abs_edge >= EDGE_FOR_FULL_UNIT:
        recommended_units = 1.0
    else:
        recommended_units = round(0.5 + 0.5 * (abs_edge - EDGE_THRESHOLD) / (EDGE_FOR_FULL_UNIT - EDGE_THRESHOLD), 2)

    print(
        f"📦 Base Projection: {away_runs:.2f} + {home_runs:.2f} = {base_projected_total:.2f} | "
        f"RunPressure={run_pressure_adjustment:+.2f} | Raw O/U={raw_projected_total:.2f} | "
        f"Calibrated O/U: {projected_total:.2f} ({ou_edge_calibration_tag}, factor={ou_edge_calibration_factor:.2f}) | "
        f"Edge: {edge:+.2f} | {prediction_str} | Conf: {confidence:.2f}"
    )

    # --- Stage B telemetry (additive, non-behavior-altering) ---
    # Replays of math already performed inside project_team_runs and the
    # confidence stack above. Values here are purely observational and do
    # NOT feed back into projection, confidence, or fire gates. Stamped
    # onto the archive per row so lineup × bullpen / fatigue / confidence-
    # stack hypotheses can be validated from the CSV without rerunning.
    def _telemetry_bp_fatigue(n_rel, ip_week):
        try:
            n = max(1, int(round(float(n_rel))))
        except (TypeError, ValueError):
            n = 7
        exp_ip = n * BULLPEN_EXPECTED_IP_PER_RELIEVER_WEEK
        try:
            ipw = float(ip_week)
        except (TypeError, ValueError):
            ipw = 0.0
        if exp_ip <= 0 or ipw <= 0:
            return 0.0, 1.0
        ratio = ipw / exp_ip
        mult = _bullpen_workload_fatigue_multiplier(ratio)
        return round(ratio, 4), round(mult, 4)

    def _telemetry_bp_quality(era, xera):
        try:
            e = float(era)
        except (TypeError, ValueError):
            return None
        if xera is None:
            return e
        try:
            x = float(xera)
        except (TypeError, ValueError):
            return e
        if not np.isfinite(x):
            return e
        return 0.75 * e + 0.25 * x

    def _telemetry_starter_adj(x, low_ip):
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return None
        return min(xf + LOW_IP_XERA_PENALTY, 6.0) if bool(low_ip) else xf

    _tel_away_bp_ratio, _tel_away_bp_fmult = _telemetry_bp_fatigue(
        bullpen_away_rel, bullpen_away_ip_week
    )
    _tel_home_bp_ratio, _tel_home_bp_fmult = _telemetry_bp_fatigue(
        bullpen_home_rel, bullpen_home_ip_week
    )

    _tel_bq_home = _telemetry_bp_quality(bullpen_home_era, bullpen_home_xera)
    _tel_bq_away = _telemetry_bp_quality(bullpen_away_era, bullpen_away_xera)

    # Mirror the low-IP starter adjustment that project_team_runs applies,
    # so Effective_ERA telemetry matches what the projection actually used.
    _tel_opp_starter_for_away = _telemetry_starter_adj(
        home_xera, safe_get(home_stats, "LowIP", False)
    )
    _tel_opp_starter_for_home = _telemetry_starter_adj(
        away_xera, safe_get(away_stats, "LowIP", False)
    )

    # Away offense runs projection faces home starter + home bullpen
    # Home offense runs projection faces away starter + away bullpen
    if _tel_opp_starter_for_away is not None and _tel_bq_home is not None:
        _tel_away_eff_era = round(
            home_starter_ip_share * _tel_opp_starter_for_away
            + home_bullpen_ip_share * _tel_bq_home,
            4,
        )
    else:
        _tel_away_eff_era = ""
    if _tel_opp_starter_for_home is not None and _tel_bq_away is not None:
        _tel_home_eff_era = round(
            away_starter_ip_share * _tel_opp_starter_for_home
            + away_bullpen_ip_share * _tel_bq_away,
            4,
        )
    else:
        _tel_home_eff_era = ""

    return {
        "skip": False,
        "projected_total": projected_total,
        "base_projected_total": base_projected_total,
        "base_ou_edge": base_ou_edge,
        "raw_projected_total": raw_projected_total,
        "raw_ou_edge": raw_edge,
        "ou_edge_calibration_factor": ou_edge_calibration_factor,
        "ou_edge_calibration_tag": ou_edge_calibration_tag,
        "away_early_run_pressure": run_pressure.get("Away_Early_Run_Pressure", ""),
        "away_early_run_suppression": run_pressure.get("Away_Early_Run_Suppression", ""),
        "away_late_run_pressure": run_pressure.get("Away_Late_Run_Pressure", ""),
        "away_late_run_suppression": run_pressure.get("Away_Late_Run_Suppression", ""),
        "home_early_run_pressure": run_pressure.get("Home_Early_Run_Pressure", ""),
        "home_early_run_suppression": run_pressure.get("Home_Early_Run_Suppression", ""),
        "home_late_run_pressure": run_pressure.get("Home_Late_Run_Pressure", ""),
        "home_late_run_suppression": run_pressure.get("Home_Late_Run_Suppression", ""),
        "weather_interaction_adjustment": run_pressure.get("Weather_Interaction_Adjustment", ""),
        "full_picture_adjustment_raw": run_pressure.get("Full_Picture_Adjustment_Raw", ""),
        "away_run_pressure_adjustment": run_pressure.get("Away_Run_Pressure_Adjustment", ""),
        "home_run_pressure_adjustment": run_pressure.get("Home_Run_Pressure_Adjustment", ""),
        "run_pressure_adjustment": run_pressure.get("Run_Pressure_Adjustment", ""),
        "run_pressure_mode": run_pressure.get("Run_Pressure_Mode", ""),
        "run_pressure_reasons": run_pressure.get("Run_Pressure_Reasons", ""),
        "away_runs": away_runs,
        "home_runs": home_runs,
        "away_runs_raw": away_runs_raw,
        "home_runs_raw": home_runs_raw,
        "away_runs_safety": away_runs_safety,
        "home_runs_safety": home_runs_safety,
        "projection_cap_hit": projection_cap_hit,
        "f5_projected_total": f5_projected_total,
        "f5_away_runs": f5_away_runs,
        "f5_home_runs": f5_home_runs,
        "f5_model_side": f5_model_side,
        "f5_starter_lean": f5_starter_lean,
        "f5_eligible": f5_eligible,
        "f5_market_line": f5_market_line,
        "f5_no_line_reason": f5_no_line_reason,
        "vegas_line": vegas_line,
        "edge": edge,
        "pick": pick,
        "prediction": prediction_str,
        "confidence": round(confidence, 2),
        "total_open": total_open,
        "total_current": total_current,
        "recommended_units": recommended_units,
        "telemetry": {
            "away_bullpen_era": bullpen_away_era,
            "home_bullpen_era": bullpen_home_era,
            "away_bullpen_xera": bullpen_away_xera if bullpen_away_xera is not None else "",
            "home_bullpen_xera": bullpen_home_xera if bullpen_home_xera is not None else "",
            "away_bullpen_ip_week": bullpen_away_ip_week,
            "home_bullpen_ip_week": bullpen_home_ip_week,
            "away_bullpen_relievers": bullpen_away_rel,
            "home_bullpen_relievers": bullpen_home_rel,
            "away_bullpen_fatigue_ratio": _tel_away_bp_ratio,
            "home_bullpen_fatigue_ratio": _tel_home_bp_ratio,
            "away_bullpen_fatigue_mult": _tel_away_bp_fmult,
            "home_bullpen_fatigue_mult": _tel_home_bp_fmult,
            "away_effective_era": _tel_away_eff_era,
            "home_effective_era": _tel_home_eff_era,
            "away_expected_starter_ip": away_expected_starter_ip,
            "home_expected_starter_ip": home_expected_starter_ip,
            "away_starter_ip_share": round(float(away_starter_ip_share), 4),
            "home_starter_ip_share": round(float(home_starter_ip_share), 4),
            "away_bullpen_ip_share": round(float(away_bullpen_ip_share), 4),
            "home_bullpen_ip_share": round(float(home_bullpen_ip_share), 4),
            "away_workload_profile": away_workload.get("profile", ""),
            "home_workload_profile": home_workload.get("profile", ""),
            "away_workload_source": away_workload.get("source", ""),
            "home_workload_source": home_workload.get("source", ""),
            # Starter xERA after days-rest adjustment — same floats passed to project_team_runs.
            "away_starter_xera": round(float(away_xera), 4),
            "home_starter_xera": round(float(home_xera), 4),
            "ou_base_confidence": round(float(base_confidence), 4),
            "ou_reliever_mult": round(float(reliever_mult), 4),
            "ou_low_ip_mult": round(float(low_ip_penalty), 4),
            "ou_league_avg_mult": round(float(league_avg_penalty), 4),
            "ou_velo_trust_mult": round(float(_velo_trust), 4),
        },
    }

# ================================
# 💬 TELEGRAM UTILS
# ================================
def send_telegram_alert(message):
    if model_telegram_suppressed():
        print("📵 Telegram alert suppressed by OVERGANG_SUPPRESS_MODEL_TELEGRAM=1")
        return False
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


def _parse_ou_confidence_percent_optional(game_data: dict):
    """O/U confidence on 0–100 scale, or None if missing/unparsable (for bucket labels only)."""
    try:
        raw_conf = game_data.get("Confidence_Value")
        if raw_conf is None:
            s = str(game_data.get("Confidence", "")).replace("%", "").strip()
            if not s:
                return None
            v = float(s)
        else:
            v = float(raw_conf)
            if 0 <= v <= 1.0:
                v = v * 100.0
        if v != v:  # NaN
            return None
        if not (0.0 <= v <= 100.0):
            return None
        return float(v)
    except Exception:
        return None


def _parse_ml_confidence_percent_optional(game_data: dict):
    """ML win-prob percent from ML_Confidence string, or None if missing/unparsable."""
    try:
        s = str(game_data.get("ML_Confidence", "")).replace("%", "").strip()
        if not s:
            return None
        v = float(s)
        if v != v:
            return None
        if not (0.0 <= v <= 100.0):
            return None
        return float(v)
    except Exception:
        return None


def _confidence_bucket_5pt(pct) -> str:
    """
    Non-overlapping 5-point labels: 0-4, 5-9, …, 95-100.
    Empty string if pct is None or not usable.
    """
    if pct is None:
        return ""
    try:
        x = float(pct)
    except (TypeError, ValueError):
        return ""
    if x != x:  # NaN
        return ""
    x = max(0.0, min(100.0, x))
    lo = min(95, (int(x) // 5) * 5)
    hi = 100 if lo == 95 else lo + 4
    return f"{lo}-{hi}"


def _fmt_num_one(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.1f}"
    return str(v)


def _alert_clean(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if s.lower() in ("nan", "none", "null", "<na>"):
        return ""
    return s


def _telegram_markdown_escape(v) -> str:
    s = _alert_clean(v)
    if not s:
        return ""
    return (
        s.replace("\\", "\\\\")
        .replace("_", "\\_")
        .replace("*", "\\*")
        .replace("`", "\\`")
        .replace("[", "\\[")
    )


def _alert_bool(v) -> bool:
    if v is True:
        return True
    return str(v).strip().lower() in ("true", "1", "yes")


def _alert_float_or_none(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "none", "null", "<na>"):
            return None
        f = float(v)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _is_f5_telegram_candidate(row: dict) -> bool:
    if not _alert_bool(row.get("F5_Market_OK")):
        return False
    edge = _alert_float_or_none(row.get("F5_Edge"))
    if edge is None:
        return False
    abs_edge = abs(edge)
    in_tight_bucket = (
        0.75 <= abs_edge <= 1.00
        or abs_edge >= 1.50
    )
    if not in_tight_bucket:
        return False
    pick = _alert_clean(row.get("Daily_F5_Profile_Pick")) or _alert_clean(row.get("F5_Pick"))
    if not pick:
        return False
    if _alert_clean(row.get("F5_Market_Line")) == "":
        return False
    grade = _alert_clean(row.get("Daily_F5_Profile_Grade"))
    if grade.lower() == "risk":
        return False
    return True


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


def format_f5_alert(game_data: dict) -> str:
    """Customer-facing F5 Telegram message (CSV remains full detail)."""
    if game_data.get("Datetime"):
        t = _alert_formatted_time(game_data)
    else:
        t = _alert_clean(game_data.get("Game_Time_MT")) or "TBD"
    pitchers = game_data.get("Pitchers")
    if not pitchers:
        away_p = game_data.get("Away_Pitcher") or "?"
        home_p = game_data.get("Home_Pitcher") or "?"
        pitchers = f"{away_p} vs {home_p}"
    away_xera = _alert_clean(game_data.get("Away_Starter_xERA"))
    home_xera = _alert_clean(game_data.get("Home_Starter_xERA"))
    away_whip = _alert_clean(game_data.get("Away_Starter_WHIP"))
    home_whip = _alert_clean(game_data.get("Home_Starter_WHIP"))
    xera_str = f"{away_xera or '-'}/{home_xera or '-'}"
    whip_str = f"{away_whip or '-'}/{home_whip or '-'}"
    pick = (
        _alert_clean(game_data.get("Daily_F5_Profile_Pick"))
        or _alert_clean(game_data.get("F5_Pick"))
        or "-"
    )
    edge_f = _alert_float_or_none(game_data.get("F5_Edge"))
    edge_str = f"{edge_f:+.1f}" if edge_f is not None else "?"
    grade = _alert_clean(game_data.get("Daily_F5_Profile_Grade"))
    analysis = _alert_clean(game_data.get("Daily_F5_Analysis_Read")) or _alert_clean(
        game_data.get("Daily_F5_Profile_Reason")
    )
    ou_context = _alert_clean(game_data.get("Daily_OU_Profile_Read"))
    game_display = _telegram_markdown_escape(game_data.get("Game", "Unknown"))
    venue_display = _telegram_markdown_escape(game_data.get("Venue", "Unknown"))
    pitchers_display = _telegram_markdown_escape(pitchers)
    pick_display = _telegram_markdown_escape(pick)
    grade_display = _telegram_markdown_escape(grade)
    analysis_display = _telegram_markdown_escape(
        analysis.replace("|", ", ") if analysis else ""
    )
    ou_context_display = _telegram_markdown_escape(
        ou_context.replace("|", ", ") if ou_context else ""
    )
    xera_display = _telegram_markdown_escape(xera_str)
    whip_display = _telegram_markdown_escape(whip_str)
    edge_display = _telegram_markdown_escape(edge_str)
    lines = [
        "\u23f1\ufe0f *F5 \u00b7 Over Gang*",
        f"\U0001f3df\ufe0f *{game_display}*",
        f"\U0001f4cd {venue_display} | \U0001f552 {t}",
        "",
        f"\U0001f3af {pitchers_display}",
        f"\U0001f4ca xERA {xera_display} \u00b7 WHIP {whip_display}",
        "",
        f"\U0001f9e0 *Pick*: {pick_display}",
        f"\U0001f4d0 *F5 Edge*: {edge_display}",
    ]
    if grade:
        lines.append(f"\U0001f3f7\ufe0f *Profile Grade*: {grade_display}")
    if analysis:
        lines.append(f"\U0001f9fe {analysis_display}")
    if ou_context:
        lines.append(f"\U0001f3af Full-game context: {ou_context_display}")
    lines.extend([
        "",
        "18+ only. For informational purposes only. Past performance does not guarantee future results. Bet responsibly.",
    ])
    return "\n".join(lines)


def send_telegram_file(file_path, caption="📊 Over Gang Predictions"):
    if model_telegram_suppressed():
        print(f"📵 Telegram file suppressed by OVERGANG_SUPPRESS_MODEL_TELEGRAM=1: {file_path}")
        return False
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
                home_team, away_team, odds_map, game_datetime=safe_get(g, "game_datetime", ""), emit_live_total_diagnostics=False
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


def _trusted_total_source_row_for_game(game, trusted_total_source_map):
    """
    Resolve the trusted total-source row for a slate game using the same event-aware lookup
    behavior as live odds resolution. Reporting/readiness only; does not change betting logic.
    """
    if not isinstance(trusted_total_source_map, dict) or not isinstance(game, dict):
        return None
    away_team = safe_get(game, "away_name", "")
    home_team = safe_get(game, "home_name", "")
    row = get_game_odds(
        away_team,
        home_team,
        trusted_total_source_map,
        commence_time=safe_get(game, "game_datetime", ""),
    )
    if not bool(row.get("_match_found", False)):
        return None
    if row.get("_raw_total_line") in (None, ""):
        return None
    return row


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
                home_team, away_team, odds_map, game_datetime=safe_get(g, "game_datetime", ""), emit_live_total_diagnostics=False
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
# StatsAPI schedule g['status'] (string or nested detailedState): allowlist only — prediction runs pregame slate only.
_PREGAME_STATUSES_FOR_PREDICTION = frozenset({
    "scheduled",
    "pre-game",
    "pregame",
    "warmup",
    "preview",
})


def _statsapi_status_base_string(g) -> str:
    """Normalized first segment of game status for allowlist match (statsapi string or dict)."""
    st = g.get("status")
    if isinstance(st, dict):
        st = st.get("detailedState") or st.get("abstractGameState") or ""
    if st is None:
        return ""
    s = str(st).strip().lower()
    if not s:
        return ""
    return s.split(":")[0].strip()


def _game_status_for_export(g) -> str:
    """Display status for CSV (additive column)."""
    st = g.get("status")
    if isinstance(st, dict):
        v = st.get("detailedState") or st.get("abstractGameState")
        return str(v).strip() if v is not None else ""
    if st is None:
        return ""
    return str(st).strip()


def _game_is_playable_for_prediction(g) -> bool:
    """
    Pregame-only slate gate: g['status'] from statsapi.schedule must match _PREGAME_STATUSES_FOR_PREDICTION.

    Excludes In Progress, Final, Game Over, Completed, postponed/cancelled/suspended, and unknown/empty status.
    """
    base = _statsapi_status_base_string(g)
    if not base:
        return False
    return base in _PREGAME_STATUSES_FOR_PREDICTION


def _main_the_odds_api_fallback_allowed():
    return str(os.getenv("OVERGANG_ALLOW_THE_ODDS_API_MAIN_FALLBACK", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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
        # Live odds / totals: Odds API only (SportsDataIO phased out of this path; fetch_mlb_odds_by_date_allow_empty_book remains available elsewhere).
        print("[ODDS] Fetching parlay_api (active live odds source; SportsDataIO not used for odds_map)...")
        odds_api_map = fetch_mlb_odds(target_date=target_date_str) or {}
        print(f"[ODDS] parlay_api rows: {len(odds_api_map)}")
        odds_source = "parlay_api" if odds_api_map else "none"

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
            if not book:
                return False
            if book.lower().strip() == "scrambled":
                return False
            return True

        # Trusted real-total lane: Odds API only
        trusted_total_source_map = {}
        for k, v in (odds_api_map or {}).items():
            if _is_trusted_total_row(v):
                trusted_total_source_map[k] = dict(v)
                trusted_total_source_map[k]["_trusted_source"] = "parlay_api"

        # Base container = Odds API only; strip weak totals, then overlay trusted rows
        odds_map = dict(odds_api_map or {})
        for k in list(odds_map.keys()):
            if k not in trusted_total_source_map:
                row = dict(odds_map.get(k, {}))
                row["total_line"] = None
                row["book"] = ""
                odds_map[k] = row

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

        print(
            f"[ODDS] Active live odds source: {odds_source}"
            + (" | trusted total keys: {}".format(len(trusted_total_source_map)) if odds_api_map else " — empty map; VegasLines will use manual CSV or 8.5 fallback where applicable")
        )

        # Keep only games whose local MT calendar date matches the target slate (today_mt)
        def game_mt_date(g):
            dt_utc = datetime.strptime(
                g["game_datetime"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=ZoneInfo("UTC"))
            return dt_utc.astimezone(ZoneInfo("America/Denver")).date()

        games = [g for g in games if game_mt_date(g) == today_mt]
        _after_mt = len(games)
        games = [g for g in games if _game_is_playable_for_prediction(g)]
        _skipped_non_playable = _after_mt - len(games)
        if _skipped_non_playable > 0:
            print(
                f"[SLATE] Skipped {_skipped_non_playable} game(s) not in pregame status allowlist "
                f"(Scheduled/Pre-Game/Warmup/Preview only)"
            )

        print(f"✅ Found {len(games)} games for {today_mt} MT")
        for g in games:
            print("•", f"{g['away_name']} @ {g['home_name']}")

        parlay_usable_non_manual_totals = _preflight_count_games_with_non_manual_real_totals(games, odds_map)
        print(f"[ODDS] Parlay usable non-manual scheduled totals: {parlay_usable_non_manual_totals}/{len(games)}")
        if games and parlay_usable_non_manual_totals == 0:
            if not _main_the_odds_api_fallback_allowed():
                print("[ODDS] STOP: Parlay returned 0 usable scheduled totals after retry.")
                print(
                    "[ODDS] The Odds API main fallback disabled by "
                    "OVERGANG_ALLOW_THE_ODDS_API_MAIN_FALLBACK."
                )
                print(
                    "[ODDS] Engine stopped to avoid synthetic fallback totals and "
                    "protect The Odds API credits."
                )
                return
            print(
                "[ODDS] The Odds API main fallback explicitly enabled via env flag; proceeding."
            )
            toa_fallback_map = fetch_full_game_odds_map() or {}
            print(f"[ODDS] The Odds API fallback rows: {len(toa_fallback_map)}")
            toa_trusted_total_source_map = {}
            for k, v in toa_fallback_map.items():
                if _is_trusted_total_row(v):
                    toa_trusted_total_source_map[k] = dict(v)
                    toa_trusted_total_source_map[k]["_trusted_source"] = "the_odds_api"
            toa_trusted_non_manual_totals = _preflight_count_games_with_non_manual_real_totals(
                games, toa_trusted_total_source_map
            )
            print(
                "[ODDS] The Odds API fallback trusted scheduled totals: "
                f"{toa_trusted_non_manual_totals}/{len(games)} "
                f"(trusted keys={len(toa_trusted_total_source_map)})"
            )
            if toa_fallback_map and toa_trusted_non_manual_totals > 0:
                odds_map = dict(toa_fallback_map)
                trusted_total_source_map = toa_trusted_total_source_map
                odds_source = "the_odds_api_fallback"
                print("[ODDS] Using The Odds API full-game odds fallback for live odds_map")
            else:
                print("[ODDS] The Odds API fallback had no trusted scheduled totals; preserving Parlay/projection-only path")

        # Trusted total lane visibility (per slate game)
        for g in games:
            away_norm = normalize_team_name(safe_get(g, "away_name", ""))
            home_norm = normalize_team_name(safe_get(g, "home_name", ""))
            game_key = f"{away_norm} @ {home_norm}"
            trow = _trusted_total_source_row_for_game(g, trusted_total_source_map)
            if trow is not None and trow != {}:
                t_source = trow.get("_trusted_source", "trusted")
                t_total = trow.get("_raw_total_line")
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

    target_game_ids_raw = str(os.environ.get("OVERGANG_TARGET_GAME_IDS") or "").strip()
    if target_game_ids_raw:
        target_game_ids = {
            part.strip()
            for part in target_game_ids_raw.split(",")
            if part.strip()
        }

        def _game_id_for_target_filter(_game):
            if not isinstance(_game, dict):
                return ""
            for _key in ("game_id", "gamePk", "game_pk", "Game_ID"):
                _value = _game.get(_key)
                if _value is not None and str(_value).strip():
                    return str(_value).strip()
            return ""

        before_target_filter = len(games) if games is not None else 0
        games = [
            game
            for game in (games or [])
            if _game_id_for_target_filter(game) in target_game_ids
        ]
        print(
            f"[TARGETED_RUN] Filtering games by OVERGANG_TARGET_GAME_IDS={target_game_ids_raw} "
            f"kept {len(games)}/{before_target_filter}"
        )

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

        away_recovered = (
            _recover_missing_probable_pitcher_from_game_feed(game, "away")
            if away_missing
            else ""
        )
        home_recovered = (
            _recover_missing_probable_pitcher_from_game_feed(game, "home")
            if home_missing
            else ""
        )

        away_used = away_recovered or ("League Avg Away" if away_missing else str(away_raw).strip())
        home_used = home_recovered or ("League Avg Home" if home_missing else str(home_raw).strip())

        if away_missing or home_missing:
            if away_recovered:
                print(f"✅ Opening audit recovered away probable: {away_team} → {away_recovered}")
            if home_recovered:
                print(f"✅ Opening audit recovered home probable: {home_team} → {home_recovered}")

            if (away_missing and not away_recovered) or (home_missing and not home_recovered):
                opening_games_missing_probable_pitchers += 1
                if away_missing and not away_recovered:
                    opening_missing_pitchers_by_name.add("League Avg Away")
                if home_missing and not home_recovered:
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

        trow = _trusted_total_source_row_for_game(game, trusted_total_source_map)
        trusted_exists = isinstance(trow, dict)
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

    reliever_df = load_reliever_stats()

    try:
        f5_totals_by_game = fetch_f5_totals_by_game()
    except Exception as _e_f5:
        print(f"⚠️ the_odds_api F5 totals fetch failed: {_e_f5}")
        f5_totals_by_game = {}
    print(f"📈 the_odds_api F5 totals loaded for {len(f5_totals_by_game)} game(s)")

    try:
        toa_ou_totals_by_game = fetch_ou_totals_by_game()
    except Exception as _e_toa_ou:
        print(f"⚠️ the_odds_api OU totals fetch failed: {_e_toa_ou}")
        toa_ou_totals_by_game = {}
    print(f"📈 the_odds_api OU totals loaded for {len(toa_ou_totals_by_game)} game(s)")

    try:
        ou_sharp_totals_by_game = fetch_ou_sharp_totals()
    except Exception as _e_ou_sharp:
        print(f"⚠️ targeted Parlay OU sharp totals fetch failed: {_e_ou_sharp}")
        ou_sharp_totals_by_game = {}
    print(f"📈 targeted Parlay OU sharp totals loaded for {len(ou_sharp_totals_by_game)} game(s)")

    for game in games:
        try:
            home_team = safe_get(game, 'home_name', 'Home Team')
            away_team = safe_get(game, 'away_name', 'Away Team')
            away_reliever_metrics = get_reliever_depth_metrics(away_team, reliever_df)
            home_reliever_metrics = get_reliever_depth_metrics(home_team, reliever_df)
            vegas_line, odds_info = VegasLines.get_vegas_line(
                home_team,
                away_team,
                odds_map,
                game_datetime=safe_get(game, "game_datetime", ""),
            )
            print(f"[ODDS] Game: {away_team} @ {home_team}")
            print(f"[ODDS]   Lookup key: {odds_info.get('_lookup_key', '?')}")
            print(f"[ODDS]   Match found in odds_map: {odds_info.get('_match_found', '?')}")
            print(f"[ODDS]   odds_info: total_line={odds_info.get('total_line')}, over_juice={odds_info.get('over_juice')}, under_juice={odds_info.get('under_juice')}, book={repr(odds_info.get('book'))}")
            print(f"[ODDS]   Source: {odds_info.get('_source', '?')}")
            print(f"[ODDS]   Total status: {'REAL sportsbook total' if odds_info.get('_has_real_total', False) else 'FALLBACK total (missing market totals)'}")

            # LIVE TOTAL BLOCKERS tracking: only count odds_map/API rows (exclude manual_totals_csv + no-match default fallback)
            _src = str(odds_info.get("_source", ""))
            if odds_info.get("_match_found", False) and _src in {"parlay_api", "8.5 fallback", "fallback (scrambled book)"}:
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

            # ✅ Recover blank/TBD probable starters before using League Avg shells.
            if not away_pitcher.strip() or away_pitcher == "TBD":
                _recovered_away_pitcher = _recover_missing_probable_pitcher_from_game_feed(game, "away")
                if _recovered_away_pitcher:
                    print(
                        f"✅ Recovered missing away pitcher for {away_team}: "
                        f"{_recovered_away_pitcher}"
                    )
                    away_pitcher = _recovered_away_pitcher
                else:
                    print(f"⚠️ Missing away pitcher for {away_team} — using League Avg Away")
                    away_pitcher = "League Avg Away"
            if not home_pitcher.strip() or home_pitcher == "TBD":
                _recovered_home_pitcher = _recover_missing_probable_pitcher_from_game_feed(game, "home")
                if _recovered_home_pitcher:
                    print(
                        f"✅ Recovered missing home pitcher for {home_team}: "
                        f"{_recovered_home_pitcher}"
                    )
                    home_pitcher = _recovered_home_pitcher
                else:
                    print(f"⚠️ Missing home pitcher for {home_team} — using League Avg Home")
                    home_pitcher = "League Avg Home"

            print(f"🎯 Matchup: {away_team} ({away_pitcher}) vs {home_team} ({home_pitcher})")

            venue = game.get("venue_name", "Unknown")
            park_factors = PARK_FACTORS.get(venue, PARK_FACTORS['Unknown'])
            weather_runs_mult = compute_weather_runs_mult(venue, safe_get(game, "game_datetime", ""), game_pk=game.get("game_id") or game.get("gamePk"))
            if abs(weather_runs_mult - 1.0) > 0.0005:
                print(f"🌤️ Weather overlay: runs_mult={weather_runs_mult:.4f} ({venue})")

            print(f"\n🔍 Processing: {away_team} @ {home_team}")
            print(f"🧪 Matching: Away = {away_pitcher} | Home = {home_pitcher}")

            away_impact = 0.0
            home_impact = 0.0
            away_scope = "none"
            home_scope = "none"
            lineup_delta = 0.0
            effective_lineup_delta = 0.0

            # --- score lineups vs the real opposing starter hand ---
            # Morning/default path remains the team-specific top-PA proxy.
            # Confirmed MLB lineups replace it only when BOTH teams have
            # complete 1-9 orders and BOTH score successfully at 9/9.
            lineup_game_pk = (
                game.get("game_id")
                or game.get("gamePk")
                or game.get("game_pk")
                or game.get("Game_ID")
            )

            lineup_source_away = "top_pa_proxy"
            lineup_source_home = "top_pa_proxy"
            lineup_confirmed_away = False
            lineup_confirmed_home = False
            lineup_player_count_away = 0
            lineup_player_count_home = 0
            lineup_matched_count_away = ""
            lineup_matched_count_home = ""
            lineup_order_away = ""
            lineup_order_home = ""
            lineup_signature_away = ""
            lineup_signature_home = ""
            lineup_fetched_at = ""
            lineup_feed_status = ""
            lineup_fetch_error = ""

            # Determine the hand each offense will face.
            try:
                home_starter_hand = Batters.get_pitcher_hand(
                    home_pitcher
                )
            except Exception:
                home_starter_hand = None

            try:
                away_starter_hand = Batters.get_pitcher_hand(
                    away_pitcher
                )
            except Exception:
                away_starter_hand = None

            # Canonical safe fallback: existing team-specific top-PA proxy.
            try:
                away_best9 = lineups.get_team_best9(
                    away_team.lower()
                )
                home_best9 = lineups.get_team_best9(
                    home_team.lower()
                )

                lineup_player_count_away = len(away_best9)
                lineup_player_count_home = len(home_best9)

                (
                    away_impact,
                    home_impact,
                    away_scope,
                    home_scope,
                ) = safe_lineup_impacts(
                    lineup_obj=lineups,
                    away_lineup=away_best9,
                    home_lineup=home_best9,
                    away_pitcher_hand=away_starter_hand,
                    home_pitcher_hand=home_starter_hand,
                    logger=None,
                )
            except Exception as e:
                print(f"⚠️ Proxy lineup impact error: {e}")
                away_impact = 0.0
                home_impact = 0.0
                away_scope = "none"
                home_scope = "none"

            # Upgrade both sides together only when the MLB feed and the
            # ordered player-ID scorer both pass completely.
            try:
                confirmed = fetch_confirmed_lineups(
                    lineup_game_pk
                )

                lineup_fetched_at = str(
                    confirmed.get("fetched_at_utc") or ""
                )
                lineup_feed_status = str(
                    confirmed.get("status") or ""
                )
                lineup_fetch_error = str(
                    confirmed.get("error") or ""
                )

                if bool(confirmed.get("both_confirmed")):
                    away_ordered = (
                        lineups.score_ordered_lineup_dict(
                            confirmed.get("away_lineup") or [],
                            pitcher_hand=(
                                home_starter_hand or "R"
                            ),
                        )
                    )
                    home_ordered = (
                        lineups.score_ordered_lineup_dict(
                            confirmed.get("home_lineup") or [],
                            pitcher_hand=(
                                away_starter_hand or "R"
                            ),
                        )
                    )

                    ordered_scoring_ok = (
                        away_ordered.get("scope")
                        == "mlb_confirmed_ordered"
                        and home_ordered.get("scope")
                        == "mlb_confirmed_ordered"
                        and away_ordered.get("matched_count") == 9
                        and home_ordered.get("matched_count") == 9
                    )

                    if ordered_scoring_ok:
                        away_impact = float(
                            away_ordered["lineup_impact"]
                        )
                        home_impact = float(
                            home_ordered["lineup_impact"]
                        )
                        away_scope = str(
                            away_ordered["scope"]
                        )
                        home_scope = str(
                            home_ordered["scope"]
                        )

                        lineup_source_away = "mlb_confirmed"
                        lineup_source_home = "mlb_confirmed"
                        lineup_confirmed_away = True
                        lineup_confirmed_home = True

                        lineup_player_count_away = 9
                        lineup_player_count_home = 9
                        lineup_matched_count_away = 9
                        lineup_matched_count_home = 9

                        away_records = (
                            confirmed.get("away_lineup") or []
                        )
                        home_records = (
                            confirmed.get("home_lineup") or []
                        )

                        lineup_order_away = "|".join(
                            f"{p['slot']}:{p['name']}"
                            for p in away_records
                        )
                        lineup_order_home = "|".join(
                            f"{p['slot']}:{p['name']}"
                            for p in home_records
                        )

                        lineup_signature_away = str(
                            confirmed.get(
                                "away_signature"
                            ) or ""
                        )
                        lineup_signature_home = str(
                            confirmed.get(
                                "home_signature"
                            ) or ""
                        )
                    else:
                        print(
                            "⚠️ Confirmed MLB lineups were posted, "
                            "but ordered scoring did not pass 9/9 "
                            f"for {away_team} @ {home_team}; "
                            "using top-PA proxies for both teams"
                        )
            except Exception as e:
                lineup_fetch_error = (
                    f"integration_failed:{type(e).__name__}"
                )
                print(
                    "⚠️ Confirmed lineup integration error for "
                    f"{away_team} @ {home_team}: {e}; "
                    "using top-PA proxies"
                )

            # Positive means home lineup projects stronger than away.
            lineup_delta = (
                float(home_impact) - float(away_impact)
            )
            _eh = max(
                -LINEUP_IMPACT_CAP,
                min(
                    LINEUP_IMPACT_CAP,
                    float(home_impact),
                ),
            )
            _ea = max(
                -LINEUP_IMPACT_CAP,
                min(
                    LINEUP_IMPACT_CAP,
                    float(away_impact),
                ),
            )
            effective_lineup_delta = _eh - _ea

            # Existing controlled total adjustment remains unchanged.
            vegas_line_adj = (
                float(vegas_line)
                + 0.3 * lineup_delta
            )

            print(
                f"🧮 Lineup impacts → AWAY {away_team}: "
                f"{away_impact:.3f} ({away_scope}; "
                f"source={lineup_source_away}) | "
                f"HOME {home_team}: "
                f"{home_impact:.3f} ({home_scope}; "
                f"source={lineup_source_home}) | "
                f"Δ={lineup_delta:+.3f}"
            )
            print(
                f"   sizing Δ (±{LINEUP_IMPACT_CAP} cap, "
                "same scale as project_team_runs): "
                f"{effective_lineup_delta:+.3f}"
            )

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

            # 🧠 Log low-IP starter sample separately from true fallback.
            # LowIP means real pitcher stats exist but sample size is thin.
            for name, stats in [(away_pitcher, away_stats), (home_pitcher, home_stats)]:
                if isinstance(stats, dict) and stats.get("LowIP", False):
                    print(f"⚠️ Low-IP pitcher sample for: {name}")

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

            def _projection_lineup_delta_vs_team(lineup_impact, offense_mult, scope):
                """
                Convert the best9/lineup split signal into a team-relative delta
                before project_team_runs applies it.

                Previous stack was:
                    team_offense_mult * (1 + absolute_lineup_impact)

                That double-counted or muted lineups because lineup_impact is
                already an absolute strength vs hand. The projection path should
                land on the lineup/best9 strength when a lineup proxy is matched,
                while preserving the team-scope fallback when no lineup exists.
                """
                try:
                    raw_impact = float(lineup_impact)
                    team_mult = float(offense_mult)
                except (TypeError, ValueError):
                    return 0.0

                if not np.isfinite(raw_impact) or not np.isfinite(team_mult) or team_mult <= 0:
                    return 0.0

                if str(scope or "").strip().lower() != "lineup":
                    return max(-LINEUP_IMPACT_CAP, min(LINEUP_IMPACT_CAP, raw_impact))

                lineup_strength = 1.0 + raw_impact
                if lineup_strength <= 0 or not np.isfinite(lineup_strength):
                    return 0.0

                delta = (lineup_strength / team_mult) - 1.0
                return max(-LINEUP_IMPACT_CAP, min(LINEUP_IMPACT_CAP, delta))

            away_projection_lineup_impact = _projection_lineup_delta_vs_team(
                away_impact, away_offense_mult, away_scope
            )
            home_projection_lineup_impact = _projection_lineup_delta_vs_team(
                home_impact, home_offense_mult, home_scope
            )

            print(
                f"🧮 Projection lineup delta vs team → AWAY {away_team}: "
                f"raw={float(away_impact):+.3f} → proj={away_projection_lineup_impact:+.3f}; "
                f"HOME {home_team}: raw={float(home_impact):+.3f} → proj={home_projection_lineup_impact:+.3f}"
            )

            # Low-IP pitchers are real matched pitchers, not unmatched pitchers.
            # Keep them out of unmatched alerts; they remain tracked via Starter_LowIP
            # and Data_Quality_Flag=low_ip.

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
                'Game_ID': game.get('game_id') or game.get('gamePk') or '',
                'Game_Num': game.get('game_num') or game.get('gameNumber') or '',
                'Doubleheader': game.get('doubleheader') or game.get('doubleHeader') or '',
                'Datetime': safe_get(game, 'game_datetime', datetime.utcnow().isoformat()),
                'Game_Date': game.get('game_date') or '',
                'Game': game_name,
                'Game_Status': _game_status_for_export(game),
                'Venue': venue,
                'Weather_Runs_Mult': weather_runs_mult,
                'Retractable_Roof_Weather': venue in RETRACTABLE_ROOF_VENUES,
                'Pitchers': f"{away_pitcher} vs {home_pitcher}",
                'vegas_line': vegas_line,
                'Total_Is_Real': bool(odds_info.get('_has_real_total', False)),
                'Odds_Line': odds_info.get('total_line', 8.5) if odds_info.get('_has_real_total', False) else '',
                'Over_Juice': odds_info.get('over_juice', -110),
                'Under_Juice': odds_info.get('under_juice', -110),
                'Odds_Book': odds_info.get('book', ''),
                'ML_Odds_Book': odds_info.get('ml_book', ''),
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
                'Projection_Cap_Flag': False,
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
                'OU_Edge': '',
                'OU_Confidence': '',
                'OU_Side': '',
                'OU_Bet_Type': 'total',
                'ML_Edge': '',
                'ML_Side': '',
                'ML_Bet_Type': 'moneyline',
                'ML_Market_OK': False,
                'ML_Market_Status': '',
            }

            # 🔮 Run prediction (compare projection to actual Vegas line; do not pass lineup-adjusted line)
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
                away_lineup_impact=float(away_projection_lineup_impact),
                home_lineup_impact=float(home_projection_lineup_impact),
                away_offense_mult=away_offense_mult,
                home_offense_mult=home_offense_mult,
                has_real_total=bool(odds_info.get("_has_real_total", False)),
                weather_runs_mult=weather_runs_mult,
                away_pitcher_name=away_pitcher,
                home_pitcher_name=home_pitcher,
                game_datetime=safe_get(game, "game_datetime", ""),
                schedule_game_date=safe_get(game, "game_date", ""),
                odds_info=odds_info,
                away_reliever_metrics=away_reliever_metrics,
                home_reliever_metrics=home_reliever_metrics,
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
            away_runs_raw = proj["away_runs_raw"]
            home_runs_raw = proj["home_runs_raw"]
            away_runs_safety = proj["away_runs_safety"]
            home_runs_safety = proj["home_runs_safety"]
            edge = proj["edge"]
            recommended_units = proj["recommended_units"]
            projection_cap_hit = bool(proj.get("projection_cap_hit", False))
            ou_pick = proj["pick"]
            ou_sharp_layer_gap = None  # Pinnacle − ESPN DK total (sharpness layer); see SHARP_OU_Delta export

            # Offense strength already in projection via away_offense_mult / home_offense_mult; no post-hoc bat_mult

            has_real_total = bool(odds_info.get("_has_real_total", False))
            ou_market_quality_note = ""
            if has_real_total:
                try:
                    _pinn_total = odds_info.get("pinnacle_total_line")
                    _retail_total = odds_info.get("espn_draftkings_total_line")
                    if _retail_total is None:
                        _retail_total = odds_info.get("draftkings_total_line")
                    if _pinn_total is not None and _retail_total is not None:
                        _pinn_total = float(_pinn_total)
                        _espn_dk_total = float(_retail_total)
                        if (5 <= _pinn_total <= 15) and (5 <= _espn_dk_total <= 15):
                            _sharp_gap = round(_pinn_total - _espn_dk_total, 2)
                            ou_sharp_layer_gap = _sharp_gap
                            _abs_sharp_gap = abs(_sharp_gap)
                            if _abs_sharp_gap >= 1.0:
                                _align_boost = 0.04
                                _conflict_trim = 0.02
                            elif _abs_sharp_gap >= 0.5:
                                _align_boost = 0.02
                                _conflict_trim = 0.01
                            else:
                                _align_boost = 0.0
                                _conflict_trim = 0.0
                            if _align_boost > 0:
                                _sharp_dir = "OVER" if _sharp_gap > 0 else "UNDER"
                                _pick_aligns = (
                                    (_sharp_dir == "OVER" and ou_pick in ("OVER", "LEAN_OVER"))
                                    or (_sharp_dir == "UNDER" and ou_pick in ("UNDER", "LEAN_UNDER"))
                                )
                                if _pick_aligns:
                                    confidence += _align_boost
                                    ou_market_quality_note = (
                                        f"ou_sharp=+{_align_boost:.2f}@{_abs_sharp_gap:.1f}"
                                    )
                                else:
                                    confidence -= _conflict_trim
                                    ou_market_quality_note = (
                                        f"ou_sharp=-{_conflict_trim:.2f}@{_abs_sharp_gap:.1f}"
                                    )
                except (TypeError, ValueError):
                    pass
            confidence = max(0.0, min(1.0, confidence))
            if not has_real_total:
                prediction = "NO BET (fallback total only)"

            # Optional: scale units by lineup conviction (aligned to LINEUP_IMPACT_CAP, not raw rails)
            try:
                units_mult = 1.0 + 0.25 * abs(float(effective_lineup_delta))
                units_mult = min(units_mult, 1.50)
                sized_units = round(recommended_units * units_mult, 2)
            except Exception:
                sized_units = recommended_units

            f5_projected_total = proj.get("f5_projected_total", "")
            f5_away_runs = proj.get("f5_away_runs", "")
            f5_home_runs = proj.get("f5_home_runs", "")
            f5_model_side = proj.get("f5_model_side", "")
            f5_starter_lean = proj.get("f5_starter_lean", "")
            f5_eligible = proj.get("f5_eligible", "")
            f5_market_line = proj.get("f5_market_line", "")
            f5_no_line_reason = proj.get("f5_no_line_reason", "")

            _f5_lookup_key = f"{normalize_team_name(away_team)} @ {normalize_team_name(home_team)}"
            _f5_market = f5_totals_by_game.get(_f5_lookup_key) if isinstance(f5_totals_by_game, dict) else None
            if _f5_market is not None:
                f5_market_line = _f5_market.get("f5_total_line", "")
                f5_over_juice = _f5_market.get("f5_over_juice", "")
                f5_under_juice = _f5_market.get("f5_under_juice", "")
                f5_source = _f5_market.get("f5_source", "")
                f5_book = _f5_market.get("f5_book", "")
                f5_market_ok = bool(_f5_market.get("f5_market_ok", False))
                f5_no_line_reason = ""
            else:
                f5_over_juice = ""
                f5_under_juice = ""
                f5_source = ""
                f5_book = ""
                f5_market_ok = False
                f5_no_line_reason = "no_f5_market_source"

            game_data.update({
                "Projected_Total": projected_total,
                "Base_Projected_Total": proj.get("base_projected_total", ""),
                "Base_OU_Edge": proj.get("base_ou_edge", ""),
                "Raw_Projected_Total": proj.get("raw_projected_total", ""),
                "Raw_OU_Edge": proj.get("raw_ou_edge", ""),
                "OU_Edge_Calibration_Factor": proj.get("ou_edge_calibration_factor", ""),
                "OU_Edge_Calibration_Tag": proj.get("ou_edge_calibration_tag", ""),
                "Away_Early_Run_Pressure": proj.get("away_early_run_pressure", ""),
                "Away_Early_Run_Suppression": proj.get("away_early_run_suppression", ""),
                "Away_Late_Run_Pressure": proj.get("away_late_run_pressure", ""),
                "Away_Late_Run_Suppression": proj.get("away_late_run_suppression", ""),
                "Home_Early_Run_Pressure": proj.get("home_early_run_pressure", ""),
                "Home_Early_Run_Suppression": proj.get("home_early_run_suppression", ""),
                "Home_Late_Run_Pressure": proj.get("home_late_run_pressure", ""),
                "Home_Late_Run_Suppression": proj.get("home_late_run_suppression", ""),
                "Weather_Interaction_Adjustment": proj.get("weather_interaction_adjustment", ""),
                "Full_Picture_Adjustment_Raw": proj.get("full_picture_adjustment_raw", ""),
                "Away_Run_Pressure_Adjustment": proj.get("away_run_pressure_adjustment", ""),
                "Home_Run_Pressure_Adjustment": proj.get("home_run_pressure_adjustment", ""),
                "Run_Pressure_Adjustment": proj.get("run_pressure_adjustment", ""),
                "Run_Pressure_Mode": proj.get("run_pressure_mode", ""),
                "Run_Pressure_Reasons": proj.get("run_pressure_reasons", ""),
                "Away_Expected_Starter_IP": proj.get("telemetry", {}).get("away_expected_starter_ip", ""),
                "Home_Expected_Starter_IP": proj.get("telemetry", {}).get("home_expected_starter_ip", ""),
                "Away_Starter_IP_Share": proj.get("telemetry", {}).get("away_starter_ip_share", ""),
                "Home_Starter_IP_Share": proj.get("telemetry", {}).get("home_starter_ip_share", ""),
                "Away_Bullpen_IP_Share": proj.get("telemetry", {}).get("away_bullpen_ip_share", ""),
                "Home_Bullpen_IP_Share": proj.get("telemetry", {}).get("home_bullpen_ip_share", ""),
                "Away_Workload_Profile": proj.get("telemetry", {}).get("away_workload_profile", ""),
                "Home_Workload_Profile": proj.get("telemetry", {}).get("home_workload_profile", ""),
                "Away_Workload_Source": proj.get("telemetry", {}).get("away_workload_source", ""),
                "Home_Workload_Source": proj.get("telemetry", {}).get("home_workload_source", ""),
                "Away_Runs": away_runs,
                "Home_Runs": home_runs,
                "Away_Runs_Raw": away_runs_raw,
                "Home_Runs_Raw": home_runs_raw,
                "Away_Runs_Safety": away_runs_safety,
                "Home_Runs_Safety": home_runs_safety,
                "Away_Cap_Diff": round(float(away_runs) - float(away_runs_safety), 2),
                "Home_Cap_Diff": round(float(home_runs) - float(home_runs_safety), 2),
                "Projection_Cap_Flag": projection_cap_hit,
                "F5_Projected_Total": f5_projected_total,
                "F5_Away_Runs": f5_away_runs,
                "F5_Home_Runs": f5_home_runs,
                "F5_Model_Side": f5_model_side,
                "F5_Starter_Lean": f5_starter_lean,
                "F5_Eligible": f5_eligible,
                "F5_Market_Line": f5_market_line,
                "F5_Over_Juice": f5_over_juice,
                "F5_Under_Juice": f5_under_juice,
                "F5_Source": f5_source,
                "F5_Book": f5_book,
                "F5_Market_OK": f5_market_ok,
                "F5_No_Line_Reason": f5_no_line_reason,
                "Lineup_Impact_Away": round(float(away_impact), 4),
                "Lineup_Impact_Home": round(float(home_impact), 4),
                "Lineup_Delta_Raw": round(float(lineup_delta), 4),
                "Lineup_Delta_Effective": round(float(effective_lineup_delta), 4),
                "Lineup_Mode_Away": (away_scope if away_scope and str(away_scope).lower() != "none" else ""),
                "Lineup_Mode_Home": (home_scope if home_scope and str(home_scope).lower() != "none" else ""),
                "Lineup_Source_Away": lineup_source_away,
                "Lineup_Source_Home": lineup_source_home,
                "Lineup_Confirmed_Away": lineup_confirmed_away,
                "Lineup_Confirmed_Home": lineup_confirmed_home,
                "Lineup_Player_Count_Away": lineup_player_count_away,
                "Lineup_Player_Count_Home": lineup_player_count_home,
                "Lineup_Matched_Count_Away": lineup_matched_count_away,
                "Lineup_Matched_Count_Home": lineup_matched_count_home,
                "Lineup_Order_Away": lineup_order_away,
                "Lineup_Order_Home": lineup_order_home,
                "Lineup_Signature_Away": lineup_signature_away,
                "Lineup_Signature_Home": lineup_signature_home,
                "Lineup_Fetched_At": lineup_fetched_at,
                "Lineup_Feed_Status": lineup_feed_status,
                "Lineup_Fetch_Error": lineup_fetch_error,
                "Lineup_Cap_Hit_Away": bool(abs(float(away_impact)) >= 0.20),
                "Lineup_Cap_Hit_Home": bool(abs(float(home_impact)) >= 0.20),
                "Vegas_Line": vegas_line,
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

            # Step 1: global compression → Step 2: quality penalty → Step 3: hard cap (max_ml_cap).
            ml_quality_penalty = 1.0
            if "League Avg" in (away_pitcher or "") or "League Avg" in (home_pitcher or ""):
                ml_quality_penalty *= 0.85
            _ml_away_low_ip = bool(safe_get(away_stats, "LowIP", False))
            _ml_home_low_ip = bool(safe_get(home_stats, "LowIP", False))
            if _ml_away_low_ip or _ml_home_low_ip:
                ml_quality_penalty *= 0.90
            compressed_home = 0.5 + (home_win_prob - 0.5) * ml_compression_k
            compressed_away = 0.5 + (away_win_prob - 0.5) * ml_compression_k
            quality_home = 0.5 + (compressed_home - 0.5) * ml_quality_penalty
            quality_away = 0.5 + (compressed_away - 0.5) * ml_quality_penalty
            adjusted_home_win_prob = min(max_ml_cap, quality_home)
            adjusted_away_win_prob = min(max_ml_cap, quality_away)

            # RECOVERY PATCH: populate per-book ML sharpness fields from a
            # list-format Parlay payload when odds_api did not flatten them.
            pinn_home = pinn_away = novig_home = novig_away = None
            _home_norm_ml = normalize_team_name(home_team)
            _away_norm_ml = normalize_team_name(away_team)
            for bm in odds_info.get("bookmakers", []):
                try:
                    bm_key = (bm.get("key") or "").lower()
                    if bm_key not in ("pinnacle", "novig"):
                        continue
                    for market in bm.get("markets", []):
                        if market.get("key") != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            outcome_norm = normalize_team_name(outcome.get("name", ""))
                            price = outcome.get("price")
                            if price is None:
                                continue
                            if bm_key == "pinnacle" and outcome_norm == _home_norm_ml:
                                pinn_home = price
                            elif bm_key == "pinnacle" and outcome_norm == _away_norm_ml:
                                pinn_away = price
                            elif bm_key == "novig" and outcome_norm == _home_norm_ml:
                                novig_home = price
                            elif bm_key == "novig" and outcome_norm == _away_norm_ml:
                                novig_away = price
                except (AttributeError, TypeError):
                    continue
            odds_info.update({
                "pinnacle_ml_home": odds_info.get("pinnacle_ml_home", pinn_home) if pinn_home is None else pinn_home,
                "pinnacle_ml_away": odds_info.get("pinnacle_ml_away", pinn_away) if pinn_away is None else pinn_away,
                "novig_ml_home": odds_info.get("novig_ml_home", novig_home) if novig_home is None else novig_home,
                "novig_ml_away": odds_info.get("novig_ml_away", novig_away) if novig_away is None else novig_away,
            })

            # Book ML lines from odds_map (both sides required for de-vig; no synthetic defaults).
            implied_home, implied_away, ml_market_status = _ml_pair_devig_implied(
                odds_info.get("ml_home"), odds_info.get("ml_away")
            )
            if ml_market_status == "missing_market" and (
                odds_info.get("ml_home") is not None or odds_info.get("ml_away") is not None
            ):
                ml_market_status = "partial"
            ml_market_ok = ml_market_status == "ok"

            # Parallel ML sharpness / market-truth gate (observational + blocking).
            # Uses additive per-book fields emitted by core/odds_api.py:
            #   novig_ml_home/away, pinnacle_ml_home/away, exchange_present, pinnacle_present, prophetx_present.
            # Never alters ml_conf, adjusted_*_win_prob, Kelly, or Telegram confidence gate.
            _novig_ih, _novig_ia, _novig_status = _ml_pair_devig_implied(
                odds_info.get("novig_ml_home"), odds_info.get("novig_ml_away")
            )
            _pinn_ih, _pinn_ia, _pinn_status = _ml_pair_devig_implied(
                odds_info.get("pinnacle_ml_home"), odds_info.get("pinnacle_ml_away")
            )
            exchange_present_flag = bool(odds_info.get("exchange_present", False))
            prophetx_present_flag = bool(odds_info.get("prophetx_present", False))
            pinnacle_present_flag = bool(odds_info.get("pinnacle_present", False))

            if ml_market_ok:
                home_kelly = calculate_kelly_units(adjusted_home_win_prob, implied_home)
                away_kelly = calculate_kelly_units(adjusted_away_win_prob, implied_away)
            else:
                home_kelly = 0.0
                away_kelly = 0.0

            if adjusted_home_win_prob > adjusted_away_win_prob:
                ml_pick = f"{home_team.upper()} ML"
                ml_conf = f"{round(adjusted_home_win_prob * 100)}%"
                ml_side = "home"
                if ml_market_ok:
                    ml_value = f"{round((adjusted_home_win_prob - implied_home) * 100)}%"
                    ml_edge_num = float(adjusted_home_win_prob) - float(implied_home)
                else:
                    ml_value = ""
                    ml_edge_num = None
                ml_kelly = f"{round(home_kelly, 2)}u"
            else:
                ml_pick = f"{away_team.upper()} ML"
                ml_conf = f"{round(adjusted_away_win_prob * 100)}%"
                ml_side = "away"
                if ml_market_ok:
                    ml_value = f"{round((adjusted_away_win_prob - implied_away) * 100)}%"
                    ml_edge_num = float(adjusted_away_win_prob) - float(implied_away)
                else:
                    ml_value = ""
                    ml_edge_num = None
                ml_kelly = f"{round(away_kelly, 2)}u"

            _league_avg_pitcher = (
                "League Avg" in (away_pitcher or "") or "League Avg" in (home_pitcher or "")
            )
            ml_win_max = max(adjusted_home_win_prob, adjusted_away_win_prob)
            ml_fire_eligible = (
                ml_market_ok
                and (not _league_avg_pitcher)
                and (ml_quality_penalty >= MIN_ML_QUALITY_PENALTY_FOR_FIRE)
            )
            ml_edge_meets_min = (
                ml_edge_num is not None
                and float(ml_edge_num) >= float(MIN_ML_EDGE_FOR_FIRE)
            )

            # ML sharpness parallel gate: observed on picked side only. Fail-open when
            # either novig or pinnacle de-vigged implied is unavailable for that side.
            # Never modifies ml_conf, adjusted probs, ml_edge_num, or Kelly.
            _picked_novig_implied = (
                _novig_ih if ml_side == "home" else _novig_ia
            )
            _picked_pinn_implied = (
                _pinn_ih if ml_side == "home" else _pinn_ia
            )
            if _picked_novig_implied is None:
                _picked_novig_price = (
                    odds_info.get("novig_ml_home")
                    if ml_side == "home"
                    else odds_info.get("novig_ml_away")
                )
                _picked_novig_implied = _american_odds_to_implied(_picked_novig_price)
            if _picked_pinn_implied is None:
                _picked_pinn_price = (
                    odds_info.get("pinnacle_ml_home")
                    if ml_side == "home"
                    else odds_info.get("pinnacle_ml_away")
                )
                _picked_pinn_implied = _american_odds_to_implied(_picked_pinn_price)
            if (_picked_novig_implied is not None) and (_picked_pinn_implied is not None):
                ml_exchange_vs_sharp_gap = float(
                    _picked_novig_implied
                ) - float(_picked_pinn_implied)
                ml_sharpness_inputs_ok = True
            else:
                ml_exchange_vs_sharp_gap = None
                ml_sharpness_inputs_ok = False
            if ml_sharpness_inputs_ok:
                ml_sharpness_ok = abs(ml_exchange_vs_sharp_gap) <= float(
                    MAX_EXCHANGE_VS_SHARP_GAP
                )
                ml_sharpness_gate_open = ml_sharpness_ok
            else:
                ml_sharpness_ok = False
                ml_sharpness_gate_open = True  # fail-open: gate cannot block without both inputs

            ml_fired = bool(
                ml_fire_eligible
                and (ml_win_max >= MIN_ML_WIN_PROB_FOR_FIRE)
                and ml_edge_meets_min
                and ml_sharpness_gate_open
            )
            ml_quality_flag = "league_avg_pitcher_fallback" if _league_avg_pitcher else ""

            if ml_fired:
                no_fire_ml = ""
            elif not ml_market_ok:
                no_fire_ml = "ml_market_missing_or_incomplete"
            elif _league_avg_pitcher:
                no_fire_ml = "ml_quality_league_avg_pitcher"
            elif ml_quality_penalty < MIN_ML_QUALITY_PENALTY_FOR_FIRE:
                no_fire_ml = "ml_quality_low"
            elif ml_win_max < MIN_ML_WIN_PROB_FOR_FIRE:
                no_fire_ml = "ml_win_prob_below_threshold"
            elif not ml_edge_meets_min:
                no_fire_ml = "ml_edge_below_threshold"
            elif not ml_sharpness_gate_open:
                no_fire_ml = "ml_sharpness_gap_exceeded"
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
                "ML_Quality_Factor": ml_quality_penalty,
                "ML_Market_OK": ml_market_ok,
                "ML_Market_Status": ml_market_status,
                "ML_Exchange_Present": exchange_present_flag,
                "ML_Prophetx_Present": prophetx_present_flag,
                "ML_Pinnacle_Present": pinnacle_present_flag,
                "ML_Sharpness_Inputs_OK": ml_sharpness_inputs_ok,
                "ML_Sharpness_OK": ml_sharpness_ok,
                "ML_Sharpness_Gate_Open": ml_sharpness_gate_open,
                "ML_Exchange_Vs_Sharp_Gap": (
                    round(ml_exchange_vs_sharp_gap, 4)
                    if ml_exchange_vs_sharp_gap is not None
                    else ""
                ),
                "ML_Edge": round(ml_edge_num, 2) if ml_edge_num is not None else "",
                "ML_Side": ml_side,
                "ML_Bet_Type": "moneyline",
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
            fire_threshold = 0.79 if is_manual_trusted else 0.79
            standard_ou_fire = (
                (confidence >= fire_threshold)
                and has_real_total
                and (not projection_cap_hit)
                and (abs(edge) >= 1.0)
                and ("League Avg" not in (away_pitcher or ""))
                and ("League Avg" not in (home_pitcher or ""))
            )
            clean_strong_ou = (
                has_real_total
                and (not projection_cap_hit)
                and (abs(edge) >= 1.0)
                and (confidence >= 0.74)
                and ("League Avg" not in (away_pitcher or ""))
                and ("League Avg" not in (home_pitcher or ""))
                and (not away_low)
                and (not home_low)
                and (not _fallback_used_from_path(away_path))
                and (not _fallback_used_from_path(home_path))
            )
            ou_fire_candidate = standard_ou_fire or clean_strong_ou
            _ou_pick = proj.get("pick", "")
            _edge_f = float(edge)
            ou_moderate_over_candidate = (
                _ou_pick == "OVER"
                and has_real_total
                and (not projection_cap_hit)
                and (0.5 <= _edge_f < 1.0)
            )
            ou_tail_over_risk = (
                _ou_pick == "OVER"
                and has_real_total
                and (1.5 <= _edge_f < 2.0)
            )
            try:
                _bl = float(game_data.get("Bet_Line", 99))
            except (TypeError, ValueError):
                _bl = 99.0
            away_reliever_depth_risk = str(
                away_reliever_metrics.get("depth_risk", "")
            ).strip().lower()
            home_reliever_depth_risk = str(
                home_reliever_metrics.get("depth_risk", "")
            ).strip().lower()

            def _ou_lane_float(key, default=0.0):
                try:
                    v = game_data.get(key, default)
                    if v in ("", None):
                        return default
                    f = float(v)
                    if f != f:
                        return default
                    return f
                except (TypeError, ValueError):
                    return default

            # Lane Separation v1:
            # Use the already-computed early/late run path to decide whether the
            # current full-game O/U pick is actually the right market.
            _away_early_pressure = _ou_lane_float("Away_Early_Run_Pressure")
            _away_early_suppression = _ou_lane_float("Away_Early_Run_Suppression")
            _home_early_pressure = _ou_lane_float("Home_Early_Run_Pressure")
            _home_early_suppression = _ou_lane_float("Home_Early_Run_Suppression")
            _away_late_pressure = _ou_lane_float("Away_Late_Run_Pressure")
            _away_late_suppression = _ou_lane_float("Away_Late_Run_Suppression")
            _home_late_pressure = _ou_lane_float("Home_Late_Run_Pressure")
            _home_late_suppression = _ou_lane_float("Home_Late_Run_Suppression")

            ou_early_net_run_environment = round(
                _away_early_pressure
                - _away_early_suppression
                + _home_early_pressure
                - _home_early_suppression,
                3,
            )
            ou_late_net_run_environment = round(
                _away_late_pressure
                - _away_late_suppression
                + _home_late_pressure
                - _home_late_suppression,
                3,
            )

            ou_early_over_path = ou_early_net_run_environment >= 0.45
            ou_early_under_path = ou_early_net_run_environment <= -0.45
            ou_late_over_path = ou_late_net_run_environment >= 0.55
            ou_late_under_path = ou_late_net_run_environment <= -0.55

            ou_late_over_strong = bool(
                ou_late_net_run_environment >= 0.75
                or _away_late_pressure >= 1.00
                or _home_late_pressure >= 1.00
            )

            if ou_early_over_path and ou_late_over_path:
                ou_run_path = "full_game_over_path"
            elif ou_early_under_path and ou_late_under_path:
                ou_run_path = "full_game_under_path"
            elif ou_early_over_path and ou_late_under_path:
                ou_run_path = "f5_over_path"
            elif ou_early_under_path and ou_late_over_path:
                ou_run_path = "f5_under_late_over_conflict"
            elif ou_late_over_path:
                ou_run_path = "late_over_path"
            elif ou_late_under_path:
                ou_run_path = "late_under_path"
            else:
                ou_run_path = "mixed_watch"

            # Controlled lane-supported OVER carveout.
            # This is intentionally not a broad threshold drop. It only helps full-game
            # OVER when the current model is already close and late-game scoring risk
            # is clearly supported.
            ou_lane_supported_over_fire = bool(
                (not ou_fire_candidate)
                and _ou_pick == "OVER"
                and has_real_total
                and (not projection_cap_hit)
                and (float(edge) >= 0.50)
                and (confidence >= 0.67)
                and ou_late_over_path
                and (ou_early_net_run_environment >= -0.20)
                and ("League Avg" not in (away_pitcher or ""))
                and ("League Avg" not in (home_pitcher or ""))
                and (not away_low)
                and (not home_low)
                and (not _fallback_used_from_path(away_path))
                and (not _fallback_used_from_path(home_path))
            )

            ou_fire_candidate = bool(ou_fire_candidate or ou_lane_supported_over_fire)

            # Full-game UNDER is unsafe when late pressure is strong.
            # If early suppression exists, this is usually F5 UNDER / full-game pass.
            # If early is neutral/positive, this can become a late-over review path.
            ou_lane_late_risk_under_block = bool(
                ou_fire_candidate
                and _ou_pick == "UNDER"
                and has_real_total
                and (
                    ou_late_over_strong
                    or ou_run_path in {"f5_under_late_over_conflict", "late_over_path"}
                )
            )

            # Early OVER but late suppression means this is often F5 OVER rather than
            # full-game OVER. Block only when the full-game edge is not very strong.
            ou_lane_f5_over_full_game_block = bool(
                ou_fire_candidate
                and _ou_pick == "OVER"
                and has_real_total
                and ou_run_path == "f5_over_path"
                and (float(edge) < 1.50)
            )

            ou_lane_full_game_block = bool(
                ou_lane_late_risk_under_block
                or ou_lane_f5_over_full_game_block
            )

            ou_full_game_under_reliever_depth_risk = bool(
                ou_fire_candidate
                and has_real_total
                and (_ou_pick in ("UNDER", "LEAN_UNDER"))
                and (_bl <= 8.5)
                and (float(edge) <= -1.5)
                and (
                    away_reliever_depth_risk == "high"
                    or home_reliever_depth_risk == "high"
                )
            )
            under_reliever_depth_block = bool(
                ou_fire_candidate
                and _ou_pick == "UNDER"
                and has_real_total
                and (
                    bool(ou_full_game_under_reliever_depth_risk)
                    or (
                        away_reliever_depth_risk == "high"
                        and home_reliever_depth_risk == "high"
                    )
                )
            )
            ou_fired = bool(
                ou_fire_candidate
                and not under_reliever_depth_block
                and not ou_lane_full_game_block
            )
            trigger_tags = "|".join(filter(None, [
                "ou_high_confidence" if ou_fired else None,
                "ou_clean_strong_carveout" if (clean_strong_ou and not standard_ou_fire) else None,
                "ml_high_signal" if ml_fired else None,
                "sportsdataio" if (odds_source == "SportsDataIO") else None,
                "parlay_api" if (odds_source == "parlay_api") else None,
                "fallback_line" if (not has_real_total) else None,
                "ou_moderate_over_candidate" if ou_moderate_over_candidate else None,
                "ou_tail_over_risk" if ou_tail_over_risk else None,
                "ou_full_game_under_reliever_depth_risk"
                if ou_full_game_under_reliever_depth_risk
                else None,
                "ou_under_reliever_depth_block"
                if under_reliever_depth_block
                else None,
                "ou_lane_supported_over_fire"
                if ou_lane_supported_over_fire
                else None,
                "ou_lane_late_risk_under_block"
                if ou_lane_late_risk_under_block
                else None,
                "ou_lane_f5_over_full_game_block"
                if ou_lane_f5_over_full_game_block
                else None,
            ]))
            game_data["Fired_Play"] = ou_fired
            game_data["OU_Fired"] = ou_fired
            game_data["Trigger_Tags"] = trigger_tags
            game_data["OU_Moderate_Over_Candidate"] = ou_moderate_over_candidate
            game_data["OU_Tail_Over_Risk"] = ou_tail_over_risk
            game_data["OU_Full_Game_Under_Reliever_Depth_Risk"] = (
                ou_full_game_under_reliever_depth_risk
            )
            game_data["OU_Under_Reliever_Depth_Block"] = under_reliever_depth_block
            game_data["OU_Early_Net_Run_Environment"] = ou_early_net_run_environment
            game_data["OU_Late_Net_Run_Environment"] = ou_late_net_run_environment
            game_data["OU_Run_Path"] = ou_run_path
            game_data["OU_Lane_Supported_Over_Fire"] = ou_lane_supported_over_fire
            game_data["OU_Lane_Late_Risk_Under_Block"] = ou_lane_late_risk_under_block
            game_data["OU_Lane_F5_Over_Full_Game_Block"] = ou_lane_f5_over_full_game_block
            game_data["OU_Lane_Full_Game_Block"] = ou_lane_full_game_block
            if ou_fired:
                no_fire_ou = ""
            else:
                if ou_lane_late_risk_under_block:
                    no_fire_ou = "lane_late_risk_under_block"
                elif ou_lane_f5_over_full_game_block:
                    no_fire_ou = "lane_f5_over_full_game_block"
                elif (
                    projection_cap_hit
                    and has_real_total
                    and (confidence >= fire_threshold)
                ):
                    no_fire_ou = "projection_cap"
                elif not has_real_total:
                    no_fire_ou = "fallback_line_used"
                elif under_reliever_depth_block:
                    no_fire_ou = "under_reliever_depth_risk"
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
            game_data["Model_Notes"] = (
                f"edge={edge:.2f}|conf={confidence:.2f}|book={odds_info.get('book', '')}"
                + (f"|{ou_market_quality_note}" if ou_market_quality_note else "")
            )
            game_data["Confidence_Tier"] = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.60 else "low")
            game_data["Edge_Tier"] = "strong" if abs(edge) >= 2.0 else ("medium" if abs(edge) >= 1.0 else "thin")
            game_data["Bet_Type"] = "total"
            game_data["Side"] = "over" if "OVER" in (prediction or "").upper() else ("under" if "UNDER" in (prediction or "").upper() else "")
            game_data["OU_Edge"] = edge
            game_data["OU_Confidence"] = f"{confidence:.0%}"
            game_data["OU_Side"] = game_data["Side"]
            game_data["OU_Bet_Type"] = "total"
            line_status = "market" if bool(odds_info.get("_has_real_total", False)) else "fallback"
            game_data["Line_Status"] = line_status
            game_data["Fallback_Used"] = not bool(odds_info.get("_has_real_total", False))
            dq_parts = []
            if line_status == "fallback":
                dq_parts.append("fallback_line")
            _away_low = bool(safe_get(away_stats, "LowIP", False))
            _home_low = bool(safe_get(home_stats, "LowIP", False))
            # Separate true missing-starter fallback from real low-IP starters.
            # League Avg means no usable real starter stats. LowIP means real stats
            # exist but sample size is thin; it should not be labeled fallback_pitcher.
            if (
                "League Avg" in (away_pitcher or "")
                or "League Avg" in (home_pitcher or "")
            ):
                dq_parts.append("fallback_pitcher")
            if _away_low or _home_low:
                dq_parts.append("low_ip")
            if not ml_market_ok:
                dq_parts.append("ml_market_missing")
            game_data["Data_Quality_Flag"] = "|".join(dq_parts)

            game_data["OU_Confidence_Bucket"] = _confidence_bucket_5pt(
                _parse_ou_confidence_percent_optional(game_data)
            )
            game_data["ML_Confidence_Bucket"] = _confidence_bucket_5pt(
                _parse_ml_confidence_percent_optional(game_data)
            )

            # Export-only O/U sharpness fields derived from odds_info + the modifier
            # note already computed above. Does NOT change behavior, thresholds,
            # confidence math, or fire gates. Read once here so the picks-board
            # CSV has a stable, compact sharpness block per row.
            #
            # Semantics:
            #   OU_Sharpness_Inputs_OK -> both Pinnacle and retail total exist
            #       (ESPN DK preferred, else DraftKings; parity with sharpness gap path)
            #   OU_Sharpness_OK        -> meaningful signal (|gap| >= 0.5, same
            #                              boundary the confidence modifier uses)
            try:
                _ou_sharp_lookup_key = f"{normalize_team_name(away_team)} @ {normalize_team_name(home_team)}"
                _targeted_ou_sharp = (
                    ou_sharp_totals_by_game.get(_ou_sharp_lookup_key)
                    if isinstance(ou_sharp_totals_by_game, dict)
                    else None
                )
                # RECOVERY PATCH: Ensure we extract from the new list-based odds_info structure.
                _pinn_export = odds_info.get("pinnacle_total_line")
                _retail_export = odds_info.get("espn_draftkings_total_line") or odds_info.get("draftkings_total_line")
                _retail_book_export = "Draftkings" if _retail_export is not None else ""

                # Fallback: if odds_info is missing these keys, recover directly
                # from a raw list-format bookmaker payload when present.
                if _pinn_export is None or _retail_export is None:
                    for bm in odds_info.get("bookmakers", []):
                        try:
                            val = bm["markets"][0]["outcomes"][0]["point"]
                            if bm["key"] == "pinnacle":
                                _pinn_export = val
                            if bm["key"] in ["draftkings", "bovada"]:
                                _retail_export = val
                        except (IndexError, KeyError, TypeError):
                            continue

                if isinstance(_targeted_ou_sharp, dict):
                    _pinn_export = _targeted_ou_sharp.get("pinnacle_total_line")
                    _retail_export = _targeted_ou_sharp.get("retail_total_line")
                    _retail_book_export = _targeted_ou_sharp.get("retail_book") or ""

                _inputs_ok_export = _pinn_export is not None and _retail_export is not None
                if _inputs_ok_export:
                    _gap_export = round(float(_pinn_export) - float(_retail_export), 2)
                    ou_sharp_layer_gap = _gap_export
                    if _gap_export > 0:
                        _dir_export = "OVER"
                    elif _gap_export < 0:
                        _dir_export = "UNDER"
                    else:
                        _dir_export = ""
                    _signal_ok_export = abs(_gap_export) >= 0.5
                else:
                    _gap_export = ""
                    _dir_export = ""
                    _signal_ok_export = False
                game_data["OU_Pinnacle_Total"] = _pinn_export if _pinn_export is not None else ""
                game_data["OU_Retail_Total"] = _retail_export if _retail_export is not None else ""
                game_data["OU_Retail_Book"] = _retail_book_export
                game_data["OU_Sharp_Gap"] = _gap_export
                game_data["OU_Sharp_Direction"] = _dir_export
                # Numeric applied-delta parsed from the existing note token
                # (format: "ou_sharp=+0.04@1.5"). Gap/direction are already
                # separate fields; this column shows only the signed confidence
                # delta actually applied by the modifier (blank when no delta).
                _modifier_delta_str = ""
                if ou_market_quality_note and ou_market_quality_note.startswith("ou_sharp="):
                    try:
                        _tok = ou_market_quality_note[len("ou_sharp="):]
                        _num_str = _tok.split("@", 1)[0]
                        _modifier_delta_str = f"{float(_num_str):+.2f}"
                    except (ValueError, IndexError):
                        _modifier_delta_str = ""
                game_data["OU_Sharp_Modifier"] = _modifier_delta_str
                game_data["OU_Sharpness_Inputs_OK"] = _inputs_ok_export
                game_data["OU_Sharpness_OK"] = _signal_ok_export
            except (TypeError, ValueError):
                game_data["OU_Pinnacle_Total"] = ""
                game_data["OU_Retail_Total"] = ""
                game_data["OU_Retail_Book"] = ""
                game_data["OU_Sharp_Gap"] = ""
                game_data["OU_Sharp_Direction"] = ""
                game_data["OU_Sharp_Modifier"] = ""
                game_data["OU_Sharpness_Inputs_OK"] = False
                game_data["OU_Sharpness_OK"] = False

            # Export-only stadium env vs sharp total gap (Pinnacle − DK). Telemetry only.
            _env_retract = venue in RETRACTABLE_ROOF_VENUES
            game_data["ENV_Is_Retractable"] = _env_retract
            if ou_sharp_layer_gap is not None:
                game_data["SHARP_OU_Delta"] = round(float(ou_sharp_layer_gap), 2)
            else:
                game_data["SHARP_OU_Delta"] = ""
            _env_dir_pick = ou_pick in ("OVER", "UNDER")
            _env_sg = ou_sharp_layer_gap
            if _env_retract and _env_dir_pick and _env_sg is not None:
                if float(_env_sg) > 0.0:
                    game_data["ENV_Validation"] = ou_pick == "OVER"
                    game_data["ENV_Conflict"] = ou_pick == "UNDER"
                elif float(_env_sg) < 0.0:
                    game_data["ENV_Validation"] = ou_pick == "UNDER"
                    game_data["ENV_Conflict"] = ou_pick == "OVER"
                else:
                    game_data["ENV_Validation"] = False
                    game_data["ENV_Conflict"] = False
            else:
                game_data["ENV_Validation"] = False
                game_data["ENV_Conflict"] = False

            # Export-only pitcher context for the readable picks board.
            # Uses the same pitcher names and stats the model used for this row
            # (including league-average fallbacks when applicable). No recompute.
            game_data["Away_Pitcher"] = away_pitcher or ""
            game_data["Home_Pitcher"] = home_pitcher or ""
            game_data["Away_xERA"] = safe_get(away_stats, "xERA", "N/A")
            game_data["Home_xERA"] = safe_get(home_stats, "xERA", "N/A")
            game_data["Away_WHIP"] = safe_get(away_stats, "WHIP", "N/A")
            game_data["Home_WHIP"] = safe_get(home_stats, "WHIP", "N/A")

            # Export-only explicit archive row-purpose. Aligned with the
            # downstream eligibility gate `Total_Is_Real OR ML_Fired`.
            # Note: `ou_only` means archive-eligible because a trusted/real
            # total exists, NOT necessarily OU_Fired=True. Blank when neither
            # condition is met (such rows are filtered out by the gate and
            # never written to the archive CSV).
            _total_is_real_flag = bool(game_data.get("Total_Is_Real", False))
            _ml_fired_flag = bool(game_data.get("ML_Fired", False))
            if _total_is_real_flag and _ml_fired_flag:
                game_data["Archive_Row_Reason"] = "ou_and_ml"
            elif _total_is_real_flag:
                game_data["Archive_Row_Reason"] = "ou_only"
            elif _ml_fired_flag:
                game_data["Archive_Row_Reason"] = "ml_only"
            else:
                game_data["Archive_Row_Reason"] = ""

            # Export-only Mountain Time game-time string for the readable
            # picks board (e.g. "1:10 PM", "7:40 PM"). Mirrors the existing
            # _alert_formatted_time helper but without the " MT" suffix and
            # with the leading zero stripped for scan-friendliness.
            _mt_time_str = ""
            try:
                _game_utc = datetime.strptime(
                    game_data.get("Datetime", ""), "%Y-%m-%dT%H:%M:%SZ"
                )
                _mt = _game_utc.replace(tzinfo=utc).astimezone(timezone("US/Mountain"))
                _mt_time_str = _mt.strftime("%I:%M %p").lstrip("0")
            except Exception:
                _mt_time_str = ""
            game_data["Game_Time_MT"] = _mt_time_str

            # Stage B: additive archive telemetry for bullpen quality,
            # bullpen workload/fatigue, effective ERA, and the O/U
            # confidence stack. All values are replays of math already
            # performed upstream; nothing here feeds back into projection,
            # confidence, or fire gates. Used to validate lineup × bullpen,
            # fatigue, and confidence-stack hypotheses row-by-row from the
            # archive CSV without rerunning the model.
            _tel = proj.get("telemetry", {}) if isinstance(proj, dict) else {}
            game_data["Away_Bullpen_ERA"] = _tel.get("away_bullpen_era", "")
            game_data["Home_Bullpen_ERA"] = _tel.get("home_bullpen_era", "")
            game_data["Away_Bullpen_xERA"] = _tel.get("away_bullpen_xera", "")
            game_data["Home_Bullpen_xERA"] = _tel.get("home_bullpen_xera", "")
            game_data["Away_Bullpen_IP_Week"] = _tel.get("away_bullpen_ip_week", "")
            game_data["Home_Bullpen_IP_Week"] = _tel.get("home_bullpen_ip_week", "")
            game_data["Away_Bullpen_Relievers"] = _tel.get("away_bullpen_relievers", "")
            game_data["Home_Bullpen_Relievers"] = _tel.get("home_bullpen_relievers", "")
            game_data["Away_Reliever_Bad_xERA_Count"] = away_reliever_metrics.get(
                "bad_xera_count", 0
            )
            game_data["Home_Reliever_Bad_xERA_Count"] = home_reliever_metrics.get(
                "bad_xera_count", 0
            )
            game_data["Away_Reliever_Bad_WHIP_Count"] = away_reliever_metrics.get(
                "bad_whip_count", 0
            )
            game_data["Home_Reliever_Bad_WHIP_Count"] = home_reliever_metrics.get(
                "bad_whip_count", 0
            )
            game_data["Away_Reliever_Recent_Bad_Arm_Count"] = away_reliever_metrics.get(
                "recent_bad_arm_count", 0
            )
            game_data["Home_Reliever_Recent_Bad_Arm_Count"] = home_reliever_metrics.get(
                "recent_bad_arm_count", 0
            )
            game_data["Away_Reliever_Depth_Risk"] = away_reliever_metrics.get(
                "depth_risk", "unknown"
            )
            game_data["Home_Reliever_Depth_Risk"] = home_reliever_metrics.get(
                "depth_risk", "unknown"
            )

            # Bullpen clarity v1: exact arm-quality + availability snapshot.
            for _side, _metrics in [
                ("Away", away_reliever_metrics),
                ("Home", home_reliever_metrics),
            ]:
                game_data[f"{_side}_Reliever_Total_Count"] = _metrics.get("reliever_total_count", 0)
                game_data[f"{_side}_Reliever_Shutdown_Arm_Count"] = _metrics.get("shutdown_arm_count", 0)
                game_data[f"{_side}_Reliever_Good_Arm_Count"] = _metrics.get("good_arm_count", 0)
                game_data[f"{_side}_Reliever_Average_Arm_Count"] = _metrics.get("average_arm_count", 0)
                game_data[f"{_side}_Reliever_Risky_Arm_Count"] = _metrics.get("risky_arm_count", 0)
                game_data[f"{_side}_Reliever_Disaster_Arm_Count"] = _metrics.get("disaster_arm_count", 0)
                game_data[f"{_side}_Reliever_Recent_Used_Count"] = _metrics.get("recent_used_count", 0)
                game_data[f"{_side}_Reliever_Taxed_Arm_Count"] = _metrics.get("taxed_arm_count", 0)
                game_data[f"{_side}_Reliever_Heavy_Week_Count"] = _metrics.get("heavy_week_count", 0)
                game_data[f"{_side}_Reliever_Good_Available_Count"] = _metrics.get("good_available_count", 0)
                game_data[f"{_side}_Reliever_Bad_Recent_Count"] = _metrics.get("bad_recent_count", 0)
                game_data[f"{_side}_Reliever_Disaster_Recent_Count"] = _metrics.get("disaster_recent_count", 0)
                game_data[f"{_side}_Bullpen_Quality_Tier"] = _metrics.get("bullpen_quality_tier", "unknown")
                game_data[f"{_side}_Bullpen_Availability_Tier"] = _metrics.get("bullpen_availability_tier", "unknown")
                game_data[f"{_side}_Bullpen_Clarity_Score"] = _metrics.get("bullpen_clarity_score", 0.0)
                game_data[f"{_side}_Late_Run_Pressure_Score"] = _metrics.get("late_run_pressure_score", 0.0)
                game_data[f"{_side}_Late_Run_Suppression_Score"] = _metrics.get("late_run_suppression_score", 0.0)
            game_data["Away_Bullpen_Fatigue_Ratio"] = _tel.get("away_bullpen_fatigue_ratio", "")
            game_data["Home_Bullpen_Fatigue_Ratio"] = _tel.get("home_bullpen_fatigue_ratio", "")
            game_data["Away_Bullpen_Fatigue_Mult"] = _tel.get("away_bullpen_fatigue_mult", "")
            game_data["Home_Bullpen_Fatigue_Mult"] = _tel.get("home_bullpen_fatigue_mult", "")
            game_data["Away_Effective_ERA"] = _tel.get("away_effective_era", "")
            game_data["Home_Effective_ERA"] = _tel.get("home_effective_era", "")

            # Export-only starter / offense snapshot (same stats objects + telemetry as the row).
            game_data["Away_Starter_ERA"] = safe_get(away_stats, "ERA", "")
            game_data["Home_Starter_ERA"] = safe_get(home_stats, "ERA", "")
            game_data["Away_Starter_xERA"] = _tel.get("away_starter_xera", "")
            game_data["Home_Starter_xERA"] = _tel.get("home_starter_xera", "")
            game_data["Away_Starter_WHIP"] = safe_get(away_stats, "WHIP", "")
            game_data["Home_Starter_WHIP"] = safe_get(home_stats, "WHIP", "")
            game_data["Away_Starter_IP"] = safe_get(away_stats, "IP", "")
            game_data["Home_Starter_IP"] = safe_get(home_stats, "IP", "")
            game_data["Away_Starter_LowIP"] = bool(safe_get(away_stats, "LowIP", False))
            game_data["Home_Starter_LowIP"] = bool(safe_get(home_stats, "LowIP", False))
            game_data["Away_Starter_ERA_xERA_Gap"] = _export_era_minus_xera(
                safe_get(away_stats, "ERA", None), safe_get(away_stats, "xERA", None)
            )
            game_data["Home_Starter_ERA_xERA_Gap"] = _export_era_minus_xera(
                safe_get(home_stats, "ERA", None), safe_get(home_stats, "xERA", None)
            )
            game_data["Away_Bullpen_ERA_xERA_Gap"] = _export_era_minus_xera(
                game_data.get("Away_Bullpen_ERA"), game_data.get("Away_Bullpen_xERA")
            )
            game_data["Home_Bullpen_ERA_xERA_Gap"] = _export_era_minus_xera(
                game_data.get("Home_Bullpen_ERA"), game_data.get("Home_Bullpen_xERA")
            )
            game_data["Away_Starter_Fallback_Used"] = _fallback_used_from_path(away_path)
            game_data["Home_Starter_Fallback_Used"] = _fallback_used_from_path(home_path)
            game_data["Away_Offense_Mult"] = round(float(away_offense_mult), 4)
            game_data["Home_Offense_Mult"] = round(float(home_offense_mult), 4)

            _tel_base_conf = _tel.get("ou_base_confidence", "")
            _tel_rel_mult = _tel.get("ou_reliever_mult", "")
            _tel_low_ip_mult = _tel.get("ou_low_ip_mult", "")
            _tel_lg_avg_mult = _tel.get("ou_league_avg_mult", "")
            _tel_velo_mult = _tel.get("ou_velo_trust_mult", "")
            game_data["OU_Base_Confidence"] = _tel_base_conf
            game_data["OU_Reliever_Mult"] = _tel_rel_mult
            game_data["OU_LowIP_Mult"] = _tel_low_ip_mult
            game_data["OU_LeagueAvg_Mult"] = _tel_lg_avg_mult
            game_data["OU_Velo_Trust_Mult"] = _tel_velo_mult

            # Numeric applied sharp delta (clean float for analytics).
            # Parsed from ou_market_quality_note; 0.0 when no modifier was
            # applied for any reason (missing inputs, sub-threshold gap,
            # etc.). Paired with OU_Sharpness_Inputs_OK / OU_Sharpness_OK
            # to disambiguate "not computed" from "computed and zero".
            _ou_sharp_delta_num = 0.0
            if ou_market_quality_note and ou_market_quality_note.startswith("ou_sharp="):
                try:
                    _tok2 = ou_market_quality_note[len("ou_sharp="):]
                    _num_str2 = _tok2.split("@", 1)[0]
                    _ou_sharp_delta_num = float(_num_str2)
                except (ValueError, IndexError):
                    _ou_sharp_delta_num = 0.0
            game_data["OU_Sharp_Modifier_Numeric"] = round(_ou_sharp_delta_num, 4)

            # Compact human-readable confidence stack trace for per-row
            # debugging. Format mirrors the internal multiplicative pipeline
            # in generate_prediction: base × velo × reliever × low_ip ×
            # league_avg (+ sharp delta, additive). Empty cells fall back
            # gracefully if any component was not computed.
            def _fmt_conf_component(v, plus_sign=False):
                try:
                    f = float(v)
                    return f"{f:+.2f}" if plus_sign else f"{f:.2f}"
                except (TypeError, ValueError):
                    return "?"
            game_data["OU_Confidence_Stack"] = (
                f"base={_fmt_conf_component(_tel_base_conf)}"
                f"|velo={_fmt_conf_component(_tel_velo_mult)}"
                f"|reliever={_fmt_conf_component(_tel_rel_mult)}"
                f"|low_ip={_fmt_conf_component(_tel_low_ip_mult)}"
                f"|league_avg={_fmt_conf_component(_tel_lg_avg_mult)}"
                f"|sharp={_fmt_conf_component(_ou_sharp_delta_num, plus_sign=True)}"
            )

            # Export-only: O/U de-vig implied vs provisional true probability (Stage 1 telemetry;
            # does not affect fire, Telegram, Kelly, confidence math, or grading).
            _pr_flags = ["provisional_not_for_fire"]
            for _pk in (
                "OU_Implied_Prob_Over",
                "OU_Implied_Prob_Under",
                "OU_Implied_Prob_Pick",
                "OU_True_Prob_Over",
                "OU_True_Prob_Under",
                "OU_True_Prob_Pick",
                "OU_Prob_Edge",
                "OU_Prob_Edge_Side",
            ):
                game_data[_pk] = ""
            game_data["OU_Prob_Method"] = "v1_edge_conf_provisional"
            game_data["OU_Prob_Juice_Source"] = ""
            game_data["OU_Prob_Book"] = ""
            game_data["OU_Prob_Over_Juice"] = ""
            game_data["OU_Prob_Under_Juice"] = ""

            def _ou_prob_float(v):
                try:
                    f = float(v)
                    if f != f:
                        return None
                    return f
                except (TypeError, ValueError):
                    return None

            _toa_ou_quote = None
            _toa_ou_lookup_key = f"{normalize_team_name(away_team)} @ {normalize_team_name(home_team)}"
            _toa_ou_quotes = (
                toa_ou_totals_by_game.get(_toa_ou_lookup_key, [])
                if isinstance(toa_ou_totals_by_game, dict)
                else []
            )
            _bet_line_float = _ou_prob_float(game_data.get("Bet_Line"))
            if _bet_line_float is not None and isinstance(_toa_ou_quotes, list):
                _toa_ou_book_order = [
                    "pinnacle", "fanduel", "draftkings", "betmgm", "caesars", "bet365",
                    "betrivers", "lowvig", "betonlineag", "bovada", "betus", "mybookieag",
                ]
                _toa_ou_book_rank = {
                    _book: _idx for _idx, _book in enumerate(_toa_ou_book_order)
                }
                _same_line_quotes = []
                for _q in _toa_ou_quotes:
                    if not isinstance(_q, dict):
                        continue
                    _q_line = _ou_prob_float(_q.get("total_line"))
                    _q_over = _ou_prob_float(_q.get("over_juice"))
                    _q_under = _ou_prob_float(_q.get("under_juice"))
                    if (
                        _q_line is not None
                        and _q_line == _bet_line_float
                        and _q_over is not None
                        and _q_under is not None
                    ):
                        _same_line_quotes.append(_q)
                if _same_line_quotes:
                    _toa_ou_quote = sorted(
                        _same_line_quotes,
                        key=lambda _q: (
                            _toa_ou_book_rank.get(
                                str(_q.get("book") or "").strip().lower(),
                                len(_toa_ou_book_order),
                            ),
                            str(_q.get("book") or "").strip().lower(),
                        ),
                    )[0]

            if _toa_ou_quote is not None:
                game_data["OU_Prob_Juice_Source"] = "the_odds_api"
                game_data["OU_Prob_Book"] = str(_toa_ou_quote.get("book") or "")
                game_data["OU_Prob_Over_Juice"] = _toa_ou_quote.get("over_juice")
                game_data["OU_Prob_Under_Juice"] = _toa_ou_quote.get("under_juice")
                game_data["OU_Prob_Method"] = "v1_edge_conf_the_odds_api_same_line"
                _pr_flags.append("toa_same_line_not_for_fire")
                _im_over, _im_under, _im_st = _ml_pair_devig_implied(
                    _toa_ou_quote.get("over_juice"), _toa_ou_quote.get("under_juice")
                )
            else:
                _im_over, _im_under, _im_st = _ml_pair_devig_implied(
                    game_data.get("Over_Juice"), game_data.get("Under_Juice")
                )
            if _im_st == "ok" and _im_over is not None and _im_under is not None:
                game_data["OU_Implied_Prob_Over"] = round(float(_im_over), 4)
                game_data["OU_Implied_Prob_Under"] = round(float(_im_under), 4)
            else:
                _pr_flags.append("ou_implied_unavailable")

            _sd = (game_data.get("Side") or "").strip().lower()
            if _sd in ("over", "under"):
                game_data["OU_Prob_Edge_Side"] = _sd
                if _im_st == "ok":
                    _im_pick = _im_over if _sd == "over" else _im_under
                    game_data["OU_Implied_Prob_Pick"] = round(float(_im_pick), 4)
                try:
                    _ou_conf01 = float(confidence)
                    if _ou_conf01 != _ou_conf01 or not (0.0 <= _ou_conf01 <= 1.0):
                        raise ValueError
                except (TypeError, ValueError):
                    _ou_conf01 = None
                try:
                    _edge_abs = abs(float(edge))
                except (TypeError, ValueError):
                    _edge_abs = None
                if _ou_conf01 is not None and _edge_abs is not None:
                    _edge_term = min(_edge_abs, 3.0) * 0.012
                    _p_pick = _clamp_unit_interval(_ou_conf01 + _edge_term, 0.48, 0.85)
                    if _p_pick is not None:
                        if _sd == "over":
                            _t_o, _t_u = _p_pick, 1.0 - _p_pick
                        else:
                            _t_u, _t_o = _p_pick, 1.0 - _p_pick
                        game_data["OU_True_Prob_Pick"] = round(_p_pick, 4)
                        game_data["OU_True_Prob_Over"] = round(_t_o, 4)
                        game_data["OU_True_Prob_Under"] = round(_t_u, 4)
                else:
                    _pr_flags.append("true_prob_inputs_missing")
                if (
                    _im_st == "ok"
                    and game_data["OU_Implied_Prob_Pick"] != ""
                    and game_data["OU_True_Prob_Pick"] != ""
                ):
                    try:
                        game_data["OU_Prob_Edge"] = round(
                            float(game_data["OU_True_Prob_Pick"])
                            - float(game_data["OU_Implied_Prob_Pick"]),
                            4,
                        )
                    except (TypeError, ValueError):
                        pass
            else:
                _pr_flags.append("ou_side_missing")

            game_data["OU_Prob_Calibration_Flag"] = "|".join(_pr_flags)

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
        "Game_ID", "Game_Num", "Doubleheader", "Datetime", "Game_Date",
        "Game", "Game_Status", "Venue", "Weather_Runs_Mult", "Retractable_Roof_Weather",
        "ENV_Is_Retractable", "ENV_Validation", "ENV_Conflict",
        "Projected_Total", "Base_Projected_Total", "Base_OU_Edge",
        "Raw_Projected_Total", "Raw_OU_Edge",
        "OU_Edge_Calibration_Factor", "OU_Edge_Calibration_Tag",
        "Away_Early_Run_Pressure", "Away_Early_Run_Suppression",
        "Away_Late_Run_Pressure", "Away_Late_Run_Suppression",
        "Home_Early_Run_Pressure", "Home_Early_Run_Suppression",
        "Home_Late_Run_Pressure", "Home_Late_Run_Suppression",
        "Weather_Interaction_Adjustment", "Full_Picture_Adjustment_Raw",
        "Away_Run_Pressure_Adjustment", "Home_Run_Pressure_Adjustment",
        "Run_Pressure_Adjustment", "Run_Pressure_Mode", "Run_Pressure_Reasons",
        "Away_Runs", "Home_Runs",
        "Away_Runs_Raw", "Home_Runs_Raw", "Away_Runs_Safety", "Home_Runs_Safety",
        "Away_Cap_Diff", "Home_Cap_Diff", "Projection_Cap_Flag",
        "F5_Projected_Total", "F5_Away_Runs", "F5_Home_Runs",
        "F5_Model_Side", "F5_Starter_Lean", "F5_Eligible", "F5_Market_Line",
        "F5_Over_Juice", "F5_Under_Juice", "F5_Source", "F5_Book", "F5_Market_OK",
        "F5_No_Line_Reason",
        "Lineup_Impact_Away", "Lineup_Impact_Home", "Lineup_Delta_Raw", "Lineup_Delta_Effective",
        "Lineup_Mode_Away", "Lineup_Mode_Home", "Lineup_Source_Away", "Lineup_Source_Home",
        "Lineup_Confirmed_Away", "Lineup_Confirmed_Home",
        "Lineup_Player_Count_Away", "Lineup_Player_Count_Home",
        "Lineup_Matched_Count_Away", "Lineup_Matched_Count_Home",
        "Lineup_Order_Away", "Lineup_Order_Home",
        "Lineup_Signature_Away", "Lineup_Signature_Home",
        "Lineup_Fetched_At", "Lineup_Feed_Status", "Lineup_Fetch_Error",
        "Lineup_Cap_Hit_Away", "Lineup_Cap_Hit_Home",
        "Vegas_Line", "Edge",
        "OU_Edge", "OU_Confidence", "OU_Side", "OU_Bet_Type",
        "ML_Edge", "ML_Side", "ML_Bet_Type", "ML_Market_OK", "ML_Market_Status",
        "Prediction", "Confidence", "Units", "Line_Open", "Line_Current",
        "Total_Is_Real", "Odds_Line", "Over_Juice", "Under_Juice", "Odds_Book", "ML_Odds_Book",
        "Total_Line_Source", "Market_Source", "Captured_Book", "Captured_Total", "Captured_ML_Home", "Captured_ML_Away",
        "Fired_Play", "OU_Fired", "ML_Fired", "Trigger_Tags",
        "OU_Moderate_Over_Candidate", "OU_Tail_Over_Risk",
        "OU_Full_Game_Under_Reliever_Depth_Risk", "OU_Under_Reliever_Depth_Block",
        "No_Fire_Reason", "No_Fire_OU_Reason", "No_Fire_ML_Reason",
        "Model_Notes",
        "Confidence_Tier", "Edge_Tier", "Bet_Type", "Side", "Play_Status", "Bettable",
        "Line_Status", "Fallback_Used", "Data_Quality_Flag",
        "Bet_Line", "Closing_Line", "CLV", "CLV_Result",
        "OU_Result", "ML_Result",
        "ML_Pick", "ML_Confidence", "ML_Value", "ML_Kelly_Units", "ML_Quality_Flag", "ML_Quality_Factor",
        "ML_Exchange_Present", "ML_Prophetx_Present", "ML_Pinnacle_Present",
        "ML_Sharpness_Inputs_OK", "ML_Sharpness_OK", "ML_Sharpness_Gate_Open", "ML_Exchange_Vs_Sharp_Gap",
        "OU_Sharpness_Inputs_OK", "OU_Sharpness_OK",
        "OU_Sharp_Direction", "OU_Sharp_Gap", "SHARP_OU_Delta", "OU_Sharp_Modifier",
        "OU_Pinnacle_Total", "OU_Retail_Total", "OU_Retail_Book",
        "OU_Confidence_Bucket", "ML_Confidence_Bucket",
        "Archive_Row_Reason",
        # Stage B telemetry (additive, non-behavior-altering).
        "Away_Bullpen_ERA", "Home_Bullpen_ERA",
        "Away_Bullpen_xERA", "Home_Bullpen_xERA",
        "Away_Bullpen_IP_Week", "Home_Bullpen_IP_Week",
        "Away_Bullpen_Relievers", "Home_Bullpen_Relievers",
        "Away_Reliever_Bad_xERA_Count", "Home_Reliever_Bad_xERA_Count",
        "Away_Reliever_Bad_WHIP_Count", "Home_Reliever_Bad_WHIP_Count",
        "Away_Reliever_Recent_Bad_Arm_Count", "Home_Reliever_Recent_Bad_Arm_Count",
        "Away_Reliever_Depth_Risk", "Home_Reliever_Depth_Risk",
        "Away_Reliever_Total_Count", "Home_Reliever_Total_Count",
        "Away_Reliever_Shutdown_Arm_Count", "Home_Reliever_Shutdown_Arm_Count",
        "Away_Reliever_Good_Arm_Count", "Home_Reliever_Good_Arm_Count",
        "Away_Reliever_Average_Arm_Count", "Home_Reliever_Average_Arm_Count",
        "Away_Reliever_Risky_Arm_Count", "Home_Reliever_Risky_Arm_Count",
        "Away_Reliever_Disaster_Arm_Count", "Home_Reliever_Disaster_Arm_Count",
        "Away_Reliever_Recent_Used_Count", "Home_Reliever_Recent_Used_Count",
        "Away_Reliever_Taxed_Arm_Count", "Home_Reliever_Taxed_Arm_Count",
        "Away_Reliever_Heavy_Week_Count", "Home_Reliever_Heavy_Week_Count",
        "Away_Reliever_Good_Available_Count", "Home_Reliever_Good_Available_Count",
        "Away_Reliever_Bad_Recent_Count", "Home_Reliever_Bad_Recent_Count",
        "Away_Reliever_Disaster_Recent_Count", "Home_Reliever_Disaster_Recent_Count",
        "Away_Bullpen_Quality_Tier", "Home_Bullpen_Quality_Tier",
        "Away_Bullpen_Availability_Tier", "Home_Bullpen_Availability_Tier",
        "Away_Bullpen_Clarity_Score", "Home_Bullpen_Clarity_Score",
        "Away_Late_Run_Pressure_Score", "Home_Late_Run_Pressure_Score",
        "Away_Late_Run_Suppression_Score", "Home_Late_Run_Suppression_Score",
        "Away_Bullpen_Fatigue_Ratio", "Home_Bullpen_Fatigue_Ratio",
        "Away_Bullpen_Fatigue_Mult", "Home_Bullpen_Fatigue_Mult",
        "Away_Effective_ERA", "Home_Effective_ERA",
        "Away_Starter_ERA", "Home_Starter_ERA",
        "Away_Starter_xERA", "Home_Starter_xERA",
        "Away_Starter_WHIP", "Home_Starter_WHIP",
        "Away_Starter_IP", "Home_Starter_IP",
        "Away_Starter_LowIP", "Home_Starter_LowIP",
        "Away_Starter_ERA_xERA_Gap", "Home_Starter_ERA_xERA_Gap",
        "Away_Bullpen_ERA_xERA_Gap", "Home_Bullpen_ERA_xERA_Gap",
        "Away_Starter_Fallback_Used", "Home_Starter_Fallback_Used",
        "Away_Offense_Mult", "Home_Offense_Mult",
        "OU_Base_Confidence", "OU_Reliever_Mult",
        "OU_LowIP_Mult", "OU_LeagueAvg_Mult", "OU_Velo_Trust_Mult",
        "OU_Sharp_Modifier_Numeric", "OU_Confidence_Stack",
        # Stage 1: implied vs provisional true probability (export-only telemetry).
        "OU_Implied_Prob_Over",
        "OU_Implied_Prob_Under",
        "OU_Implied_Prob_Pick",
        "OU_True_Prob_Over",
        "OU_True_Prob_Under",
        "OU_True_Prob_Pick",
        "OU_Prob_Edge",
        "OU_Prob_Edge_Side",
        "OU_Prob_Method",
        "OU_Prob_Juice_Source",
        "OU_Prob_Book",
        "OU_Prob_Over_Juice",
        "OU_Prob_Under_Juice",
        "OU_Prob_Calibration_Flag",
    ]
    eligible_export = [
        r for r in results
        if r.get("Total_Is_Real", False) or r.get("ML_Fired", False)
    ]
    telegram_f5_alerts = []
    _f5_candidate_n = 0
    if eligible_export:
        # Filename YYYYMMDD = active MT slate date (04:00 rollover), not wall-clock
        # calendar date — so late-night reruns of the prior slate do not shift the
        # archive prefix to "tomorrow". HHMM suffix remains wall-clock for ordering.
        archive_date = f"{today_mt.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M')}"
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        combined_path = archive_output_path("predictions", archive_date)
        combined_df = pd.DataFrame(eligible_export, columns=export_cols)
        combined_df.to_csv(combined_path, index=False)
        print(f"\n💾 Saved {len(eligible_export)} combined row(s) → {combined_path}")
        diagnostics_path = archive_output_path("diagnostics", archive_date)
        diagnostics_cols = list(export_cols)
        for _row in eligible_export:
            for _col in _row.keys():
                if _col not in diagnostics_cols:
                    diagnostics_cols.append(_col)
        diagnostics_df = pd.DataFrame(eligible_export, columns=diagnostics_cols)
        diagnostics_df.to_csv(diagnostics_path, index=False)
        print(f"💾 Saved {len(eligible_export)} diagnostics row(s) → {diagnostics_path}")
        send_telegram_file(
            diagnostics_path,
            caption=f"🧪 Over Gang diagnostics — {datetime.now().strftime('%b %d')}",
        )
        send_telegram_file(
            combined_path,
            caption=f"📊 Over Gang predictions — {datetime.now().strftime('%b %d')}",
        )

        # Pitcher K prop board: separate market module export only.
        # Uses today's probable starters from eligible_export and real K9 from data/pitcher_k_stats.csv.
        k_board_rows = []
        k_stats_path = os.path.join(DATA_DIR, "pitcher_k_stats.csv")
        try:
            k_stats_df = pd.read_csv(k_stats_path)
        except Exception as _e_k_stats:
            print(f"⚠️ pitcher_k_stats.csv unavailable for K board: {_e_k_stats}")
            k_stats_df = pd.DataFrame()
        if not k_stats_df.empty and "Name" in k_stats_df.columns:
            k_stats_df["_norm_name"] = k_stats_df["Name"].astype(str).apply(DataManager.normalize_name)
            k_stats_by_name = {
                row["_norm_name"]: row
                for _, row in k_stats_df.iterrows()
                if row.get("_norm_name")
            }
        else:
            k_stats_by_name = {}

        k_prop_rows = []
        k_prop_source = "none"
        parlay_k_props_last_error = ""

        props_url = "https://api.parlay-api.com/v1/sports/baseball_mlb/props"
        props_params = {
            "regions": "us",
            "markets": "player_strikeouts",
            "oddsFormat": "american",
            "limit": 1000,
        }
        odds_api_key = os.getenv("ODDS_API_KEY", "")
        if not odds_api_key:
            print("⚠️ ODDS_API_KEY missing; skipping Parlay pitcher K props fetch.")
        else:
            props_headers = {"X-API-Key": odds_api_key, "Accept": "application/json"}
            for _attempt in range(1, 4):
                try:
                    props_resp = requests.get(
                        props_url,
                        params=props_params,
                        headers=props_headers,
                        timeout=25,
                    )
                    props_resp.raise_for_status()
                    props_payload = props_resp.json()

                    if isinstance(props_payload, dict) and isinstance(props_payload.get("data"), list):
                        k_prop_rows = props_payload.get("data") or []
                    elif isinstance(props_payload, list):
                        k_prop_rows = props_payload
                    else:
                        k_prop_rows = []

                    if k_prop_rows:
                        k_prop_source = "parlay_api"
                        print(
                            f"🎯 Parlay pitcher K props loaded: {len(k_prop_rows)} raw row(s) "
                            f"on attempt {_attempt}/3"
                        )
                        break

                    parlay_k_props_last_error = "empty_or_unexpected_payload"
                    print(
                        f"⚠️ Parlay pitcher K props attempt {_attempt}/3 returned "
                        "no usable raw rows."
                    )
                except Exception as _e_k_props:
                    parlay_k_props_last_error = str(_e_k_props)
                    print(
                        f"⚠️ Parlay pitcher K props attempt {_attempt}/3 unavailable: "
                        f"{_e_k_props}"
                    )

                if _attempt < 3:
                    time.sleep(2)

        starter_norms = set()
        for _r in eligible_export:
            for _pk in ("Away_Pitcher", "Home_Pitcher"):
                _p = str(_r.get(_pk) or "").strip()
                _pn = DataManager.normalize_name(_p)
                if _pn:
                    starter_norms.add(_pn)

        book_order = [
            "pinnacle", "draftkings", "fanduel", "bet365", "betmgm",
            "betrivers", "fanatics", "caesars", "parx", "hardrock",
        ]
        book_rank = {book: i for i, book in enumerate(book_order)}
        valid_k_market_keys = {"player_strikeouts", "player_pitcher_strikeouts", "pitcher_strikeouts"}

        def _select_k_props_from_rows(rows, source_label):
            valid_k_props_by_pitcher = {}
            reject_counts = {}
            raw_count = 0

            def _reject(reason):
                reject_counts[reason] = reject_counts.get(reason, 0) + 1

            for prop in rows or []:
                raw_count += 1
                if not isinstance(prop, dict):
                    _reject("not_dict")
                    continue

                player = str(prop.get("player") or "").strip()
                market_key = str(prop.get("market_key") or "").strip().lower()
                market_name = str(prop.get("market") or "").strip().lower()

                if market_key not in valid_k_market_keys:
                    _reject("market_key")
                    continue
                if market_name and market_name != "pitcher strikeouts":
                    _reject("market_name")
                    continue
                if "combo" in market_key or "combo" in market_name:
                    _reject("combo")
                    continue
                if prop.get("line") is None:
                    _reject("line")
                    continue
                if prop.get("over_price") is None:
                    _reject("over_price")
                    continue
                if prop.get("under_price") is None:
                    _reject("under_price")
                    continue
                if not player:
                    _reject("player")
                    continue
                if "+" in player or "{" in player or "}" in player:
                    _reject("compound_player")
                    continue

                pnorm = DataManager.normalize_name(player)
                if pnorm not in starter_norms:
                    _reject("not_slate_starter")
                    continue

                valid_k_props_by_pitcher.setdefault(pnorm, []).append(prop)

            selected = {}
            for pnorm, rows_for_pitcher in valid_k_props_by_pitcher.items():
                selected[pnorm] = sorted(
                    rows_for_pitcher,
                    key=lambda x: (
                        book_rank.get(str(x.get("bookmaker") or "").lower(), len(book_order)),
                        str(x.get("bookmaker") or "").lower(),
                    ),
                )[0]

            print(
                f"🎯 Pitcher K prop attach [{source_label}]: "
                f"raw={raw_count}, matched_pitchers={len(valid_k_props_by_pitcher)}, "
                f"selected={len(selected)}, rejects={reject_counts}"
            )
            return selected

        selected_k_props = _select_k_props_from_rows(k_prop_rows, k_prop_source)

        if not selected_k_props:
            try:
                from core.the_odds_api import fetch_pitcher_strikeout_props

                print(
                    "⚠️ Parlay pitcher K props did not attach to slate starters "
                    f"(last_error={parlay_k_props_last_error or 'none'}). "
                    "Trying The Odds API pitcher_strikeouts fallback."
                )
                toa_k_prop_rows = fetch_pitcher_strikeout_props()
                toa_selected_k_props = _select_k_props_from_rows(
                    toa_k_prop_rows,
                    "the_odds_api_fallback",
                )
                if toa_selected_k_props:
                    selected_k_props = toa_selected_k_props
                    k_prop_rows = toa_k_prop_rows
                    k_prop_source = "the_odds_api_fallback"
            except Exception as _e_toa_k_props:
                print(f"⚠️ The Odds API pitcher K props fallback unavailable: {_e_toa_k_props}")

        def _k_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _k_int(v):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None

        def _load_opponent_k_profiles():
            """Build team strikeout profiles from local batter_stats.csv for K prop board context.

            Export-only helper. Does not change K projections, K edge, K fired logic,
            or Telegram behavior.
            """
            profiles = {}
            batter_path = os.path.join(DATA_DIR, "batter_stats.csv")
            try:
                bdf = pd.read_csv(batter_path)
            except Exception as _e_opp_k:
                print(f"⚠️ opponent K profiles unavailable for K board: {_e_opp_k}")
                return profiles

            required_cols = {"team_name", "pa", "so"}
            if not required_cols.issubset(set(bdf.columns)):
                missing = sorted(required_cols - set(bdf.columns))
                print(f"⚠️ opponent K profiles unavailable; batter_stats missing columns: {missing}")
                return profiles

            bdf = bdf.copy()
            bdf["_pa_num"] = pd.to_numeric(bdf["pa"], errors="coerce")
            bdf["_so_num"] = pd.to_numeric(bdf["so"], errors="coerce")

            team = (
                bdf.groupby("team_name", dropna=False)
                .agg(PA=("_pa_num", "sum"), SO=("_so_num", "sum"))
                .reset_index()
            )

            total_pa = float(team["PA"].sum()) if "PA" in team.columns else 0.0
            total_so = float(team["SO"].sum()) if "SO" in team.columns else 0.0
            if total_pa <= 0 or total_so < 0:
                print("⚠️ opponent K profiles unavailable; invalid league PA/SO totals")
                return profiles

            league_k_pct = total_so / total_pa
            if league_k_pct <= 0:
                print("⚠️ opponent K profiles unavailable; invalid league K%")
                return profiles

            def _opp_k_profile(k_index):
                if k_index >= 1.08:
                    return "HIGH_K"
                if k_index <= 0.92:
                    return "LOW_K"
                return "NEUTRAL"

            for _, trow in team.iterrows():
                team_name = str(trow.get("team_name") or "").strip()
                pa = _k_float(trow.get("PA"))
                so_val = _k_float(trow.get("SO"))
                if not team_name or pa is None or so_val is None or pa <= 0:
                    continue

                k_pct = so_val / pa
                k_index = k_pct / league_k_pct if league_k_pct else None
                if k_index is None:
                    continue

                rec = {
                    "Opponent_K_Pct": k_pct,
                    "Opponent_K_Index": k_index,
                    "Opponent_K_Profile": _opp_k_profile(k_index),
                }
                profiles[team_name] = rec
                profiles[DataManager.normalize_name(team_name)] = rec

            print(
                f"✅ Loaded opponent K profiles for K board: "
                f"{len({str(x).strip() for x in team['team_name'].dropna()})} team(s), "
                f"league_k_pct={league_k_pct:.4f}"
            )
            return profiles

        opponent_k_profiles = _load_opponent_k_profiles()

        for _r in eligible_export:
            game_name_k = str(_r.get("Game") or "")
            if " @ " in game_name_k:
                away_team_k, home_team_k = game_name_k.split(" @ ", 1)
            else:
                away_team_k, home_team_k = "", ""
            for side, pitcher_key, team_k, opp_k in (
                ("away", "Away_Pitcher", away_team_k, home_team_k),
                ("home", "Home_Pitcher", home_team_k, away_team_k),
            ):
                pitcher = str(_r.get(pitcher_key) or "").strip()
                pnorm = DataManager.normalize_name(pitcher)
                krow = k_stats_by_name.get(pnorm)
                prop = selected_k_props.get(pnorm)
                no_fire_k_reason = ""

                ip = so = k9 = games_started = games_pitched = None
                raw_ip_per_start = expected_starter_ip = None
                expected_starter_ip_source = ""
                starter_workload_profile = ""
                workload_eligible = False
                projected_ip = projected_ks = k_edge = None
                if krow is None:
                    no_fire_k_reason = "missing_k_stats"
                else:
                    ip = _k_float(krow.get("IP"))
                    so = _k_int(krow.get("SO"))
                    k9 = _k_float(krow.get("K9"))
                    games_started = _k_float(krow.get("Games_Started"))
                    games_pitched = _k_float(krow.get("Games_Pitched"))
                    raw_ip_per_start = _k_float(krow.get("Raw_IP_Per_Start"))
                    expected_starter_ip = _k_float(krow.get("Expected_Starter_IP"))
                    expected_starter_ip_source = str(krow.get("Expected_Starter_IP_Source") or "").strip()
                    starter_workload_profile = str(krow.get("Starter_Workload_Profile") or "").strip()
                    _workload_eligible_raw = krow.get("Workload_Eligible")
                    workload_eligible = str(_workload_eligible_raw).strip().lower() in {"true", "1", "yes", "y"}

                k_line = _k_float(prop.get("line")) if prop else None
                over_price = _k_float(prop.get("over_price")) if prop else None
                under_price = _k_float(prop.get("under_price")) if prop else None

                if not no_fire_k_reason and not starter_workload_profile:
                    no_fire_k_reason = "workload_source_missing"
                elif not no_fire_k_reason and not workload_eligible:
                    if starter_workload_profile == "mixed_role":
                        no_fire_k_reason = "mixed_role_workload"
                    elif starter_workload_profile == "relief_only":
                        no_fire_k_reason = "relief_only_workload"
                    elif starter_workload_profile == "clean_starter_raw_out_of_range":
                        no_fire_k_reason = "inflated_ip_per_start"
                    elif starter_workload_profile == "invalid_workload":
                        no_fire_k_reason = "invalid_workload_sample"
                    else:
                        no_fire_k_reason = "workload_not_eligible"

                if (
                    not no_fire_k_reason
                    and (
                        prop is None
                        or k_line is None
                        or over_price is None
                        or under_price is None
                    )
                ):
                    no_fire_k_reason = "no_market_line"

                if not no_fire_k_reason and (games_started is None or games_started < 3):
                    no_fire_k_reason = "not_starter_sample"
                if not no_fire_k_reason and k_line < 2.5:
                    no_fire_k_reason = "low_k_line"

                if not no_fire_k_reason and expected_starter_ip is None:
                    no_fire_k_reason = "missing_expected_starter_ip"

                if not no_fire_k_reason and (ip is None or ip < 20):
                    no_fire_k_reason = "low_ip_k_sample"

                if not no_fire_k_reason and k9 is not None and expected_starter_ip is not None:
                    projected_ip = max(4.0, min(6.5, expected_starter_ip))
                    projected_ks = (k9 / 9.0) * projected_ip
                    k_edge = projected_ks - k_line
                    if abs(k_edge) < 0.75:
                        no_fire_k_reason = "edge_too_small"
                elif not no_fire_k_reason:
                    no_fire_k_reason = "missing_k_stats"

                k_pick = ""
                if k_edge is not None:
                    if k_edge >= 0.75:
                        k_pick = "OVER"
                    elif k_edge <= -0.75:
                        k_pick = "UNDER"

                k_fired = bool(
                    not no_fire_k_reason
                    and prop is not None
                    and krow is not None
                    and games_started is not None
                    and games_started >= 3
                    and games_pitched is not None
                    and games_pitched > 0
                    and workload_eligible
                    and starter_workload_profile == "clean_starter"
                    and expected_starter_ip is not None
                    and ip is not None
                    and ip >= 20
                    and k_line is not None
                    and k_line >= 2.5
                    and over_price is not None
                    and under_price is not None
                    and k_edge is not None
                    and abs(k_edge) >= 0.75
                )
                if k_fired:
                    no_fire_k_reason = "fired"

                opp_k_clean = str(opp_k or "").strip()
                opp_k_profile = (
                    opponent_k_profiles.get(opp_k_clean)
                    or opponent_k_profiles.get(DataManager.normalize_name(opp_k_clean))
                    or {}
                )
                opponent_k_pct = opp_k_profile.get("Opponent_K_Pct")
                opponent_k_index = opp_k_profile.get("Opponent_K_Index")
                opponent_k_profile_name = opp_k_profile.get("Opponent_K_Profile", "")

                k_matchup_support = ""
                k_matchup_tag = ""
                if k_pick == "OVER":
                    if opponent_k_profile_name == "HIGH_K":
                        k_matchup_support = "support"
                        k_matchup_tag = "over_supported_by_high_opp_k"
                    elif opponent_k_profile_name == "LOW_K":
                        k_matchup_support = "downgrade"
                        k_matchup_tag = "over_downgrade_low_opp_k"
                    elif opponent_k_profile_name == "NEUTRAL":
                        k_matchup_support = "neutral"
                        k_matchup_tag = "over_neutral_opp_k"
                elif k_pick == "UNDER":
                    if opponent_k_profile_name == "LOW_K":
                        k_matchup_support = "support"
                        k_matchup_tag = "under_supported_by_low_opp_k"
                    elif opponent_k_profile_name == "HIGH_K":
                        k_matchup_support = "downgrade"
                        k_matchup_tag = "under_downgrade_high_opp_k"
                    elif opponent_k_profile_name == "NEUTRAL":
                        k_matchup_support = "neutral"
                        k_matchup_tag = "under_neutral_opp_k"

                k_board_rows.append({
                    "Game": game_name_k,
                    "Pitcher": pitcher,
                    "Team": team_k,
                    "Opponent": opp_k,
                    "Opponent_K_Pct": round(opponent_k_pct, 4) if opponent_k_pct is not None else "",
                    "Opponent_K_Index": round(opponent_k_index, 3) if opponent_k_index is not None else "",
                    "Opponent_K_Profile": opponent_k_profile_name,
                    "K_Matchup_Support": k_matchup_support,
                    "K_Matchup_Tag": k_matchup_tag,
                    "Bookmaker": prop.get("bookmaker") if prop else "",
                    "K_Line": k_line if k_line is not None else "",
                    "Over_Price": over_price if over_price is not None else "",
                    "Under_Price": under_price if under_price is not None else "",
                    "K9": round(k9, 2) if k9 is not None else "",
                    "IP": round(ip, 3) if ip is not None else "",
                    "SO": so if so is not None else "",
                    "Games_Started": games_started if games_started is not None else "",
                    "Games_Pitched": games_pitched if games_pitched is not None else "",
                    "Raw_IP_Per_Start": raw_ip_per_start if raw_ip_per_start is not None else "",
                    "Expected_Starter_IP": expected_starter_ip if expected_starter_ip is not None else "",
                    "Expected_Starter_IP_Source": expected_starter_ip_source,
                    "Starter_Workload_Profile": starter_workload_profile,
                    "Workload_Eligible": workload_eligible,
                    "Projected_IP": round(projected_ip, 2) if projected_ip is not None else "",
                    "Projected_Ks": round(projected_ks, 2) if projected_ks is not None else "",
                    "K_Edge": round(k_edge, 2) if k_edge is not None else "",
                    "K_Pick": k_pick,
                    "K_Fired": k_fired,
                    "No_Fire_K_Reason": no_fire_k_reason,
                    "Prop_Last_Update": prop.get("last_update") if prop else "",
                    "Canonical_Event_ID": prop.get("canonical_event_id") if prop else "",
                })

        pitcher_k_board_cols = [
            "Game", "Pitcher", "Team", "Opponent",
            "Opponent_K_Pct", "Opponent_K_Index", "Opponent_K_Profile",
            "K_Matchup_Support", "K_Matchup_Tag",
            "Bookmaker", "K_Line", "Over_Price", "Under_Price", "K9", "IP", "SO",
            "Games_Started", "Games_Pitched",
            "Raw_IP_Per_Start", "Expected_Starter_IP", "Expected_Starter_IP_Source",
            "Starter_Workload_Profile", "Workload_Eligible",
            "Projected_IP", "Projected_Ks",
            "K_Edge", "K_Pick", "K_Fired", "No_Fire_K_Reason", "Prop_Last_Update",
            "Canonical_Event_ID",
        ]
        pitcher_k_board_path = archive_output_path("pitcher_k_board", archive_date)
        pitcher_k_board_df = pd.DataFrame(k_board_rows, columns=pitcher_k_board_cols)
        pitcher_k_board_df.to_csv(pitcher_k_board_path, index=False)
        print(f"💾 Saved {len(k_board_rows)} pitcher-K row(s) → {pitcher_k_board_path}")
        try:
            _k_market_line_count = int(
                pd.to_numeric(pitcher_k_board_df.get("K_Line"), errors="coerce")
                .notna()
                .sum()
            )
        except Exception:
            _k_market_line_count = 0

        if _k_market_line_count > 0:
            send_telegram_file(
                pitcher_k_board_path,
                caption=f"🎯 Over Gang pitcher K board — {datetime.now().strftime('%b %d')}",
            )
        else:
            print(
                "⚠️ Pitcher-K board archived but NOT sent to Telegram: "
                "0 usable K market line(s)."
            )

        def _f5_board_float(v):
            try:
                if v is None or str(v).strip() == "":
                    return None
                f = float(v)
                if f != f:
                    return None
                return f
            except (TypeError, ValueError):
                return None

        f5_board_cols = [
            "Game_Date", "Game_Time_MT", "Game", "Venue",
            "Away_Pitcher", "Home_Pitcher",
            "F5_Projected_Total", "F5_Away_Runs", "F5_Home_Runs",
            "F5_Market_Line", "F5_Edge", "F5_Pick",
            "F5_Model_Side", "F5_Starter_Lean", "F5_Eligible",
            "F5_Over_Juice", "F5_Under_Juice", "F5_Source", "F5_Book",
            "F5_Market_OK", "F5_No_Line_Reason",
            "OU_Full_Game_Under_Reliever_Depth_Risk",
            "Data_Quality_Flag", "Model_Notes",
        ]
        f5_board_rows = []
        for _r in eligible_export:
            _f5_projected = _f5_board_float(_r.get("F5_Projected_Total"))
            _f5_line = _f5_board_float(_r.get("F5_Market_Line"))
            _f5_edge = ""
            _f5_pick = ""
            if _f5_projected is not None and _f5_line is not None:
                _f5_edge_val = round(_f5_projected - _f5_line, 2)
                _f5_edge = _f5_edge_val
                if _f5_edge_val > 0:
                    _f5_pick = "OVER"
                elif _f5_edge_val < 0:
                    _f5_pick = "UNDER"

            f5_board_rows.append({
                "Game_Date": _r.get("Game_Date", ""),
                "Game_Time_MT": _r.get("Game_Time_MT", ""),
                "Game": _r.get("Game", ""),
                "Venue": _r.get("Venue", ""),
                "Away_Pitcher": _r.get("Away_Pitcher", ""),
                "Home_Pitcher": _r.get("Home_Pitcher", ""),
                "F5_Projected_Total": _r.get("F5_Projected_Total", ""),
                "F5_Away_Runs": _r.get("F5_Away_Runs", ""),
                "F5_Home_Runs": _r.get("F5_Home_Runs", ""),
                "F5_Market_Line": _r.get("F5_Market_Line", ""),
                "F5_Edge": _f5_edge,
                "F5_Pick": _f5_pick,
                "F5_Model_Side": _r.get("F5_Model_Side", ""),
                "F5_Starter_Lean": _r.get("F5_Starter_Lean", ""),
                "F5_Eligible": _r.get("F5_Eligible", ""),
                "F5_Over_Juice": _r.get("F5_Over_Juice", ""),
                "F5_Under_Juice": _r.get("F5_Under_Juice", ""),
                "F5_Source": _r.get("F5_Source", ""),
                "F5_Book": _r.get("F5_Book", ""),
                "F5_Market_OK": _r.get("F5_Market_OK", ""),
                "F5_No_Line_Reason": _r.get("F5_No_Line_Reason", ""),
                "OU_Full_Game_Under_Reliever_Depth_Risk": _r.get(
                    "OU_Full_Game_Under_Reliever_Depth_Risk", ""
                ),
                "Data_Quality_Flag": _r.get("Data_Quality_Flag", ""),
                "Model_Notes": _r.get("Model_Notes", ""),
            })
        f5_board_path = archive_output_path("f5_board", archive_date)
        f5_board_df = pd.DataFrame(f5_board_rows, columns=f5_board_cols)
        f5_board_df.to_csv(f5_board_path, index=False)
        print(f"💾 Saved {len(f5_board_rows)} F5 row(s) → {f5_board_path}")
        send_telegram_file(
            f5_board_path,
            caption=f"⚾ Over Gang F5 board — {datetime.now().strftime('%b %d')}",
        )

        def _ou_prob_board_float(v):
            try:
                if v is None or str(v).strip() == "":
                    return None
                f = float(v)
                if f != f:
                    return None
                return f
            except (TypeError, ValueError):
                return None

        def _ou_prob_board_bool(v) -> bool:
            return str(v).strip().lower() in ("true", "1", "yes")

        def _ou_prob_board_clean_data(v) -> bool:
            if v is None:
                return True
            s = str(v).strip()
            return s == "" or s.lower() == "nan"

        def _ou_prob_support_bucket(prob_edge):
            if prob_edge is None:
                return ""
            if prob_edge <= 0:
                return "prob_edge_negative_or_zero"
            if prob_edge < 0.05:
                return "prob_edge_0_00_to_0_05"
            if prob_edge <= 0.10:
                return "prob_edge_0_05_to_0_10"
            if prob_edge <= 0.15:
                return "prob_edge_0_10_to_0_15"
            if prob_edge <= 0.20:
                return "prob_edge_0_15_to_0_20"
            return "prob_edge_0_20_plus"

        ou_prob_edge_board_cols = [
            "Game_Date", "Datetime", "Game", "Venue", "Prediction",
            "Projected_Total", "Bet_Line", "OU_Edge", "OU_Confidence",
            "OU_Fired", "Fired_Play",
            "OU_Implied_Prob_Pick", "OU_True_Prob_Pick", "OU_Prob_Edge",
            "OU_Prob_Edge_Side", "OU_Prob_Method", "OU_Prob_Juice_Source",
            "OU_Prob_Book", "OU_Prob_Over_Juice", "OU_Prob_Under_Juice",
            "OU_Prob_Calibration_Flag", "OU_Sharpness_OK", "OU_Sharp_Direction",
            "OU_Sharp_Gap", "SHARP_OU_Delta", "Projection_Cap_Flag",
            "Total_Is_Real", "Data_Quality_Flag", "No_Fire_OU_Reason",
            "OU_Full_Game_Under_Reliever_Depth_Risk",
            "OU_Under_Reliever_Depth_Block",
            "Away_Reliever_Depth_Risk", "Home_Reliever_Depth_Risk",
            "Model_Notes", "OU_Prob_Support_Candidate",
            "OU_Prob_Edge_Sweet_Spot", "OU_Prob_Support_Bucket",
            "OU_Prob_Support_Reason",
        ]
        ou_prob_edge_board_rows = []
        for _r in eligible_export:
            _prob_edge = _ou_prob_board_float(_r.get("OU_Prob_Edge"))
            _ou_edge = _ou_prob_board_float(_r.get("OU_Edge"))
            _total_is_real = _ou_prob_board_bool(_r.get("Total_Is_Real"))
            _projection_cap = _ou_prob_board_bool(_r.get("Projection_Cap_Flag"))
            _clean_data = _ou_prob_board_clean_data(_r.get("Data_Quality_Flag"))
            _support_candidate = bool(
                _total_is_real
                and not _projection_cap
                and _clean_data
                and _prob_edge is not None
                and _prob_edge >= 0.05
                and _ou_edge is not None
                and abs(_ou_edge) < 1.0
            )
            _sweet_spot = bool(
                _prob_edge is not None
                and 0.05 <= _prob_edge <= 0.10
            )
            _support_reason = ""
            if _support_candidate:
                _support_reason = "clean_prob_edge_thin_raw_edge"
                if _sweet_spot:
                    _support_reason += "|sweet_spot"

            ou_prob_edge_board_rows.append({
                "Game_Date": _r.get("Game_Date", ""),
                "Datetime": _r.get("Datetime", ""),
                "Game": _r.get("Game", ""),
                "Venue": _r.get("Venue", ""),
                "Prediction": _r.get("Prediction", ""),
                "Projected_Total": _r.get("Projected_Total", ""),
                "Bet_Line": _r.get("Bet_Line", ""),
                "OU_Edge": _r.get("OU_Edge", ""),
                "OU_Confidence": _r.get("OU_Confidence", ""),
                "OU_Fired": _r.get("OU_Fired", ""),
                "Fired_Play": _r.get("Fired_Play", ""),
                "OU_Implied_Prob_Pick": _r.get("OU_Implied_Prob_Pick", ""),
                "OU_True_Prob_Pick": _r.get("OU_True_Prob_Pick", ""),
                "OU_Prob_Edge": _r.get("OU_Prob_Edge", ""),
                "OU_Prob_Edge_Side": _r.get("OU_Prob_Edge_Side", ""),
                "OU_Prob_Method": _r.get("OU_Prob_Method", ""),
                "OU_Prob_Juice_Source": _r.get("OU_Prob_Juice_Source", ""),
                "OU_Prob_Book": _r.get("OU_Prob_Book", ""),
                "OU_Prob_Over_Juice": _r.get("OU_Prob_Over_Juice", ""),
                "OU_Prob_Under_Juice": _r.get("OU_Prob_Under_Juice", ""),
                "OU_Prob_Calibration_Flag": _r.get("OU_Prob_Calibration_Flag", ""),
                "OU_Sharpness_OK": _r.get("OU_Sharpness_OK", ""),
                "OU_Sharp_Direction": _r.get("OU_Sharp_Direction", ""),
                "OU_Sharp_Gap": _r.get("OU_Sharp_Gap", ""),
                "SHARP_OU_Delta": _r.get("SHARP_OU_Delta", ""),
                "Projection_Cap_Flag": _r.get("Projection_Cap_Flag", ""),
                "Total_Is_Real": _r.get("Total_Is_Real", ""),
                "Data_Quality_Flag": _r.get("Data_Quality_Flag", ""),
                "No_Fire_OU_Reason": _r.get("No_Fire_OU_Reason", ""),
                "OU_Full_Game_Under_Reliever_Depth_Risk": _r.get(
                    "OU_Full_Game_Under_Reliever_Depth_Risk", ""
                ),
                "OU_Under_Reliever_Depth_Block": _r.get(
                    "OU_Under_Reliever_Depth_Block", False
                ),
                "Away_Reliever_Depth_Risk": _r.get("Away_Reliever_Depth_Risk", ""),
                "Home_Reliever_Depth_Risk": _r.get("Home_Reliever_Depth_Risk", ""),
                "Model_Notes": _r.get("Model_Notes", ""),
                "OU_Prob_Support_Candidate": _support_candidate,
                "OU_Prob_Edge_Sweet_Spot": _sweet_spot,
                "OU_Prob_Support_Bucket": _ou_prob_support_bucket(_prob_edge),
                "OU_Prob_Support_Reason": _support_reason,
            })
        ou_prob_edge_board_path = archive_output_path("ou_prob_edge_board", archive_date)
        ou_prob_edge_board_df = pd.DataFrame(
            ou_prob_edge_board_rows,
            columns=ou_prob_edge_board_cols,
        )
        ou_prob_edge_board_df.to_csv(ou_prob_edge_board_path, index=False)
        print(
            f"💾 Saved {len(ou_prob_edge_board_rows)} OU prob-edge row(s) "
            f"→ {ou_prob_edge_board_path}"
        )
        send_telegram_file(
            ou_prob_edge_board_path,
            caption=f"📈 Over Gang OU prob-edge board — {datetime.now().strftime('%b %d')}",
        )

        def _ou_over_profile_float(v):
            try:
                if v is None or str(v).strip() == "":
                    return None
                f = float(v)
                if f != f:
                    return None
                return f
            except (TypeError, ValueError):
                return None

        def _ou_over_profile_bool(v) -> bool:
            return str(v).strip().lower() in ("true", "1", "yes")

        def _ou_over_profile_clean_data(v) -> bool:
            if v is None:
                return True
            s = str(v).strip()
            return s == "" or s.lower() == "nan"

        _ou_batter_df_cache = {"df": None}

        def _ou_batter_df():
            if _ou_batter_df_cache["df"] is None:
                try:
                    _ou_batter_df_cache["df"] = Batters.load_batter_table()
                except Exception as _e_batter_pressure:
                    print(f"⚠️ OU batter pressure unavailable: {_e_batter_pressure}")
                    _ou_batter_df_cache["df"] = pd.DataFrame()
            return _ou_batter_df_cache["df"]

        def _ou_split_game_teams(game_name):
            game_s = str(game_name or "").strip()
            if " @ " not in game_s:
                return "", ""
            away_team, home_team = game_s.split(" @ ", 1)
            return away_team.strip(), home_team.strip()

        def _ou_batter_pressure_profile(mult):
            if mult is None:
                return "UNKNOWN"
            try:
                mult_f = float(mult)
            except (TypeError, ValueError):
                return "UNKNOWN"

            # Strong pressure thresholds drive grade movement.
            # Borderline zones add readable caution tags only.
            if mult_f >= 1.06:
                return "HIGH_PRESSURE"
            if mult_f >= 1.04:
                return "HIGH_PRESSURE_CAUTION"
            if mult_f <= 0.94:
                return "LOW_PRESSURE"
            if mult_f <= 0.96:
                return "LOW_PRESSURE_CAUTION"
            return "NEUTRAL_PRESSURE"

        def _ou_batter_pressure_for_row(row):
            away_team, home_team = _ou_split_game_teams(row.get("Game", ""))
            away_pitcher = str(row.get("Away_Pitcher", "") or "").strip()
            home_pitcher = str(row.get("Home_Pitcher", "") or "").strip()

            away_hand = Batters.get_pitcher_hand(away_pitcher)
            home_hand = Batters.get_pitcher_hand(home_pitcher)

            bdf = _ou_batter_df()
            away_vs_home = Batters.offense_vs_hand_dict(bdf, away_team, home_hand)
            home_vs_away = Batters.offense_vs_hand_dict(bdf, home_team, away_hand)

            away_mult = away_vs_home.get("mult")
            home_mult = home_vs_away.get("mult")

            vals = []
            for _v in (away_mult, home_mult):
                try:
                    vals.append(float(_v))
                except (TypeError, ValueError):
                    pass

            combined = sum(vals) / len(vals) if vals else None
            profile = _ou_batter_pressure_profile(combined)

            return {
                "away_mult": away_mult,
                "home_mult": home_mult,
                "combined": combined,
                "profile": profile,
                "away_pop": away_vs_home.get("pop"),
                "home_pop": home_vs_away.get("pop"),
                "away_hand": away_hand,
                "home_hand": home_hand,
            }

        def _ou_batter_pressure_support(side, profile):
            side = str(side or "").strip().upper()
            if side == "OVER":
                if profile == "HIGH_PRESSURE":
                    return "support", "over_supported_by_batter_pressure"
                if profile == "HIGH_PRESSURE_CAUTION":
                    return "caution", "over_caution_positive_vs_hand_pressure"
                if profile == "LOW_PRESSURE":
                    return "downgrade", "over_downgrade_low_vs_hand_pressure"
                if profile == "LOW_PRESSURE_CAUTION":
                    return "caution", "over_caution_low_vs_hand_pressure"
            elif side == "UNDER":
                if profile == "LOW_PRESSURE":
                    return "support", "under_supported_by_offense_suppression"
                if profile == "LOW_PRESSURE_CAUTION":
                    return "caution", "under_caution_light_offense_suppression"
                if profile == "HIGH_PRESSURE":
                    return "downgrade", "under_downgrade_batter_pressure_conflict"
                if profile == "HIGH_PRESSURE_CAUTION":
                    return "caution", "under_caution_high_vs_hand_pressure"
            return "neutral", "batter_pressure_neutral"

        def _ou_downgrade_grade_one_step(grade):
            grade_s = str(grade or "").strip()
            if grade_s == "A":
                return "B"
            if grade_s == "B":
                return "C"
            if grade_s == "C":
                return "Pass"
            return grade_s

        def _ou_batter_adjust_grade(grade, side, support, ou_edge, bet_line,
                                    reliever_depth_risk=False, high_high_bullpen_risk=False):
            grade_s = str(grade or "").strip()
            side_s = str(side or "").strip().upper()

            # Batter pressure cannot rescue existing risk conditions.
            if grade_s == "Risk":
                return grade_s

            if support == "downgrade":
                return _ou_downgrade_grade_one_step(grade_s)

            if support != "support":
                return grade_s

            # Conservative support: strengthen review grades, but do not force A.
            if grade_s == "C":
                return "B"

            # Allow only UNDER Pass -> C when edge is meaningful, total is not the
            # fragile 7.5-under zone, and bullpen/depth risk is clean.
            try:
                edge_f = float(ou_edge)
            except (TypeError, ValueError):
                edge_f = None
            try:
                bet_line_f = float(bet_line)
            except (TypeError, ValueError):
                bet_line_f = None

            if (
                grade_s == "Pass"
                and side_s == "UNDER"
                and edge_f is not None
                and edge_f <= -0.50
                and (bet_line_f is None or bet_line_f > 7.5)
                and not reliever_depth_risk
                and not high_high_bullpen_risk
            ):
                return "C"

            return grade_s

        ou_over_profile_board_cols = [
            "Game_Date", "Datetime", "Game", "Venue",
            "Prediction", "OU_Side", "Projected_Total", "Bet_Line",
            "OU_Edge", "OU_Confidence", "OU_Fired", "Fired_Play",
            "No_Fire_OU_Reason", "Total_Is_Real", "Projection_Cap_Flag",
            "Data_Quality_Flag", "Trigger_Tags", "Model_Notes",
            "OU_Implied_Prob_Pick", "OU_True_Prob_Pick", "OU_Prob_Edge",
            "OU_Prob_Edge_Side", "OU_Prob_Method", "OU_Prob_Juice_Source",
            "OU_Prob_Book", "OU_Prob_Over_Juice", "OU_Prob_Under_Juice",
            "OU_Prob_Calibration_Flag",
            "OU_Sharpness_OK", "OU_Sharp_Direction", "OU_Sharp_Gap",
            "SHARP_OU_Delta", "OU_Pinnacle_Total", "OU_Retail_Total",
            "OU_Retail_Book",
            "Away_Pitcher", "Home_Pitcher",
            "Away_Starter_xERA", "Home_Starter_xERA",
            "Away_Starter_WHIP", "Home_Starter_WHIP",
            "Away_Starter_IP", "Home_Starter_IP",
            "Away_Starter_LowIP", "Home_Starter_LowIP",
            "Away_Reliever_Depth_Risk", "Home_Reliever_Depth_Risk",
            "Away_Reliever_Bad_xERA_Count", "Home_Reliever_Bad_xERA_Count",
            "Away_Reliever_Bad_WHIP_Count", "Home_Reliever_Bad_WHIP_Count",
            "Away_Reliever_Recent_Bad_Arm_Count", "Home_Reliever_Recent_Bad_Arm_Count",
            "Weather_Runs_Mult", "ENV_Is_Retractable", "ENV_Validation", "ENV_Conflict",
            "F5_Projected_Total", "F5_Away_Runs", "F5_Home_Runs",
            "F5_Market_Line", "F5_Edge", "F5_Model_Side", "F5_Starter_Lean",
            "F5_Eligible", "F5_Market_OK", "F5_Source", "F5_Book",
            "F5_Over_Juice", "F5_Under_Juice", "F5_No_Line_Reason",
            "OU_Over_Low_Total_Profile", "OU_Over_Prob_Sweet_Spot",
            "OU_Over_Thin_Edge_Profile", "OU_Over_Large_Edge_Risk",
            "OU_Over_High_Total_Risk", "OU_Over_Profile_Bucket",
            "OU_Over_Profile_Grade",
            "F5_Over_Profile_Signal", "F5_Over_Preference",
            "F5_Over_Profile_Reason", "F5_Over_Profile_Grade",
        ]
        ou_over_profile_board_rows = []
        for _r in eligible_export:
            _ou_side = str(_r.get("OU_Side", _r.get("Side", "")) or "").strip()
            _prediction = str(_r.get("Prediction", "") or "")
            _is_over_row = (
                _ou_side.lower() == "over"
                or "OVER" in _prediction.upper()
            )
            if not _is_over_row:
                continue

            _bet_line = _ou_over_profile_float(_r.get("Bet_Line"))
            _ou_edge = _ou_over_profile_float(_r.get("OU_Edge"))
            _prob_edge = _ou_over_profile_float(_r.get("OU_Prob_Edge"))
            _f5_projected = _ou_over_profile_float(_r.get("F5_Projected_Total"))
            _f5_line = _ou_over_profile_float(_r.get("F5_Market_Line"))
            _f5_edge = ""
            _f5_edge_num = None
            if _f5_projected is not None and _f5_line is not None:
                _f5_edge_num = round(_f5_projected - _f5_line, 2)
                _f5_edge = _f5_edge_num

            _total_is_real = _ou_over_profile_bool(_r.get("Total_Is_Real"))
            _clean_data = _ou_over_profile_clean_data(_r.get("Data_Quality_Flag"))
            _away_reliever_risk = str(_r.get("Away_Reliever_Depth_Risk", "") or "").strip().lower()
            _home_reliever_risk = str(_r.get("Home_Reliever_Depth_Risk", "") or "").strip().lower()
            _f5_market_ok = _ou_over_profile_bool(_r.get("F5_Market_OK"))
            _f5_eligible = _ou_over_profile_bool(_r.get("F5_Eligible"))
            _f5_model_side = str(_r.get("F5_Model_Side", "") or "").strip().upper()
            _f5_starter_lean = str(_r.get("F5_Starter_Lean", "") or "").strip().upper()

            _low_total_profile = bool(
                _is_over_row
                and _total_is_real
                and _bet_line is not None
                and _bet_line <= 7.5
            )
            _prob_sweet_spot = bool(
                _is_over_row
                and _prob_edge is not None
                and 0.05 <= _prob_edge <= 0.10
            )
            _thin_edge_profile = bool(
                _is_over_row
                and _ou_edge is not None
                and 0.0 < _ou_edge < 1.0
            )
            _large_edge_risk = bool(
                _is_over_row
                and _ou_edge is not None
                and 1.0 <= _ou_edge < 2.0
            )
            _high_total_risk = bool(
                _is_over_row
                and _bet_line is not None
                and _bet_line >= 9.0
            )

            _bucket_tags = []
            if _low_total_profile:
                _bucket_tags.append("low_total_over")
            if _prob_sweet_spot:
                _bucket_tags.append("prob_sweet_spot_over")
            if _thin_edge_profile:
                _bucket_tags.append("thin_edge_over")
            if _large_edge_risk:
                _bucket_tags.append("large_edge_risk_over")
            if _high_total_risk:
                _bucket_tags.append("high_total_risk_over")

            _batter_pressure = _ou_batter_pressure_for_row(_r)
            _batter_pressure_support, _batter_pressure_tag = _ou_batter_pressure_support(
                "OVER",
                _batter_pressure.get("profile"),
            )
            if _batter_pressure_tag != "batter_pressure_neutral":
                _bucket_tags.append(_batter_pressure_tag)

            _profile_bucket = "|".join(_bucket_tags) if _bucket_tags else "standard_over"

            if _clean_data and _low_total_profile and _prob_sweet_spot:
                _profile_grade = "A"
            elif _clean_data and (_low_total_profile or _prob_sweet_spot):
                _profile_grade = "B"
            elif (
                _clean_data
                and _thin_edge_profile
                and _prob_edge is not None
                and _prob_edge >= 0.05
            ):
                _profile_grade = "C"
            elif _large_edge_risk or _high_total_risk:
                _profile_grade = "Risk"
            else:
                _profile_grade = "Pass"

            _profile_grade = _ou_batter_adjust_grade(
                _profile_grade,
                "OVER",
                _batter_pressure_support,
                _ou_edge,
                _bet_line,
            )

            _f5_signal = bool(
                _is_over_row
                and _f5_market_ok
                and _f5_eligible
                and _f5_edge_num is not None
                and _f5_edge_num > 0
            )
            _bullpen_variance = _away_reliever_risk == "high" or _home_reliever_risk == "high"
            _f5_preference = bool(
                _f5_signal
                and (
                    _high_total_risk
                    or _large_edge_risk
                    or _bullpen_variance
                )
            )
            _f5_reason_tags = []
            if _f5_edge_num is not None and _f5_edge_num > 0:
                _f5_reason_tags.append("f5_positive_edge")
            if _f5_model_side == "OVER":
                _f5_reason_tags.append("f5_model_side_over")
            if "OVER" in _f5_starter_lean:
                _f5_reason_tags.append("starter_lean_over")
            if _large_edge_risk:
                _f5_reason_tags.append("full_game_large_edge_risk")
            if _high_total_risk:
                _f5_reason_tags.append("full_game_high_total_risk")
            if _bullpen_variance:
                _f5_reason_tags.append("bullpen_variance")
            if _clean_data and _low_total_profile:
                _f5_reason_tags.append("clean_low_total_over")

            if not _f5_market_ok or _f5_line is None:
                _f5_profile_grade = "Pass"
            elif (
                _f5_edge_num is not None
                and _f5_edge_num >= 0.50
                and _f5_model_side == "OVER"
                and "OVER" in _f5_starter_lean
            ):
                _f5_profile_grade = "A"
            elif (
                _f5_edge_num is not None
                and _f5_edge_num > 0
                and _f5_model_side == "OVER"
            ):
                _f5_profile_grade = "B"
            elif _f5_edge_num is not None and _f5_edge_num > 0:
                _f5_profile_grade = "C"
            else:
                _f5_profile_grade = "Watch"

            ou_over_profile_board_rows.append({
                "Game_Date": _r.get("Game_Date", ""),
                "Datetime": _r.get("Datetime", ""),
                "Game": _r.get("Game", ""),
                "Venue": _r.get("Venue", ""),
                "Prediction": _r.get("Prediction", ""),
                "OU_Side": _ou_side,
                "Projected_Total": _r.get("Projected_Total", ""),
                "Bet_Line": _r.get("Bet_Line", ""),
                "OU_Edge": _r.get("OU_Edge", ""),
                "OU_Confidence": _r.get("OU_Confidence", ""),
                "OU_Fired": _r.get("OU_Fired", ""),
                "Fired_Play": _r.get("Fired_Play", ""),
                "No_Fire_OU_Reason": _r.get("No_Fire_OU_Reason", ""),
                "Total_Is_Real": _r.get("Total_Is_Real", ""),
                "Projection_Cap_Flag": _r.get("Projection_Cap_Flag", ""),
                "Data_Quality_Flag": _r.get("Data_Quality_Flag", ""),
                "Trigger_Tags": _r.get("Trigger_Tags", ""),
                "Model_Notes": _r.get("Model_Notes", ""),
                "OU_Implied_Prob_Pick": _r.get("OU_Implied_Prob_Pick", ""),
                "OU_True_Prob_Pick": _r.get("OU_True_Prob_Pick", ""),
                "OU_Prob_Edge": _r.get("OU_Prob_Edge", ""),
                "OU_Prob_Edge_Side": _r.get("OU_Prob_Edge_Side", ""),
                "OU_Prob_Method": _r.get("OU_Prob_Method", ""),
                "OU_Prob_Juice_Source": _r.get("OU_Prob_Juice_Source", ""),
                "OU_Prob_Book": _r.get("OU_Prob_Book", ""),
                "OU_Prob_Over_Juice": _r.get("OU_Prob_Over_Juice", ""),
                "OU_Prob_Under_Juice": _r.get("OU_Prob_Under_Juice", ""),
                "OU_Prob_Calibration_Flag": _r.get("OU_Prob_Calibration_Flag", ""),
                "OU_Sharpness_OK": _r.get("OU_Sharpness_OK", ""),
                "OU_Sharp_Direction": _r.get("OU_Sharp_Direction", ""),
                "OU_Sharp_Gap": _r.get("OU_Sharp_Gap", ""),
                "SHARP_OU_Delta": _r.get("SHARP_OU_Delta", ""),
                "OU_Pinnacle_Total": _r.get("OU_Pinnacle_Total", ""),
                "OU_Retail_Total": _r.get("OU_Retail_Total", ""),
                "OU_Retail_Book": _r.get("OU_Retail_Book", ""),
                "Away_Pitcher": _r.get("Away_Pitcher", ""),
                "Home_Pitcher": _r.get("Home_Pitcher", ""),
                "Away_Starter_xERA": _r.get("Away_Starter_xERA", ""),
                "Home_Starter_xERA": _r.get("Home_Starter_xERA", ""),
                "Away_Starter_WHIP": _r.get("Away_Starter_WHIP", ""),
                "Home_Starter_WHIP": _r.get("Home_Starter_WHIP", ""),
                "Away_Starter_IP": _r.get("Away_Starter_IP", ""),
                "Home_Starter_IP": _r.get("Home_Starter_IP", ""),
                "Away_Starter_LowIP": _r.get("Away_Starter_LowIP", ""),
                "Home_Starter_LowIP": _r.get("Home_Starter_LowIP", ""),
                "Away_Reliever_Depth_Risk": _r.get("Away_Reliever_Depth_Risk", ""),
                "Home_Reliever_Depth_Risk": _r.get("Home_Reliever_Depth_Risk", ""),
                "Away_Reliever_Bad_xERA_Count": _r.get("Away_Reliever_Bad_xERA_Count", ""),
                "Home_Reliever_Bad_xERA_Count": _r.get("Home_Reliever_Bad_xERA_Count", ""),
                "Away_Reliever_Bad_WHIP_Count": _r.get("Away_Reliever_Bad_WHIP_Count", ""),
                "Home_Reliever_Bad_WHIP_Count": _r.get("Home_Reliever_Bad_WHIP_Count", ""),
                "Away_Reliever_Recent_Bad_Arm_Count": _r.get("Away_Reliever_Recent_Bad_Arm_Count", ""),
                "Home_Reliever_Recent_Bad_Arm_Count": _r.get("Home_Reliever_Recent_Bad_Arm_Count", ""),
                "Weather_Runs_Mult": _r.get("Weather_Runs_Mult", ""),
                "ENV_Is_Retractable": _r.get("ENV_Is_Retractable", ""),
                "ENV_Validation": _r.get("ENV_Validation", ""),
                "ENV_Conflict": _r.get("ENV_Conflict", ""),
                "F5_Projected_Total": _r.get("F5_Projected_Total", ""),
                "F5_Away_Runs": _r.get("F5_Away_Runs", ""),
                "F5_Home_Runs": _r.get("F5_Home_Runs", ""),
                "F5_Market_Line": _r.get("F5_Market_Line", ""),
                "F5_Edge": _f5_edge,
                "F5_Model_Side": _r.get("F5_Model_Side", ""),
                "F5_Starter_Lean": _r.get("F5_Starter_Lean", ""),
                "F5_Eligible": _r.get("F5_Eligible", ""),
                "F5_Market_OK": _r.get("F5_Market_OK", ""),
                "F5_Source": _r.get("F5_Source", ""),
                "F5_Book": _r.get("F5_Book", ""),
                "F5_Over_Juice": _r.get("F5_Over_Juice", ""),
                "F5_Under_Juice": _r.get("F5_Under_Juice", ""),
                "F5_No_Line_Reason": _r.get("F5_No_Line_Reason", ""),
                "OU_Over_Low_Total_Profile": _low_total_profile,
                "OU_Over_Prob_Sweet_Spot": _prob_sweet_spot,
                "OU_Over_Thin_Edge_Profile": _thin_edge_profile,
                "OU_Over_Large_Edge_Risk": _large_edge_risk,
                "OU_Over_High_Total_Risk": _high_total_risk,
                "OU_Over_Profile_Bucket": _profile_bucket,
                "OU_Over_Profile_Grade": _profile_grade,
                "F5_Over_Profile_Signal": _f5_signal,
                "F5_Over_Preference": _f5_preference,
                "F5_Over_Profile_Reason": "|".join(_f5_reason_tags),
                "F5_Over_Profile_Grade": _f5_profile_grade,
            })
        ou_over_profile_board_path = archive_output_path("ou_over_profile_board", archive_date)
        ou_over_profile_board_df = pd.DataFrame(
            ou_over_profile_board_rows,
            columns=ou_over_profile_board_cols,
        )
        ou_over_profile_board_df.to_csv(ou_over_profile_board_path, index=False)
        print(
            f"💾 Saved {len(ou_over_profile_board_rows)} OVER profile row(s) "
            f"→ {ou_over_profile_board_path}"
        )
        send_telegram_file(
            ou_over_profile_board_path,
            caption=f"📊 Over Gang OVER profile board — {datetime.now().strftime('%b %d')}",
        )

        def _ou_under_profile_float(v):
            try:
                if v is None or str(v).strip() == "":
                    return None
                f = float(v)
                if f != f:
                    return None
                return f
            except (TypeError, ValueError):
                return None

        def _ou_under_profile_bool(v) -> bool:
            return str(v).strip().lower() in ("true", "1", "yes")

        def _ou_under_profile_clean_data(v) -> bool:
            if v is None:
                return True
            s = str(v).strip()
            return s == "" or s.lower() == "nan"

        ou_under_profile_board_cols = [
            "Game_Date", "Datetime", "Game", "Venue",
            "Prediction", "OU_Side", "Projected_Total", "Bet_Line",
            "OU_Edge", "OU_Confidence", "OU_Fired", "Fired_Play",
            "No_Fire_OU_Reason", "Total_Is_Real", "Projection_Cap_Flag",
            "Data_Quality_Flag", "Trigger_Tags", "Model_Notes",
            "OU_Implied_Prob_Pick", "OU_True_Prob_Pick", "OU_Prob_Edge",
            "OU_Prob_Edge_Side", "OU_Prob_Method", "OU_Prob_Juice_Source",
            "OU_Prob_Book", "OU_Prob_Over_Juice", "OU_Prob_Under_Juice",
            "OU_Prob_Calibration_Flag",
            "OU_Sharpness_OK", "OU_Sharp_Direction", "OU_Sharp_Gap",
            "SHARP_OU_Delta", "OU_Pinnacle_Total", "OU_Retail_Total",
            "OU_Retail_Book",
            "Away_Pitcher", "Home_Pitcher",
            "Away_Starter_xERA", "Home_Starter_xERA",
            "Away_Starter_WHIP", "Home_Starter_WHIP",
            "Away_Starter_IP", "Home_Starter_IP",
            "Away_Starter_LowIP", "Home_Starter_LowIP",
            "Away_Reliever_Depth_Risk", "Home_Reliever_Depth_Risk",
            "Away_Reliever_Bad_xERA_Count", "Home_Reliever_Bad_xERA_Count",
            "Away_Reliever_Bad_WHIP_Count", "Home_Reliever_Bad_WHIP_Count",
            "Away_Reliever_Recent_Bad_Arm_Count", "Home_Reliever_Recent_Bad_Arm_Count",
            "OU_Full_Game_Under_Reliever_Depth_Risk", "OU_Under_Reliever_Depth_Block",
            "Weather_Runs_Mult", "ENV_Is_Retractable", "ENV_Validation", "ENV_Conflict",
            "F5_Projected_Total", "F5_Away_Runs", "F5_Home_Runs",
            "F5_Market_Line", "F5_Edge", "F5_Model_Side", "F5_Starter_Lean",
            "F5_Eligible", "F5_Market_OK", "F5_Source", "F5_Book",
            "F5_Over_Juice", "F5_Under_Juice", "F5_No_Line_Reason",
            "OU_Under_Prob_Support", "OU_Under_Low_Total_Risk",
            "OU_Under_High_Total_Cushion", "OU_Under_Mid_Edge_Risk",
            "OU_Under_Large_Edge_Profile", "OU_Under_Reliever_Depth_Risk_Profile",
            "OU_Under_HighHigh_Bullpen_Risk", "OU_Under_Profile_Bucket",
            "OU_Under_Profile_Grade",
            "F5_Under_Profile_Signal", "F5_Under_Preference",
            "F5_Under_Profile_Reason", "F5_Under_Profile_Grade",
        ]
        ou_under_profile_board_rows = []
        for _r in eligible_export:
            _ou_side = str(_r.get("OU_Side", _r.get("Side", "")) or "").strip()
            _prediction = str(_r.get("Prediction", "") or "")
            _is_under_row = (
                _ou_side.lower() == "under"
                or "UNDER" in _prediction.upper()
            )
            if not _is_under_row:
                continue

            _bet_line = _ou_under_profile_float(_r.get("Bet_Line"))
            _ou_edge = _ou_under_profile_float(_r.get("OU_Edge"))
            _prob_edge = _ou_under_profile_float(_r.get("OU_Prob_Edge"))
            _f5_projected = _ou_under_profile_float(_r.get("F5_Projected_Total"))
            _f5_line = _ou_under_profile_float(_r.get("F5_Market_Line"))
            _f5_edge = ""
            _f5_edge_num = None
            if _f5_projected is not None and _f5_line is not None:
                _f5_edge_num = round(_f5_projected - _f5_line, 2)
                _f5_edge = _f5_edge_num

            _clean_data = _ou_under_profile_clean_data(_r.get("Data_Quality_Flag"))
            _away_reliever_risk = str(_r.get("Away_Reliever_Depth_Risk", "") or "").strip().lower()
            _home_reliever_risk = str(_r.get("Home_Reliever_Depth_Risk", "") or "").strip().lower()
            _f5_market_ok = _ou_under_profile_bool(_r.get("F5_Market_OK"))
            _f5_eligible = _ou_under_profile_bool(_r.get("F5_Eligible"))
            _f5_model_side = str(_r.get("F5_Model_Side", "") or "").strip().upper()
            _f5_starter_lean = str(_r.get("F5_Starter_Lean", "") or "").strip().upper()

            _prob_support = bool(_is_under_row and _prob_edge is not None and _prob_edge >= 0.05)
            _low_total_risk = bool(_is_under_row and _bet_line is not None and _bet_line <= 7.5)
            _high_total_cushion = bool(_is_under_row and _bet_line is not None and _bet_line >= 9.0)
            _mid_edge_risk = bool(
                _is_under_row
                and _ou_edge is not None
                and 1.5 <= abs(_ou_edge) < 2.0
            )
            _large_edge_profile = bool(
                _is_under_row
                and _ou_edge is not None
                and abs(_ou_edge) >= 2.0
            )
            _reliever_depth_risk_profile = bool(
                _is_under_row
                and _ou_under_profile_bool(_r.get("OU_Full_Game_Under_Reliever_Depth_Risk"))
            )
            _high_high_bullpen_risk = bool(
                _is_under_row
                and _away_reliever_risk == "high"
                and _home_reliever_risk == "high"
            )

            _bucket_tags = []
            if _prob_support:
                _bucket_tags.append("prob_support_under")
            if _low_total_risk:
                _bucket_tags.append("low_total_under_risk")
            if _high_total_cushion:
                _bucket_tags.append("high_total_under_cushion")
            if _mid_edge_risk:
                _bucket_tags.append("mid_edge_under_risk")
            if _large_edge_profile:
                _bucket_tags.append("large_edge_under")
            if _reliever_depth_risk_profile:
                _bucket_tags.append("reliever_depth_risk_under")
            if _high_high_bullpen_risk:
                _bucket_tags.append("high_high_bullpen_risk_under")

            _batter_pressure = _ou_batter_pressure_for_row(_r)
            _batter_pressure_support, _batter_pressure_tag = _ou_batter_pressure_support(
                "UNDER",
                _batter_pressure.get("profile"),
            )
            if _batter_pressure_tag != "batter_pressure_neutral":
                _bucket_tags.append(_batter_pressure_tag)

            _profile_bucket = "|".join(_bucket_tags) if _bucket_tags else "standard_under"

            if (
                _clean_data
                and _prob_support
                and _high_total_cushion
                and not _reliever_depth_risk_profile
                and not _high_high_bullpen_risk
            ):
                _profile_grade = "A"
            elif (
                _clean_data
                and _prob_support
                and not _reliever_depth_risk_profile
                and not _high_high_bullpen_risk
            ):
                _profile_grade = "B"
            elif (
                _clean_data
                and _large_edge_profile
                and not _reliever_depth_risk_profile
                and not _high_high_bullpen_risk
            ):
                _profile_grade = "C"
            elif (
                _reliever_depth_risk_profile
                or _high_high_bullpen_risk
                or _mid_edge_risk
                or _low_total_risk
            ):
                _profile_grade = "Risk"
            else:
                _profile_grade = "Pass"

            _profile_grade = _ou_batter_adjust_grade(
                _profile_grade,
                "UNDER",
                _batter_pressure_support,
                _ou_edge,
                _bet_line,
                reliever_depth_risk=_reliever_depth_risk_profile,
                high_high_bullpen_risk=_high_high_bullpen_risk,
            )

            _f5_signal = bool(
                _is_under_row
                and _f5_market_ok
                and _f5_eligible
                and _f5_edge_num is not None
                and _f5_edge_num < 0
            )
            _bullpen_variance = _away_reliever_risk == "high" or _home_reliever_risk == "high"
            _f5_preference = bool(
                _f5_signal
                and (
                    _reliever_depth_risk_profile
                    or _high_high_bullpen_risk
                    or _bullpen_variance
                    or _mid_edge_risk
                    or _low_total_risk
                )
            )
            _f5_reason_tags = []
            if _f5_edge_num is not None and _f5_edge_num < 0:
                _f5_reason_tags.append("f5_negative_edge")
            if _f5_model_side == "UNDER":
                _f5_reason_tags.append("f5_model_side_under")
            if "UNDER" in _f5_starter_lean:
                _f5_reason_tags.append("starter_lean_under")
            if _reliever_depth_risk_profile:
                _f5_reason_tags.append("full_game_under_reliever_depth_risk")
            if _high_high_bullpen_risk:
                _f5_reason_tags.append("high_high_bullpen_risk")
            if _bullpen_variance:
                _f5_reason_tags.append("bullpen_variance")
            if _mid_edge_risk:
                _f5_reason_tags.append("mid_edge_under_risk")
            if _low_total_risk:
                _f5_reason_tags.append("low_total_under_risk")
            if _prob_support:
                _f5_reason_tags.append("prob_edge_support")

            if not _f5_market_ok or _f5_line is None:
                _f5_profile_grade = "Pass"
            elif (
                _f5_edge_num is not None
                and _f5_edge_num <= -0.50
                and _f5_model_side == "UNDER"
                and "UNDER" in _f5_starter_lean
            ):
                _f5_profile_grade = "A"
            elif (
                _f5_edge_num is not None
                and _f5_edge_num < 0
                and _f5_model_side == "UNDER"
            ):
                _f5_profile_grade = "B"
            elif _f5_edge_num is not None and _f5_edge_num < 0:
                _f5_profile_grade = "C"
            else:
                _f5_profile_grade = "Watch"

            ou_under_profile_board_rows.append({
                "Game_Date": _r.get("Game_Date", ""),
                "Datetime": _r.get("Datetime", ""),
                "Game": _r.get("Game", ""),
                "Venue": _r.get("Venue", ""),
                "Prediction": _r.get("Prediction", ""),
                "OU_Side": _ou_side,
                "Projected_Total": _r.get("Projected_Total", ""),
                "Bet_Line": _r.get("Bet_Line", ""),
                "OU_Edge": _r.get("OU_Edge", ""),
                "OU_Confidence": _r.get("OU_Confidence", ""),
                "OU_Fired": _r.get("OU_Fired", ""),
                "Fired_Play": _r.get("Fired_Play", ""),
                "No_Fire_OU_Reason": _r.get("No_Fire_OU_Reason", ""),
                "Total_Is_Real": _r.get("Total_Is_Real", ""),
                "Projection_Cap_Flag": _r.get("Projection_Cap_Flag", ""),
                "Data_Quality_Flag": _r.get("Data_Quality_Flag", ""),
                "Trigger_Tags": _r.get("Trigger_Tags", ""),
                "Model_Notes": _r.get("Model_Notes", ""),
                "OU_Implied_Prob_Pick": _r.get("OU_Implied_Prob_Pick", ""),
                "OU_True_Prob_Pick": _r.get("OU_True_Prob_Pick", ""),
                "OU_Prob_Edge": _r.get("OU_Prob_Edge", ""),
                "OU_Prob_Edge_Side": _r.get("OU_Prob_Edge_Side", ""),
                "OU_Prob_Method": _r.get("OU_Prob_Method", ""),
                "OU_Prob_Juice_Source": _r.get("OU_Prob_Juice_Source", ""),
                "OU_Prob_Book": _r.get("OU_Prob_Book", ""),
                "OU_Prob_Over_Juice": _r.get("OU_Prob_Over_Juice", ""),
                "OU_Prob_Under_Juice": _r.get("OU_Prob_Under_Juice", ""),
                "OU_Prob_Calibration_Flag": _r.get("OU_Prob_Calibration_Flag", ""),
                "OU_Sharpness_OK": _r.get("OU_Sharpness_OK", ""),
                "OU_Sharp_Direction": _r.get("OU_Sharp_Direction", ""),
                "OU_Sharp_Gap": _r.get("OU_Sharp_Gap", ""),
                "SHARP_OU_Delta": _r.get("SHARP_OU_Delta", ""),
                "OU_Pinnacle_Total": _r.get("OU_Pinnacle_Total", ""),
                "OU_Retail_Total": _r.get("OU_Retail_Total", ""),
                "OU_Retail_Book": _r.get("OU_Retail_Book", ""),
                "Away_Pitcher": _r.get("Away_Pitcher", ""),
                "Home_Pitcher": _r.get("Home_Pitcher", ""),
                "Away_Starter_xERA": _r.get("Away_Starter_xERA", ""),
                "Home_Starter_xERA": _r.get("Home_Starter_xERA", ""),
                "Away_Starter_WHIP": _r.get("Away_Starter_WHIP", ""),
                "Home_Starter_WHIP": _r.get("Home_Starter_WHIP", ""),
                "Away_Starter_IP": _r.get("Away_Starter_IP", ""),
                "Home_Starter_IP": _r.get("Home_Starter_IP", ""),
                "Away_Starter_LowIP": _r.get("Away_Starter_LowIP", ""),
                "Home_Starter_LowIP": _r.get("Home_Starter_LowIP", ""),
                "Away_Reliever_Depth_Risk": _r.get("Away_Reliever_Depth_Risk", ""),
                "Home_Reliever_Depth_Risk": _r.get("Home_Reliever_Depth_Risk", ""),
                "Away_Reliever_Bad_xERA_Count": _r.get("Away_Reliever_Bad_xERA_Count", ""),
                "Home_Reliever_Bad_xERA_Count": _r.get("Home_Reliever_Bad_xERA_Count", ""),
                "Away_Reliever_Bad_WHIP_Count": _r.get("Away_Reliever_Bad_WHIP_Count", ""),
                "Home_Reliever_Bad_WHIP_Count": _r.get("Home_Reliever_Bad_WHIP_Count", ""),
                "Away_Reliever_Recent_Bad_Arm_Count": _r.get("Away_Reliever_Recent_Bad_Arm_Count", ""),
                "Home_Reliever_Recent_Bad_Arm_Count": _r.get("Home_Reliever_Recent_Bad_Arm_Count", ""),
                "OU_Full_Game_Under_Reliever_Depth_Risk": _r.get(
                    "OU_Full_Game_Under_Reliever_Depth_Risk", ""
                ),
                "OU_Under_Reliever_Depth_Block": _r.get(
                    "OU_Under_Reliever_Depth_Block", ""
                ),
                "Weather_Runs_Mult": _r.get("Weather_Runs_Mult", ""),
                "ENV_Is_Retractable": _r.get("ENV_Is_Retractable", ""),
                "ENV_Validation": _r.get("ENV_Validation", ""),
                "ENV_Conflict": _r.get("ENV_Conflict", ""),
                "F5_Projected_Total": _r.get("F5_Projected_Total", ""),
                "F5_Away_Runs": _r.get("F5_Away_Runs", ""),
                "F5_Home_Runs": _r.get("F5_Home_Runs", ""),
                "F5_Market_Line": _r.get("F5_Market_Line", ""),
                "F5_Edge": _f5_edge,
                "F5_Model_Side": _r.get("F5_Model_Side", ""),
                "F5_Starter_Lean": _r.get("F5_Starter_Lean", ""),
                "F5_Eligible": _r.get("F5_Eligible", ""),
                "F5_Market_OK": _r.get("F5_Market_OK", ""),
                "F5_Source": _r.get("F5_Source", ""),
                "F5_Book": _r.get("F5_Book", ""),
                "F5_Over_Juice": _r.get("F5_Over_Juice", ""),
                "F5_Under_Juice": _r.get("F5_Under_Juice", ""),
                "F5_No_Line_Reason": _r.get("F5_No_Line_Reason", ""),
                "OU_Under_Prob_Support": _prob_support,
                "OU_Under_Low_Total_Risk": _low_total_risk,
                "OU_Under_High_Total_Cushion": _high_total_cushion,
                "OU_Under_Mid_Edge_Risk": _mid_edge_risk,
                "OU_Under_Large_Edge_Profile": _large_edge_profile,
                "OU_Under_Reliever_Depth_Risk_Profile": _reliever_depth_risk_profile,
                "OU_Under_HighHigh_Bullpen_Risk": _high_high_bullpen_risk,
                "OU_Under_Profile_Bucket": _profile_bucket,
                "OU_Under_Profile_Grade": _profile_grade,
                "F5_Under_Profile_Signal": _f5_signal,
                "F5_Under_Preference": _f5_preference,
                "F5_Under_Profile_Reason": "|".join(_f5_reason_tags),
                "F5_Under_Profile_Grade": _f5_profile_grade,
            })
        ou_under_profile_board_path = archive_output_path("ou_under_profile_board", archive_date)
        ou_under_profile_board_df = pd.DataFrame(
            ou_under_profile_board_rows,
            columns=ou_under_profile_board_cols,
        )
        ou_under_profile_board_df.to_csv(ou_under_profile_board_path, index=False)
        print(
            f"💾 Saved {len(ou_under_profile_board_rows)} UNDER profile row(s) "
            f"→ {ou_under_profile_board_path}"
        )
        send_telegram_file(
            ou_under_profile_board_path,
            caption=f"📉 Over Gang UNDER profile board — {datetime.now().strftime('%b %d')}",
        )

        # NEW: readable pregame picks board CSV. Sibling output to
        # predictions_*.csv and client_predictions_*.csv; does not replace or
        # alter either. Uses the same eligible_export rows and the same
        # timestamp as the archive file. Renames internal `Prediction` to
        # `OU_Pick` for readability without touching the archive schema.

        def _picks_board_key(row):
            date = str(row.get("Game_Date", "") or "")[:10]
            game = str(row.get("Game", "") or "").strip()
            dt = str(row.get("Datetime", "") or "").strip()
            if dt:
                return (date, game, dt)
            game_num = str(row.get("Game_Num", "") or "").strip()
            return (date, game, game_num)

        def _build_profile_lookup(rows_or_df, fields):
            lookup = {}
            if rows_or_df is None:
                return lookup
            if hasattr(rows_or_df, "empty"):
                if rows_or_df.empty:
                    return lookup
                rows = rows_or_df.to_dict("records")
            else:
                rows = rows_or_df or []
            if not rows:
                return lookup
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = _picks_board_key(row)
                payload = {f: row.get(f, "") for f in fields}
                if key in lookup:
                    print(
                        f"[PICKS_BOARD] duplicate profile key while enriching picks board: {key}"
                    )
                lookup[key] = payload
            return lookup

        def _format_f5_pick(side, line):
            side = str(side or "").strip().upper()
            if side not in ("OVER", "UNDER"):
                return ""
            if line is None or not pd.notna(line):
                return ""
            line_str = str(line).strip()
            if line_str == "" or line_str.lower() == "nan":
                return ""
            try:
                return f"{side} {float(line):.1f}"
            except (TypeError, ValueError):
                return f"{side} {line_str}"

        over_profile_fields = [
            "OU_Over_Low_Total_Profile",
            "OU_Over_Prob_Sweet_Spot",
            "OU_Over_Thin_Edge_Profile",
            "OU_Over_Large_Edge_Risk",
            "OU_Over_High_Total_Risk",
            "OU_Over_Profile_Bucket",
            "OU_Over_Profile_Grade",
            "F5_Over_Profile_Signal",
            "F5_Over_Preference",
            "F5_Over_Profile_Reason",
            "F5_Over_Profile_Grade",
        ]
        under_profile_fields = [
            "OU_Under_Prob_Support",
            "OU_Under_Low_Total_Risk",
            "OU_Under_High_Total_Cushion",
            "OU_Under_Mid_Edge_Risk",
            "OU_Under_Large_Edge_Profile",
            "OU_Under_Reliever_Depth_Risk_Profile",
            "OU_Under_HighHigh_Bullpen_Risk",
            "OU_Under_Profile_Bucket",
            "OU_Under_Profile_Grade",
            "F5_Under_Profile_Signal",
            "F5_Under_Preference",
            "F5_Under_Profile_Reason",
            "F5_Under_Profile_Grade",
        ]
        over_profile_lookup = _build_profile_lookup(
            ou_over_profile_board_df, over_profile_fields
        )
        under_profile_lookup = _build_profile_lookup(
            ou_under_profile_board_df, under_profile_fields
        )

        picks_board_front_cols = [
            "Game_Date", "Game_Time_MT", "Venue", "Doubleheader", "Game",
            "Away_Pitcher", "Away_xERA", "Away_WHIP",
            "Home_Pitcher", "Home_xERA", "Home_WHIP",
        ]
        picks_board_starter_cols = [
            "Away_Starter_xERA", "Home_Starter_xERA",
            "Away_Starter_WHIP", "Home_Starter_WHIP",
            "Away_Starter_ERA_xERA_Gap", "Home_Starter_ERA_xERA_Gap",
        ]
        picks_board_ou_cols = [
            "OU_Pick", "OU_Confidence", "OU_Fired", "OU_Edge",
        ]
        picks_board_f5_cols = [
            "F5_Pick",
            "F5_Projected_Total",
            "F5_Market_Line",
            "F5_Edge",
            "F5_Edge_Side",
            "F5_Model_Side",
            "F5_Model_Edge_Conflict",
            "F5_Starter_Lean",
            "F5_Eligible",
            "F5_Market_OK",
            "F5_Source",
            "F5_Book",
            "F5_Over_Juice",
            "F5_Under_Juice",
            "F5_No_Line_Reason",
        ]
        picks_board_daily_analysis_cols = [
            "Daily_OU_Profile_Side",
            "Daily_OU_Profile_Grade",
            "Daily_OU_Profile_Bucket",
            "Daily_OU_Profile_Read",
            "Daily_F5_Profile_Side",
            "Daily_F5_Profile_Pick",
            "Daily_F5_Profile_Grade",
            "Daily_F5_Profile_Reason",
            "Daily_F5_Analysis_Read",
        ]
        picks_board_ou_prob_cols = [
            "OU_Implied_Prob_Pick", "OU_True_Prob_Pick",
            "OU_Prob_Edge", "OU_Prob_Edge_Side", "OU_Prob_Method",
            "OU_Prob_Juice_Source", "OU_Prob_Book",
            "OU_Prob_Over_Juice", "OU_Prob_Under_Juice",
            "OU_Prob_Calibration_Flag",
            "No_Fire_OU_Reason", "Projection_Cap_Flag", "Total_Is_Real",
        ]
        picks_board_ou_over_profile_cols = [
            "OU_Over_Low_Total_Profile",
            "OU_Over_Prob_Sweet_Spot",
            "OU_Over_Thin_Edge_Profile",
            "OU_Over_Large_Edge_Risk",
            "OU_Over_High_Total_Risk",
            "OU_Over_Profile_Bucket",
            "OU_Over_Profile_Grade",
            "F5_Over_Profile_Signal",
            "F5_Over_Preference",
            "F5_Over_Profile_Reason",
            "F5_Over_Profile_Grade",
        ]
        picks_board_ou_under_profile_cols = [
            "OU_Under_Prob_Support",
            "OU_Under_Low_Total_Risk",
            "OU_Under_High_Total_Cushion",
            "OU_Under_Mid_Edge_Risk",
            "OU_Under_Large_Edge_Profile",
            "OU_Under_Reliever_Depth_Risk_Profile",
            "OU_Under_HighHigh_Bullpen_Risk",
            "OU_Under_Profile_Bucket",
            "OU_Under_Profile_Grade",
            "F5_Under_Profile_Signal",
            "F5_Under_Preference",
            "F5_Under_Profile_Reason",
            "F5_Under_Profile_Grade",
        ]
        picks_board_ml_cols = [
            "ML_Pick", "ML_Confidence", "ML_Fired", "ML_Edge",
            "ML_Kelly_Units", "No_Fire_ML_Reason",
        ]
        picks_board_market_cols = [
            "Fired_Play", "Play_Status", "Bettable",
            "Vegas_Line", "Odds_Line", "Bet_Line", "Projected_Total",
            "Odds_Book", "ML_Odds_Book", "Total_Line_Source", "Market_Source",
            "Trigger_Tags",
        ]
        picks_board_ou_sharp_cols = [
            "OU_Sharpness_Inputs_OK", "OU_Sharpness_OK",
            "OU_Sharp_Direction", "OU_Sharp_Gap",
            "OU_Sharp_Modifier", "OU_Pinnacle_Total", "OU_Retail_Total",
            "OU_Retail_Book", "SHARP_OU_Delta",
        ]
        picks_board_ml_quality_cols = [
            "ML_Market_OK", "ML_Market_Status",
            "ML_Sharpness_Inputs_OK", "ML_Sharpness_OK", "ML_Sharpness_Gate_Open",
            "ML_Exchange_Vs_Sharp_Gap",
        ]
        picks_board_reliever_cols = [
            "Away_Reliever_Depth_Risk", "Home_Reliever_Depth_Risk",
            "Away_Reliever_Bad_xERA_Count", "Home_Reliever_Bad_xERA_Count",
            "Away_Reliever_Bad_WHIP_Count", "Home_Reliever_Bad_WHIP_Count",
            "Away_Reliever_Recent_Bad_Arm_Count", "Home_Reliever_Recent_Bad_Arm_Count",
            "OU_Full_Game_Under_Reliever_Depth_Risk",
            "OU_Under_Reliever_Depth_Block",
        ]
        picks_board_tail_cols = ["Model_Notes", "Data_Quality_Flag"]
        picks_board_cols = (
            picks_board_front_cols
            + picks_board_starter_cols
            + picks_board_ou_cols
            + picks_board_f5_cols
            + picks_board_daily_analysis_cols
            + picks_board_ou_prob_cols
            + picks_board_ou_over_profile_cols
            + picks_board_ou_under_profile_cols
            + picks_board_ml_cols
            + picks_board_market_cols
            + picks_board_ou_sharp_cols
            + picks_board_ml_quality_cols
            + picks_board_reliever_cols
            + picks_board_tail_cols
        )
        _seen_picks_board_cols = set()
        _deduped_picks_board_cols = []
        _dup_picks_board_cols = []
        for _c in picks_board_cols:
            if _c in _seen_picks_board_cols:
                _dup_picks_board_cols.append(_c)
                continue
            _seen_picks_board_cols.add(_c)
            _deduped_picks_board_cols.append(_c)
        if _dup_picks_board_cols:
            print(
                f"[PICKS_BOARD] removed duplicate picks-board columns: {_dup_picks_board_cols}"
            )
        picks_board_cols = _deduped_picks_board_cols

        def _daily_blank(v):
            if v is None:
                return True
            try:
                if pd.isna(v):
                    return True
            except (TypeError, ValueError):
                pass
            s = str(v).strip()
            return s == "" or s.lower() in ("nan", "none", "null", "<na>")

        def _daily_clean(v):
            return "" if _daily_blank(v) else str(v).strip()

        def _daily_ou_side_from_pick(pick):
            p = str(pick or "").upper().strip()
            if p.startswith("LEAN OVER") or p.startswith("OVER"):
                return "OVER"
            if p.startswith("LEAN UNDER") or p.startswith("UNDER"):
                return "UNDER"
            return ""

        def _f5_pref_true(v):
            if v is True:
                return True
            s = str(v).strip().lower()
            return s in ("true", "1", "yes")

        def _daily_f5_profile_side(row):
            over_pref = _f5_pref_true(row.get("F5_Over_Preference"))
            under_pref = _f5_pref_true(row.get("F5_Under_Preference"))
            if over_pref and not under_pref:
                return "OVER"
            if under_pref and not over_pref:
                return "UNDER"
            return ""

        def _f5_edge_side_from_edge(edge):
            try:
                if edge is None or not pd.notna(edge):
                    return ""
                edge_f = float(edge)
            except (TypeError, ValueError):
                return ""
            if edge_f > 0:
                return "OVER"
            if edge_f < 0:
                return "UNDER"
            return "EVEN"

        picks_board_rows = []
        for _r in eligible_export:
            _row = dict(_r)
            _row["OU_Pick"] = _r.get("Prediction", "")
            _pb_key = _picks_board_key(_row)
            _row.update(over_profile_lookup.get(_pb_key, {}))
            _row.update(under_profile_lookup.get(_pb_key, {}))
            _f5_edge_val = _row.get("F5_Edge")
            _f5_edge_missing = (
                _f5_edge_val is None
                or _f5_edge_val == ""
                or (isinstance(_f5_edge_val, float) and pd.isna(_f5_edge_val))
            )
            if _f5_edge_missing:
                try:
                    _row["F5_Edge"] = round(
                        float(_row["F5_Projected_Total"]) - float(_row["F5_Market_Line"]),
                        2,
                    )
                except (TypeError, ValueError):
                    pass
            _row["F5_Edge_Side"] = _f5_edge_side_from_edge(_row.get("F5_Edge"))
            _row["F5_Pick"] = _format_f5_pick(
                _row.get("F5_Edge_Side", ""),
                _row.get("F5_Market_Line"),
            )
            _f5_model_side = str(_row.get("F5_Model_Side", "") or "").strip().upper()
            _f5_edge_side = str(_row.get("F5_Edge_Side", "") or "").strip().upper()
            _row["F5_Model_Edge_Conflict"] = bool(
                _f5_model_side in ("OVER", "UNDER")
                and _f5_edge_side in ("OVER", "UNDER")
                and _f5_model_side != _f5_edge_side
            )

            _ou_over_grade = _daily_clean(_row.get("OU_Over_Profile_Grade"))
            _ou_over_bucket = _daily_clean(_row.get("OU_Over_Profile_Bucket"))
            _ou_under_grade = _daily_clean(_row.get("OU_Under_Profile_Grade"))
            _ou_under_bucket = _daily_clean(_row.get("OU_Under_Profile_Bucket"))
            if _ou_over_grade or _ou_over_bucket:
                _daily_ou_side = "OVER"
                _daily_ou_grade = _ou_over_grade
                _daily_ou_bucket = _ou_over_bucket
            elif _ou_under_grade or _ou_under_bucket:
                _daily_ou_side = "UNDER"
                _daily_ou_grade = _ou_under_grade
                _daily_ou_bucket = _ou_under_bucket
            else:
                _daily_ou_side = _daily_ou_side_from_pick(_row.get("OU_Pick", ""))
                _daily_ou_grade = ""
                _daily_ou_bucket = ""
            _row["Daily_OU_Profile_Side"] = _daily_ou_side
            _row["Daily_OU_Profile_Grade"] = _daily_ou_grade
            _row["Daily_OU_Profile_Bucket"] = _daily_ou_bucket
            _daily_ou_read_parts = []
            if _daily_ou_side:
                _daily_ou_read_parts.append(_daily_ou_side)
            if _daily_ou_grade:
                _daily_ou_read_parts.append(f"Grade: {_daily_ou_grade}")
            if _daily_ou_bucket:
                _daily_ou_read_parts.append(f"Bucket: {_daily_ou_bucket}")
            _row["Daily_OU_Profile_Read"] = " | ".join(_daily_ou_read_parts)

            _daily_f5_side = _daily_f5_profile_side(_row)
            if _daily_f5_side == "OVER":
                _daily_f5_grade = _daily_clean(_row.get("F5_Over_Profile_Grade"))
                _daily_f5_reason = _daily_clean(_row.get("F5_Over_Profile_Reason"))
                _derived_f5_pick = _format_f5_pick("OVER", _row.get("F5_Market_Line"))
            elif _daily_f5_side == "UNDER":
                _daily_f5_grade = _daily_clean(_row.get("F5_Under_Profile_Grade"))
                _daily_f5_reason = _daily_clean(_row.get("F5_Under_Profile_Reason"))
                _derived_f5_pick = _format_f5_pick("UNDER", _row.get("F5_Market_Line"))
            else:
                _daily_f5_grade = ""
                _daily_f5_reason = ""
                _derived_f5_pick = ""
            _row["Daily_F5_Profile_Side"] = _daily_f5_side
            _row["Daily_F5_Profile_Grade"] = _daily_f5_grade
            _row["Daily_F5_Profile_Reason"] = _daily_f5_reason
            if not _daily_blank(_derived_f5_pick):
                _row["Daily_F5_Profile_Pick"] = _derived_f5_pick
            else:
                _existing_f5_pick_for_daily = _daily_clean(_row.get("F5_Pick"))
                _row["Daily_F5_Profile_Pick"] = _existing_f5_pick_for_daily
            _daily_f5_read_parts = []
            if not _daily_blank(_row.get("Daily_F5_Profile_Pick")):
                _daily_f5_read_parts.append(str(_row.get("Daily_F5_Profile_Pick")))
            if _daily_f5_grade:
                _daily_f5_read_parts.append(f"Profile Grade: {_daily_f5_grade}")
            if _daily_f5_reason:
                _daily_f5_read_parts.append(_daily_f5_reason.replace("|", ", "))
            _row["Daily_F5_Analysis_Read"] = " | ".join(_daily_f5_read_parts)

            picks_board_rows.append(_row)
        _f5_candidate_n = len(picks_board_rows)
        telegram_f5_alerts = []
        _seen_f5_telegram = set()
        for _f5_row in picks_board_rows:
            if not _is_f5_telegram_candidate(_f5_row):
                continue
            _f5_pick = (
                _alert_clean(_f5_row.get("Daily_F5_Profile_Pick"))
                or _alert_clean(_f5_row.get("F5_Pick"))
            )
            _f5_key = (str(_f5_row.get("Game", "")), _f5_pick)
            if _f5_key in _seen_f5_telegram:
                continue
            _seen_f5_telegram.add(_f5_key)
            telegram_f5_alerts.append(_f5_row)
        # Export-only OU/F5 decision board.
        # Purpose: one clean totals board for deciding full-game O/U vs F5.
        # No model math, fire logic, Telegram, ML, or K behavior changes.
        def _ou_f5_board_float(_v):
            try:
                if _v is None or _v == "":
                    return None
                return float(_v)
            except (TypeError, ValueError):
                return None

        def _ou_f5_board_bool(_v):
            return str(_v).strip().lower() in ("true", "1", "yes", "y")

        def _ou_prob_bucket(_v):
            _f = _ou_f5_board_float(_v)
            if _f is None:
                return ""
            if _f < 0:
                return "prob_edge_lt_0"
            if _f < 0.03:
                return "prob_edge_0_00_to_0_03"
            if _f < 0.05:
                return "prob_edge_0_03_to_0_05"
            if _f < 0.10:
                return "prob_edge_0_05_to_0_10"
            if _f < 0.15:
                return "prob_edge_0_10_to_0_15"
            if _f < 0.20:
                return "prob_edge_0_15_to_0_20"
            return "prob_edge_ge_0_20"

        def _ou_side_from_pick(_v):
            _s = str(_v or "").upper()
            if "OVER" in _s:
                return "OVER"
            if "UNDER" in _s:
                return "UNDER"
            return ""

        def _clean_text(_v):
            if _v is None:
                return ""
            _s = str(_v)
            if _s.lower() in ("nan", "none"):
                return ""
            return _s

        def _ou_f5_market_view(_row):
            _ou_fired = _ou_f5_board_bool(_row.get("OU_Fired"))
            _f5_pick = _clean_text(_row.get("Daily_F5_Profile_Pick") or _row.get("F5_Pick"))
            _f5_grade = _clean_text(_row.get("Daily_F5_Profile_Grade"))
            _f5_market_ok = _ou_f5_board_bool(_row.get("F5_Market_OK"))
            _prob_watch = _ou_f5_board_bool(_row.get("OU_Prob_Over_Sweet_Spot_Watch"))
            if _ou_fired:
                return "FULL_GAME_OU"
            if _prob_watch:
                return "WATCH_FULL_GAME_OVER"
            if _f5_market_ok and _f5_pick and _f5_grade in ("A", "B"):
                return "F5"
            return "WATCH"

        def _ou_f5_decision_reason(_row):
            _parts = []
            if _ou_f5_board_bool(_row.get("OU_Fired")):
                _parts.append("ou_fired")
            if _ou_f5_board_bool(_row.get("OU_Prob_Over_Sweet_Spot_Watch")):
                _parts.append("prob_over_sweet_spot_watch")
            _ou_bucket = _clean_text(_row.get("Daily_OU_Profile_Bucket"))
            if _ou_bucket:
                _parts.append(_ou_bucket)
            _f5_read = _clean_text(_row.get("Daily_F5_Analysis_Read"))
            if _f5_read:
                _parts.append("f5=" + _f5_read)
            _dq = _clean_text(_row.get("Data_Quality_Flag"))
            if _dq:
                _parts.append("dq=" + _dq)
            return " | ".join(_parts)

        ou_f5_decision_board_rows = []
        for _src in picks_board_rows:
            _row = dict(_src)
            _ou_pick = _clean_text(_row.get("OU_Pick") or _row.get("Prediction"))
            _ou_side = _ou_side_from_pick(_ou_pick)
            _ou_prob_edge = _ou_f5_board_float(_row.get("OU_Prob_Edge"))
            _ou_edge = _ou_f5_board_float(_row.get("OU_Edge"))
            _dq = _clean_text(_row.get("Data_Quality_Flag")).lower()
            _bucket_text = (
                _clean_text(_row.get("Daily_OU_Profile_Bucket")) + "|"
                + _clean_text(_row.get("OU_Over_Profile_Bucket")) + "|"
                + _clean_text(_row.get("OU_Under_Profile_Bucket")) + "|"
                + _clean_text(_row.get("Trigger_Tags"))
            ).lower()

            _prob_over_watch = bool(
                _ou_side == "OVER"
                and _ou_prob_edge is not None
                and 0.05 <= _ou_prob_edge < 0.10
                and _ou_edge is not None
                and 0.20 <= _ou_edge <= 0.60
                and "fallback_pitcher" not in _dq
                and "low_ip" not in _dq
                and "high_total_risk_over" not in _bucket_text
                and "large_edge_risk_over" not in _bucket_text
                and "over_caution_low_vs_hand_pressure" not in _bucket_text
                and "over_downgrade_low_vs_hand_pressure" not in _bucket_text
            )

            _row["OU_Prob_Bucket"] = _ou_prob_bucket(_row.get("OU_Prob_Edge"))
            _row["OU_Prob_Over_Sweet_Spot_Watch"] = _prob_over_watch
            _row["Preferred_Market"] = _ou_f5_market_view(_row)
            _row["Decision_Reason"] = _ou_f5_decision_reason(_row)

            ou_f5_decision_board_rows.append(_row)

        ou_f5_decision_board_cols = [
            "Game_Date", "Datetime", "Game_Status", "Venue", "Game",

            "Away_Pitcher", "Away_Starter_ERA", "Away_Starter_xERA",
            "Away_Starter_ERA_xERA_Gap", "Away_Starter_WHIP",
            "Home_Pitcher", "Home_Starter_ERA", "Home_Starter_xERA",
            "Home_Starter_ERA_xERA_Gap", "Home_Starter_WHIP",

            "Vegas_Line", "Base_Projected_Total", "Base_OU_Edge",
            "Raw_Projected_Total", "Raw_OU_Edge",
            "Projected_Total", "OU_Edge",
            "OU_Edge_Calibration_Factor", "OU_Edge_Calibration_Tag",
            "Run_Pressure_Adjustment", "Full_Picture_Adjustment_Raw",
            "Run_Pressure_Mode", "Run_Pressure_Reasons",
            "Away_Early_Run_Pressure", "Away_Early_Run_Suppression",
            "Away_Late_Run_Pressure", "Away_Late_Run_Suppression",
            "Home_Early_Run_Pressure", "Home_Early_Run_Suppression",
            "Home_Late_Run_Pressure", "Home_Late_Run_Suppression",
            "OU_Pick", "OU_Confidence", "OU_Fired",

            "OU_Implied_Prob_Pick", "OU_True_Prob_Pick",
            "OU_Prob_Edge", "OU_Prob_Bucket",
            "OU_Prob_Over_Sweet_Spot_Watch",

            "Daily_OU_Profile_Side", "Daily_OU_Profile_Grade",
            "Daily_OU_Profile_Bucket", "Daily_OU_Profile_Read",

            "F5_Pick", "Daily_F5_Profile_Pick",
            "F5_Market_Line", "F5_Projected_Total", "F5_Edge",
            "F5_Market_OK", "Daily_F5_Profile_Side",
            "Daily_F5_Profile_Grade", "Daily_F5_Analysis_Read",

            "Away_Reliever_Depth_Risk", "Home_Reliever_Depth_Risk",
            "OU_Full_Game_Under_Reliever_Depth_Risk",
            "OU_Under_HighHigh_Bullpen_Risk",
            "OU_Over_High_Total_Risk", "OU_Over_Large_Edge_Risk",

            "Data_Quality_Flag", "No_Fire_Reason", "Trigger_Tags",
            "Preferred_Market", "Decision_Reason",
        ]

        ou_f5_decision_board_df = pd.DataFrame(
            ou_f5_decision_board_rows,
            columns=ou_f5_decision_board_cols,
        )
        ou_f5_decision_board_path = archive_output_path("ou_f5_decision_board", archive_date)
        ou_f5_decision_board_df.to_csv(ou_f5_decision_board_path, index=False)
        print(
            f"💾 Saved {len(ou_f5_decision_board_rows)} OU/F5 decision-board row(s) "
            f"→ {ou_f5_decision_board_path}"
        )
        if os.path.exists(ou_f5_decision_board_path):
            send_telegram_file(
                ou_f5_decision_board_path,
                caption=f"📊 Over Gang OU/F5 decision board — {datetime.now().strftime('%b %d')}",
            )

        # Export-only ML decision board.
        # Purpose: one clean moneyline board with ML-specific context only.
        # No ML fire logic, Kelly, sharpness, O/U, F5, K, or model math changes.
        def _ml_board_clean(_v):
            if _v is None:
                return ""
            _s = str(_v)
            if _s.lower() in ("nan", "none", "null", "<na>"):
                return ""
            return _s

        def _ml_board_float(_v):
            try:
                if _v is None or str(_v).strip() == "":
                    return None
                _f = float(str(_v).replace("%", "").replace("u", "").strip())
                if _f != _f:
                    return None
                return _f
            except (TypeError, ValueError):
                return None

        def _ml_board_bool(_v):
            return str(_v).strip().lower() in ("true", "1", "yes", "y")

        def _ml_board_pick_side(_v):
            _s = str(_v or "").upper()
            if "HOME" in _s:
                return "home"
            if "AWAY" in _s:
                return "away"
            return ""

        def _ml_board_first_present(_row, _keys):
            for _k in _keys:
                _v = _row.get(_k)
                if _v is None:
                    continue
                _s = str(_v).strip()
                if _s and _s.lower() not in ("nan", "none", "null", "<na>"):
                    return _v
            return ""

        def _ml_board_picked_odds(_row):
            _side = str(_row.get("ML_Side") or "").strip().lower()
            if _side == "home":
                return _ml_board_first_present(
                    _row,
                    ["ML_Home_Odds", "Captured_ML_Home", "Home_ML", "ML_Home", "Home_Odds"],
                )
            if _side == "away":
                return _ml_board_first_present(
                    _row,
                    ["ML_Away_Odds", "Captured_ML_Away", "Away_ML", "ML_Away", "Away_Odds"],
                )
            return ""

        def _ml_board_short_favorite(_row):
            _odds = _ml_board_float(_ml_board_picked_odds(_row))
            if _odds is None:
                return False
            return -125 <= _odds <= -110

        def _ml_board_lane(_row):
            _ml_fired = _ml_board_bool(_row.get("ML_Fired"))
            _short_fav = _ml_board_short_favorite(_row)
            _market_ok = _ml_board_bool(_row.get("ML_Market_OK"))
            _sharp_ok = _ml_board_bool(_row.get("ML_Sharpness_OK")) or not _ml_board_bool(
                _row.get("ML_Sharpness_Inputs_OK")
            )
            if _ml_fired and _short_fav:
                return "ML_SHORT_FAVORITE_FIRE"
            if _ml_fired:
                return "ML_FIRE"
            if _short_fav and _market_ok and _sharp_ok:
                return "ML_SHORT_FAVORITE_WATCH"
            if not _market_ok:
                return "ML_MARKET_INCOMPLETE"
            return "ML_WATCH"

        def _ml_board_decision_reason(_row):
            _parts = []
            if _ml_board_bool(_row.get("ML_Fired")):
                _parts.append("ml_fired")
            _lane = _ml_board_clean(_row.get("ML_Lane"))
            if _lane:
                _parts.append(_lane)
            _edge = _ml_board_clean(_row.get("ML_Edge"))
            if _edge:
                _parts.append(f"edge={_edge}")
            _conf = _ml_board_clean(_row.get("ML_Confidence"))
            if _conf:
                _parts.append(f"conf={_conf}")
            _kelly = _ml_board_clean(_row.get("ML_Kelly_Units"))
            if _kelly:
                _parts.append(f"kelly={_kelly}")
            _market_status = _ml_board_clean(_row.get("ML_Market_Status"))
            if _market_status:
                _parts.append(f"market={_market_status}")
            _sharp_ok = _ml_board_clean(_row.get("ML_Sharpness_OK"))
            if _sharp_ok:
                _parts.append(f"sharp_ok={_sharp_ok}")
            _no_fire = _ml_board_clean(_row.get("No_Fire_ML_Reason"))
            if _no_fire and _no_fire.lower() not in ("nan", "none"):
                _parts.append(f"no_fire={_no_fire}")
            _dq = _ml_board_clean(_row.get("Data_Quality_Flag"))
            if _dq:
                _parts.append(f"dq={_dq}")
            return " | ".join(_parts)

        ml_decision_board_rows = []
        for _src in picks_board_rows:
            _row = dict(_src)
            _row["ML_Picked_Odds"] = _ml_board_picked_odds(_row)
            _row["ML_Short_Favorite_Flag"] = _ml_board_short_favorite(_row)
            _row["ML_Lane"] = _ml_board_lane(_row)
            _row["ML_Decision_Reason"] = _ml_board_decision_reason(_row)
            ml_decision_board_rows.append(_row)

        ml_decision_board_cols = [
            "Game_Date", "Datetime", "Game_Status", "Venue", "Game",

            "Away_Pitcher", "Away_Starter_ERA", "Away_Starter_xERA", "Away_Starter_WHIP",
            "Home_Pitcher", "Home_Starter_ERA", "Home_Starter_xERA", "Home_Starter_WHIP",

            "ML_Pick", "ML_Side", "ML_Bet_Type",
            "ML_Confidence", "ML_Edge", "ML_Value", "ML_Kelly_Units",
            "ML_Picked_Odds", "ML_Short_Favorite_Flag", "ML_Lane",

            "ML_Fired", "No_Fire_ML_Reason",
            "ML_Market_OK", "ML_Market_Status",
            "ML_Quality_Flag", "ML_Quality_Factor",

            "ML_Sharpness_Inputs_OK", "ML_Sharpness_OK",
            "ML_Sharpness_Gate_Open", "ML_Exchange_Vs_Sharp_Gap",
            "ML_Exchange_Present", "ML_Prophetx_Present", "ML_Pinnacle_Present",

            "Data_Quality_Flag", "Trigger_Tags", "ML_Decision_Reason",
        ]

        ml_decision_board_df = pd.DataFrame(
            ml_decision_board_rows,
            columns=ml_decision_board_cols,
        )
        ml_decision_board_path = archive_output_path("ml_decision_board", archive_date)
        ml_decision_board_df.to_csv(ml_decision_board_path, index=False)
        print(
            f"💾 Saved {len(ml_decision_board_rows)} ML decision-board row(s) "
            f"→ {ml_decision_board_path}"
        )
        if os.path.exists(ml_decision_board_path):
            send_telegram_file(
                ml_decision_board_path,
                caption=f"📈 Over Gang ML decision board — {datetime.now().strftime('%b %d')}",
            )

        picks_board_df = pd.DataFrame(picks_board_rows, columns=picks_board_cols)
        picks_board_path = archive_output_path("picks_board", archive_date)
        picks_board_df.to_csv(picks_board_path, index=False)
        print(f"💾 Saved {len(picks_board_rows)} picks-board row(s) → {picks_board_path}")
        # Legacy picks_board is still generated temporarily for compatibility,
        # but it is no longer sent to Telegram. Operator workflow is moving to
        # market-specific boards: OU/F5 decision board, ML decision board, K board.
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

    if telegram_f5_alerts:
        print(f"\n\U0001f6a8 Sending {len(telegram_f5_alerts)} F5 Telegram alert(s) (edge 0.75-1.00 or >= 1.50)...")
        for alert in telegram_f5_alerts:
            message = format_f5_alert(alert)
            if send_telegram_alert(message):
                print(f"\U0001f4e4 F5 alert sent for {alert['Game']}")
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
    print(f"  F5 Telegram alerts sent: {len(telegram_f5_alerts)} (candidates: {_f5_candidate_n})")
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
