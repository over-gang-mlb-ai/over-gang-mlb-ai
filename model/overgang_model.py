"""
OVER GANG MLB PREDICTOR v4.0 — True projection model

Projects expected runs per team (away + home), sums to a game total, then compares
to the Vegas total for edge, confidence, and recommended bet.

Inputs: starter xERA/WHIP, bullpen ERA/fatigue (IP_Week), park factors, velocity
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
import json
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
from rapidfuzz import fuzz, process
from functools import lru_cache
from statsapi import schedule
from pybaseball import pitching_stats, statcast_pitcher
import numpy as np
import subprocess
import sys
from pathlib import Path

from scrapers.velocity_tracker import VelocityTracker
velocity_tracker = VelocityTracker()
from core.public_betting_dummy import DUMMY_PUBLIC_BETTING
from core.public_betting_loader import load_public_betting_data
public_data = load_public_betting_data()
from core.public_betting_loader import split_game_key
from core.ml_predictor import get_team_ml_data, calculate_team_win_probability
from core.public_betting_loader import normalize_team_name
from core.kelly_utils import calculate_kelly_units
from core.odds_api import fetch_mlb_odds, get_game_odds
from core.sportsdataio import fetch_mlb_odds_by_date
from core.batters import Batters, LineupImpact, BATTER_DF
from model.data_manager import DataManager
manual_fallback_df = DataManager.load_manual_fallback_pitchers()

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
BULLPEN_FATIGUE_IP_CUTOFF = 12.0   # IP_Week above this = tired bullpen
BULLPEN_FATIGUE_RUNS_BOOST = 0.03  # +3% runs per 10 IP over cutoff (capped)
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

    if opponent_bullpen_ip_week > BULLPEN_FATIGUE_IP_CUTOFF:
        excess_ip = min(20.0, opponent_bullpen_ip_week - BULLPEN_FATIGUE_IP_CUTOFF)
        fatigue_mult = 1.0 + BULLPEN_FATIGUE_RUNS_BOOST * (excess_ip / 10.0)
        runs *= min(1.08, fatigue_mult)

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
        # ensure numeric
        def _num(v, default=0.0):
            try: return float(v)
            except: return default

        return {
            'ERA': _num(row.get('ERA'), 4.25),
            'IP_Week': _num(row.get('IP_Week'), 12.0),
            'Relievers': int(_num(row.get('Relievers'), 7)),
            'source': 'MLB Stats API'
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
    def get_vegas_line(home_team, away_team, odds_map=None):
        """
        Return (vegas_line_float, odds_info_dict).
        odds_info_dict has total_line, over_juice, under_juice, ml_home, ml_away, book.
        Uses The Odds API when odds_map provided and match found; else CSV; else 8.5 + default juice/book.
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
            row = get_game_odds(away_team, home_team, odds_map)
            raw_line = row.get("total_line")
            if raw_line is None or raw_line == "":
                line = 8.5
            else:
                try:
                    line = float(raw_line)
                except (TypeError, ValueError):
                    line = 8.5
            info = dict(row)
            match_found = lookup_key in odds_map
            book_empty = not (row.get("book") or "").strip()
            book_scrambled = (row.get("book") or "").strip().lower() == "scrambled"
            has_real_total = (not book_empty) and (raw_line is not None and raw_line != "") and (not book_scrambled)
            is_fallback_line = not has_real_total
            if match_found:
                source = "fallback (scrambled book)" if book_scrambled else ("8.5 fallback" if is_fallback_line else "Odds API")
            else:
                source = "8.5 fallback (no match in odds_map)"
            info["_source"] = source
            info["_lookup_key"] = lookup_key
            info["_match_found"] = match_found
            info["_has_real_total"] = bool(has_real_total)
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

class VelocityTracker:
    @staticmethod
    @lru_cache(maxsize=100)
    def get_velocity_drop(pitcher_name):
        try:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            data = statcast_pitcher(start_date, end_date, pitcher_name)
            if data is None or len(data) < 10:
                return 0
            recent_avg = data['release_speed'].tail(3).mean()
            season_peak = data['release_speed'].quantile(0.75)
            return recent_avg - season_peak
        except Exception as e:
            print(f"⚠️ Velocity error for {pitcher_name}: {e}")
            return 0

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
    over_boost, _ = park_factors

    # ---------- Project runs for each team ----------
    # Away offense faces home pitcher + home bullpen; away_offense_mult from Batters.offense_vs_hand_dict(away_team vs home_hand)
    away_runs = project_team_runs(
        opponent_starter_xera=home_xera,
        opponent_starter_whip=home_whip,
        opponent_bullpen_era=bullpen_home_era,
        opponent_bullpen_ip_week=bullpen_home_ip_week,
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

    total_open = safe_float(public_data.get("Total_Open"), default=vegas_line)
    total_current = safe_float(public_data.get("Total_Current"), default=total_open)

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

def format_alert(game_data: dict) -> str:
    try:
        game_utc_time = datetime.strptime(game_data['Datetime'], "%Y-%m-%dT%H:%M:%SZ")
        mt_time = game_utc_time.replace(tzinfo=utc).astimezone(timezone("US/Mountain"))
        formatted_time = mt_time.strftime('%I:%M %p MT')
    except:
        formatted_time = "TBD"

    velo_away = game_data.get('VeloDrop_Away', '?')
    velo_home = game_data.get('VeloDrop_Home', '?')
    ou_over = game_data.get('ou_bets_pct_over', '?')
    ou_under = game_data.get('ou_bets_pct_under', '?')
    ml_home = game_data.get('ml_bets_pct_home', '?')
    ml_away = game_data.get('ml_bets_pct_away', '?')
    total_open = game_data.get('total_open', '?')
    total_current = game_data.get('total_current', '?')
    vegas_line = game_data.get('vegas_line', '?')

    # Determine emoji for line movement
    if total_open != "?" and total_current != "?":
        try:
            open_val = float(total_open)
            current_val = float(total_current)
            if current_val > open_val:
                line_move_emoji = "📈"
            elif current_val < open_val:
                line_move_emoji = "📉"
            else:
                line_move_emoji = "⏸️"
            line_movement = f"{total_open} → {total_current} {line_move_emoji}"
        except:
            line_movement = f"{total_open} → {total_current}"
    else:
        line_movement = f"{total_open} → {total_current}"

    try:
        raw_conf = game_data.get('Confidence_Value')
        if raw_conf is None:
            raw_conf = float(str(game_data.get('Confidence', '0')).replace('%', ''))
        confidence_clean = f"{raw_conf:.0f}%" if raw_conf is not None else str(game_data.get('Confidence', '?'))

        if raw_conf >= 95:
            confidence_emoji = "🔥"
        elif raw_conf >= 90:
            confidence_emoji = "💪"
        elif raw_conf >= 80:
            confidence_emoji = "👍"
        elif raw_conf >= 70:
            confidence_emoji = "🤞"
        else:
            confidence_emoji = "😬"

    except:
        confidence_clean = game_data.get('Confidence', '?')
        confidence_emoji = ""

    proj_total = game_data.get('Projected_Total', '?')
    edge_val = game_data.get('Edge', '?')
    edge_str = f"{edge_val:+.2f}" if isinstance(edge_val, (int, float)) else edge_val
    away_r = game_data.get('Away_Runs', '?')
    home_r = game_data.get('Home_Runs', '?')
    return (
        f"🔥 *OVER GANG ALERT* 🔥\n\n"
        f"🏟️ *{game_data['Game']}*\n"
        f"📍 {game_data.get('Venue', 'Unknown')} | 🕒 {formatted_time}\n\n"
        f"🎯 *Pitchers*: {game_data['Pitchers']}\n"
        f"📊 xERA: {game_data['xERA']} | WHIP: {game_data['WHIP']}\n"
        f"📐 *Projection*: {away_r} + {home_r} = *{proj_total}* runs\n"
        f"📏 *Edge*: {edge_str} vs Vegas {vegas_line}\n"
        f"🧠 *Pick*: {game_data['Prediction']}\n"
        f"💪 *Confidence*: {confidence_clean}{f' {confidence_emoji}' if confidence_emoji else ''} | *Units*: {game_data.get('Units', '-')}\n"
        f"🏆 *ML Pick*: {game_data.get('ML_Pick', '-')} | *Kelly*: {game_data.get('ML_Kelly_Units', '-')}\n"
        f"📉 *Public*: {ou_over if ou_over != '?' else '-'}% Over / {ou_under if ou_under != '?' else '-'}% Under | Line: {line_movement}\n"
        f"🧾 *ML Bets*: {ml_home if ml_home != '?' else '-'}% Home / {ml_away if ml_away != '?' else '-'}% Away"
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

        odds_n = len(odds_map) if (odds_map is not None and isinstance(odds_map, dict)) else 0
        odds_ok = odds_n > 0
        odds_coverage_ok = games_n > 0 and odds_n >= max(1, games_n - 1)
        odds_status = f"{odds_n} games" if odds_ok else "empty or missing"

        manual_totals = _load_manual_totals()
        manual_loaded = isinstance(manual_totals, dict) and len(manual_totals) >= 1

        if not pitcher_ok:
            issues.append("pitcher data missing or empty")
        if not batter_ok:
            warnings.append("batter data missing or empty")
        if not bullpen_ok:
            warnings.append("bullpen data missing or empty")
        if games_n == 0:
            issues.append("no games for slate")
        if not public_ok:
            warnings.append("public betting empty or missing")

        if not pitcher_ok or games_n == 0:
            mode = "stop"
            if not pitcher_ok:
                issues.append("run mode stop: missing pitcher data")
            if games_n == 0:
                issues.append("run mode stop: no games for slate")
        elif (
            pitcher_ok
            and bullpen_ok
            and public_ok
            and odds_ok
            and odds_coverage_ok
        ):
            mode = "full_auto"
            warnings.append("all systems go; full auto mode")
        elif pitcher_ok and bullpen_ok and manual_loaded and (not public_ok or not odds_coverage_ok):
            mode = "manual_test"
            warnings.append("manual_totals loaded; public or odds partial — manual_test mode")
        else:
            mode = "projection_only"
            warnings.append("odds weak/fallback and no trusted manual totals — projection_only mode")

        ok = mode != "stop"

        print("\n--- PREFLIGHT ---")
        print(f"  Pitcher data:   {pitcher_status} (n={pitcher_n})")
        print(f"  Batter data:   {batter_status} (n={batter_n})")
        print(f"  Bullpen data:  {bullpen_status} (n={bullpen_n})")
        print(f"  Games found:   {games_status}")
        print(f"  Public betting: {public_status} (n={public_n})")
        print(f"  Odds status:   {odds_status} (n={odds_n})")
        print(f"  Manual totals: {'loaded' if manual_loaded else 'none'} ({len(manual_totals) if isinstance(manual_totals, dict) else 0} rows)")
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
        # --- Mountain Time "today"
        today_mt = datetime.now(ZoneInfo("America/Denver")).date()

        # Pull MLB schedule for that calendar day (UTC-based API)
        games = schedule(
            start_date=today_mt.strftime('%Y-%m-%d'),
            end_date=today_mt.strftime('%Y-%m-%d')
        )

        public_betting_data = load_public_betting_data()
        target_date_str = today_mt.strftime("%Y-%m-%d")
        print("[ODDS] Trying SportsDataIO first...")
        odds_map = fetch_mlb_odds_by_date(target_date_str)
        print(f"[ODDS] SportsDataIO odds_map size: {len(odds_map)}")
        odds_source = "none"
        if odds_map:
            odds_source = "SportsDataIO"
            print("[ODDS] Backfilling missing games from Odds API...")
            odds_api_map = fetch_mlb_odds(target_date=target_date_str)
            backfill_count = 0
            for k, v in (odds_api_map or {}).items():
                if k not in odds_map:
                    odds_map[k] = v
                    backfill_count += 1
            print(f"[ODDS] Odds API backfill size: {backfill_count}")
            print(f"[ODDS] Combined odds_map size after backfill: {len(odds_map)}")
        else:
            print("[ODDS] Falling back to Odds API...")
            odds_map = fetch_mlb_odds(target_date=target_date_str)
            if odds_map:
                odds_source = "Odds API"
        print(f"[ODDS] Final odds source: {odds_source}")

        # Keep only games that are actually TODAY in MT
        def game_mt_date(g):
            dt_utc = datetime.strptime(
                g["game_datetime"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=ZoneInfo("UTC"))
            return dt_utc.astimezone(ZoneInfo("America/Denver")).date()

        games = [g for g in games if game_mt_date(g) == today_mt]

        print(f"✅ Found {len(games)} games for {today_mt} MT")
        for g in games:
            print("•", f"{g['away_name']} @ {g['home_name']}")

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
        if mode == "stop":
            print("[PREFLIGHT] STOP engaged | exiting before game processing")
            return

    except Exception as e:
        print(f"❌ Schedule API error: {e}")
        return

    results = []
    alerts = []
    unmatched_pitchers = set()
    alias_log = []

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

            away_stats = DataManager.match_pitcher_row(stats_df, away_pitcher, alias_log=alias_log)
            home_stats = DataManager.match_pitcher_row(stats_df, home_pitcher, alias_log=alias_log)

            # 🧠 Log fallback usage if LowIP
            for name, stats in [(away_pitcher, away_stats), (home_pitcher, home_stats)]:
                if isinstance(stats, dict) and stats.get("LowIP", False):
                    print(f"⚠️ Using fallback stats for: {name}")

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
                'Odds_ML_Home': odds_info.get('ml_home'),
                'Odds_ML_Away': odds_info.get('ml_away'),
                'Market_Source': odds_source if 'odds_source' in locals() else '',
                'Captured_Book': odds_info.get('book', ''),
                'Captured_Total': odds_info.get('total_line', 8.5) if odds_info.get('_has_real_total', False) else '',
                'Captured_ML_Home': odds_info.get('ml_home'),
                'Captured_ML_Away': odds_info.get('ml_away'),
                'Fired_Play': False,
                'Trigger_Tags': '',
                'No_Fire_Reason': '',
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
            }

            # 🔮 Run prediction (compare projection to actual Vegas line; do not pass lineup-adjusted line)
            data_quality_degraded = (
                public is None or public == {} or
                "League Avg" in (away_pitcher or "") or
                "League Avg" in (home_pitcher or "") or
                bool(safe_get(away_stats, "LowIP", False)) or
                bool(safe_get(home_stats, "LowIP", False))
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
            is_manual_trusted = (odds_info.get("_source") == "manual_totals_csv") and has_real_total
            fire_threshold = 0.79 if is_manual_trusted else MIN_CONFIDENCE_ALERT
            fired = (confidence >= fire_threshold) and has_real_total
            trigger_tags = "|".join(filter(None, [
                "high_confidence" if fired else None,
                "sportsdataio" if (odds_source == "SportsDataIO") else None,
                "odds_api" if (odds_source == "Odds API") else None,
                "fallback_line" if (not has_real_total) else None,
            ]))
            game_data["Fired_Play"] = fired
            game_data["Trigger_Tags"] = trigger_tags
            if fired:
                game_data["No_Fire_Reason"] = ""
            else:
                if not has_real_total:
                    game_data["No_Fire_Reason"] = "fallback_line_used"
                elif abs(edge) < 1.0:
                    game_data["No_Fire_Reason"] = "edge_too_small"
                elif public is None or public == {} or ("League Avg" in (away_pitcher or "") or "League Avg" in (home_pitcher or "")):
                    game_data["No_Fire_Reason"] = "data_quality_degraded"
                elif confidence < fire_threshold:
                    game_data["No_Fire_Reason"] = "confidence_below_alert_threshold"
                else:
                    game_data["No_Fire_Reason"] = "manual_review"
            game_data["Play_Status"] = "BETTABLE" if has_real_total else "PROJECTION_ONLY"
            game_data["Bettable"] = bool(has_real_total)
            game_data["Model_Notes"] = f"edge={edge:.2f}|conf={confidence:.2f}|book={odds_info.get('book', '')}"
            game_data["Confidence_Tier"] = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.60 else "low")
            game_data["Edge_Tier"] = "strong" if abs(edge) >= 2.0 else ("medium" if abs(edge) >= 1.0 else "thin")
            game_data["Bet_Type"] = "total"
            game_data["Side"] = "over" if "OVER" in (prediction or "").upper() else ("under" if "UNDER" in (prediction or "").upper() else "")
            line_status = "market" if bool(odds_info.get('_has_real_total', False)) else "fallback"
            game_data["Line_Status"] = line_status
            game_data["Fallback_Used"] = not bool(odds_info.get('_has_real_total', False))
            dq_parts = []
            if line_status == "fallback":
                dq_parts.append("fallback_line")
            # Fallback pitcher stats: synthetic league-avg starters or any starter on LowIP / league-avg row from matcher
            _away_low = bool(safe_get(away_stats, "LowIP", False))
            _home_low = bool(safe_get(home_stats, "LowIP", False))
            if (
                "League Avg" in (away_pitcher or "")
                or "League Avg" in (home_pitcher or "")
                or _away_low
                or _home_low
            ):
                dq_parts.append("fallback_pitcher")
            if public is None or public == {}:
                dq_parts.append("missing_public_data")
            if _away_low or _home_low:
                dq_parts.append("low_ip")
            game_data["Data_Quality_Flag"] = "|".join(dq_parts)

            # 💵 MONEYLINE PREDICTION
            home_ml_data = get_team_ml_data(home_team, home_pitcher)
            away_ml_data = get_team_ml_data(away_team, away_pitcher)

            home_win_prob, away_win_prob = calculate_team_win_probability(home_ml_data, away_ml_data)

            # Get implied odds for home ML (from public/Vegas data)
            try:
                odds_str = public.get("ML_Home", "-130") if isinstance(public, dict) else "-130"
                odds_value = float(odds_str) if isinstance(odds_str, (int, float)) else float(str(odds_str).strip())
                if odds_value < 0:
                    implied_home = abs(odds_value) / (100 + abs(odds_value))  # favorite
                else:
                    implied_home = 100 / (100 + odds_value)  # underdog
            except Exception:
                implied_home = 0.53

            implied_away = 1 - implied_home

            home_kelly = calculate_kelly_units(home_win_prob, implied_home)
            away_kelly = calculate_kelly_units(away_win_prob, implied_away)

            # ✅ Always give a pick (even if edge is small)
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

            # Add to game data
            game_data.update({
                "ML_Pick": ml_pick,
                "ML_Confidence": ml_conf,
                "ML_Value": ml_value,
                "ML_Kelly_Units": ml_kelly,
            })

            if isinstance(public, dict):
                print(f"Public keys found: {list(public.keys())}")
            else:
                print("Public betting data is missing (NoneType)")

            if public:
                game_data['ou_bets_pct_over'] = public.get('ou_bets_pct_over', '?')
                game_data['ou_bets_pct_under'] = public.get('ou_bets_pct_under', '?')
                game_data['ml_bets_pct_home'] = public.get('ml_bets_pct_home', '?')
                game_data['ml_bets_pct_away'] = public.get('ml_bets_pct_away', '?')
                game_data['total_open'] = public.get('total_open', '?')
                game_data['total_current'] = public.get('total_current', '?')

            results.append(game_data)
            print(f"✅ Prediction: {prediction} | Confidence: {confidence:.0%}")
            if fired:
                alerts.append(game_data)

        except Exception as e:
            print(f"❌ Game processing error: {e}")
            traceback.print_exc()
            continue

    # Save results (only real-market totals)
    eligible_results = [r for r in results if r.get("Total_Is_Real", False)]
    if eligible_results:
        archive_date = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs(ARCHIVE_DIR, exist_ok=True)

        results_df = pd.DataFrame(eligible_results, columns=[
            "Game", "Projected_Total", "Away_Runs", "Home_Runs", "Vegas_Line", "Edge",
            "Prediction", "Confidence", "Units", "Line_Open", "Line_Current",
            "Total_Is_Real", "Odds_Line", "Over_Juice", "Under_Juice", "Odds_Book",
            "Market_Source", "Captured_Book", "Captured_Total", "Captured_ML_Home", "Captured_ML_Away",
            "Fired_Play", "Trigger_Tags", "No_Fire_Reason", "Model_Notes",
            "Confidence_Tier", "Edge_Tier", "Bet_Type", "Side", "Play_Status", "Bettable",
            "Line_Status", "Fallback_Used", "Data_Quality_Flag",
            "Bet_Line", "Closing_Line", "CLV", "CLV_Result",
            "ML_Pick", "ML_Confidence", "ML_Value", "ML_Kelly_Units"
        ])

        csv_path = f"{ARCHIVE_DIR}/predictions_{archive_date}.csv"
        results_df.to_csv(csv_path, index=False)

        print(f"\n💾 Saved {len(eligible_results)} predictions")

        # 📤 Auto-upload to Telegram
        send_telegram_file(csv_path, caption=f"📊 Over Gang Predictions for {datetime.now().strftime('%b %d')}")
    elif results:
        print("\nℹ️ No eligible real-market plays to export; all games were fallback/no-bet.")

    # Send alerts
    if alerts:
        print(f"\n🚨 Sending {len(alerts)} alerts...")
        for alert in alerts:
            message = format_alert(alert)
            if send_telegram_alert(message):
                print(f"📤 Alert sent for {alert['Game']}")
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

    # Final run summary
    _manual = _load_manual_totals()
    print("\n--- RUN SUMMARY ---")
    print(f"  Mode: {preflight.get('mode', 'projection_only')}")
    print(f"  Games processed: {len(results)}")
    print(f"  Bettable plays saved: {len(eligible_results)}")
    print(f"  Alerts sent: {len(alerts)}")
    print(f"  Manual totals loaded: {len(_manual)}")
    print("-------------------")

# ================================
# 🚀 ENTRY POINT
# ================================
if __name__ == "__main__":
    run_predictions()
