#!/usr/bin/env python3
"""Standalone Polymarket vs Pinnacle sentiment board.

Reads an Over Gang predictions CSV, fetches live Polymarket O/U markets via the
Parlay event-markets search, pairs with Pinnacle sharp totals from the Parlay
odds endpoint, and writes a separate archive board.

Does not modify predictions, public_betting.csv, or any model logic.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - missing dotenv handled by env-only fallback
    load_dotenv = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "archive"
PARLAY_BASE = "https://api.parlay-api.com/v1"
PARLAY_ODDS_URL = f"{PARLAY_BASE}/sports/baseball_mlb/odds"
PARLAY_EVENT_MARKETS_URL = f"{PARLAY_BASE}/event-markets/search"
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
THE_ODDS_API_MLB_ODDS_URL = f"{THE_ODDS_API_BASE}/sports/baseball_mlb/odds"
REQUEST_TIMEOUT_S = 20
SEARCH_SLEEP_S = 0.5
RATE_LIMIT_SLEEP_S = 3.0
MIN_PM_CONFIDENCE = 0.75
MAX_PM_SPREAD = 0.10
MIN_PM_LIQUIDITY = 1000.0
EXACT_LINE_TOLERANCE = 0.0
NEAREST_LINE_TOLERANCE = 0.5

OUTPUT_COLUMNS = [
    "Game_Date", "Datetime", "Game", "Prediction", "OU_Side", "Bet_Line",
    "Projected_Total", "OU_Edge", "OU_Confidence", "OU_Fired", "Fired_Play",
    "Total_Is_Real", "Data_Quality_Flag", "Market_Source", "Odds_Book",
    "Over_Juice", "Under_Juice",
    "PM_Source", "PM_Market_ID", "PM_Title", "PM_Event_Title", "PM_Status",
    "PM_Close_Time", "PM_URL", "PM_Total_Line", "PM_Line_Match_Type",
    "PM_Match_Status", "PM_Match_Confidence",
    "PM_Over_Price", "PM_Under_Price", "PM_Best_Bid", "PM_Best_Ask", "PM_Last",
    "PM_Spread", "PM_Volume", "PM_Liquidity",
    "PM_Crowd_Side", "PM_Crowd_Lean_Strength",
    "Sharp_Source", "Pinnacle_Total", "Pinnacle_Over_Juice", "Pinnacle_Under_Juice",
    "Sharp_Over_Implied", "Sharp_Under_Implied",
    "PM_vs_Sharp_Over_Gap", "PM_vs_Sharp_Under_Gap",
    "PM_Signal_Direction", "PM_Signal_Strength", "PM_Signal_Reason",
    "PM_Query", "PM_Candidate_Count", "PM_Eligible_Candidate_Count",
    "PM_All_Total_Lines_Found", "PM_Error",
]

CARRY_FORWARD_COLUMNS = [
    "Game", "Game_Date", "Datetime", "Prediction", "Bet_Line", "Projected_Total",
    "OU_Side", "OU_Edge", "OU_Confidence", "OU_Fired", "Fired_Play",
    "Total_Is_Real", "Data_Quality_Flag", "Market_Source", "Odds_Book",
    "Over_Juice", "Under_Juice",
]

_LIGHT_ALIAS_MAP = {
    "athletics": "oakland athletics",
    "a's": "oakland athletics",
    "as": "oakland athletics",
    "oakland athletics": "oakland athletics",
    "la angels": "los angeles angels",
    "la dodgers": "los angeles dodgers",
    "ny mets": "new york mets",
    "ny yankees": "new york yankees",
    "chi cubs": "chicago cubs",
    "chi white sox": "chicago white sox",
    "st louis cardinals": "st. louis cardinals",
    "st. louis cardinals": "st. louis cardinals",
}


def _ensure_root_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    if load_dotenv is None:
        return
    env_path = ROOT / ".env"
    try:
        load_dotenv(env_path) if env_path.exists() else load_dotenv()
    except Exception:
        pass


def _get_parlay_api_key() -> Optional[str]:
    raw = os.getenv("ODDS_API_KEY")
    if raw is None:
        return None
    key = str(raw).strip()
    return key or None


def _get_the_odds_api_key() -> Optional[str]:
    raw = os.getenv("THE_ODDS_API_KEY")
    if raw is None:
        return None
    key = str(raw).strip()
    return key or None


def _normalize_team_name_fallback(name: Any) -> str:
    return str(name or "").lower().strip()


def _resolve_normalizer():
    _ensure_root_path()
    try:
        from core.public_betting_loader import normalize_team_name  # type: ignore

        return normalize_team_name
    except Exception:
        return _normalize_team_name_fallback


_NORMALIZE_TEAM = _resolve_normalizer()


def _normalize_team(name: Any) -> str:
    base = _NORMALIZE_TEAM(name or "")
    s = str(base or "").strip().lower()
    return _LIGHT_ALIAS_MAP.get(s, s)


def _split_game_label(game: Any) -> Tuple[str, str]:
    s = str(game or "").strip()
    if " @ " not in s:
        return "", ""
    away, home = s.split(" @ ", 1)
    return away.strip(), home.strip()


def _team_match_in_text(text: str, team_norm: str) -> bool:
    if not team_norm:
        return False
    text_norm = re.sub(r"\s+", " ", text.lower())
    if team_norm in text_norm:
        return True
    # Try alias variants of the same logical team
    for alias, canonical in _LIGHT_ALIAS_MAP.items():
        if canonical == team_norm and alias in text_norm:
            return True
    # Try short city/team words derived from canonical
    last_word = team_norm.split()[-1]
    if last_word and last_word in text_norm:
        return True
    return False


def _event_title_matches(event_title: str, away_team: str, home_team: str) -> bool:
    if not event_title:
        return False
    away_norm = _normalize_team(away_team)
    home_norm = _normalize_team(home_team)
    text = str(event_title).strip().lower()
    return _team_match_in_text(text, away_norm) and _team_match_in_text(text, home_norm)


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        f = float(value)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def american_to_implied(odds: Any) -> Optional[float]:
    f = _parse_float(odds)
    if f is None:
        return None
    if f < 0:
        return abs(f) / (abs(f) + 100.0)
    return 100.0 / (f + 100.0)


def devig_pair(over_juice: Any, under_juice: Any) -> Tuple[Optional[float], Optional[float]]:
    raw_over = american_to_implied(over_juice)
    raw_under = american_to_implied(under_juice)
    if raw_over is None or raw_under is None:
        return None, None
    total = raw_over + raw_under
    if total <= 0:
        return None, None
    return raw_over / total, raw_under / total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prediction market sentiment board.")
    p.add_argument("--predictions-file", help="Path to a predictions_*.csv")
    p.add_argument("--date", help="Slate date as YYYYMMDD")
    return p.parse_args()


def _resolve_predictions_file(supplied: Optional[str], date: Optional[str]) -> Optional[Path]:
    if supplied:
        p = Path(supplied)
        if not p.is_absolute():
            p = ROOT / p
        return p
    if date:
        matches = sorted(glob.glob(str(ARCHIVE_DIR / f"predictions_{date}_*.csv")))
        if matches:
            return Path(max(matches, key=lambda x: os.path.getmtime(x)))
        return None
    matches = sorted(glob.glob(str(ARCHIVE_DIR / "predictions_*.csv")))
    if not matches:
        return None
    return Path(max(matches, key=lambda x: os.path.getmtime(x)))


def _date_from_predictions_path(path: Path) -> Optional[str]:
    m = re.match(r"predictions_(\d{8})_\d{4}\.csv$", path.name, re.IGNORECASE)
    return m.group(1) if m else None


def _output_path(predictions_path: Path, date_arg: Optional[str]) -> Path:
    date = _date_from_predictions_path(predictions_path) or (date_arg or datetime.now().strftime("%Y%m%d"))
    stamp = datetime.now().strftime("%H%M")
    return ARCHIVE_DIR / f"prediction_market_board_{date}_{stamp}.csv"


def fetch_pinnacle_odds_map(api_key: str) -> Dict[str, Dict[str, Any]]:
    """Return mapping of normalized 'away @ home' -> Pinnacle totals dict."""
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    params = {"regions": "us", "markets": "h2h,totals", "oddsFormat": "american"}
    try:
        r = requests.get(PARLAY_ODDS_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_S)
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        print(f"Parlay odds fetch failed: {e}", file=sys.stderr)
        return {}

    if isinstance(body, dict) and isinstance(body.get("data"), list):
        events = body["data"]
    elif isinstance(body, list):
        events = body
    else:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        away = (event.get("away_team") or "").strip()
        home = (event.get("home_team") or "").strip()
        if not away or not home:
            continue
        game_key = f"{_normalize_team(away)} @ {_normalize_team(home)}"
        if not game_key.strip(" @ "):
            continue

        pinnacle_book = None
        for bm in event.get("bookmakers") or []:
            if isinstance(bm, dict) and str(bm.get("key") or "").strip().lower() == "pinnacle":
                pinnacle_book = bm
                break
        if pinnacle_book is None:
            continue

        pinn_total = pinn_over = pinn_under = None
        for market in pinnacle_book.get("markets") or []:
            if not isinstance(market, dict) or market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes") or []:
                if not isinstance(outcome, dict):
                    continue
                name = (outcome.get("name") or "").strip().lower()
                if name not in ("over", "under"):
                    continue
                point = _parse_float(outcome.get("point"))
                price = _parse_int(outcome.get("price"))
                if point is None or price is None:
                    continue
                pinn_total = point
                if name == "over":
                    pinn_over = price
                else:
                    pinn_under = price
        if pinn_total is not None and pinn_over is not None and pinn_under is not None:
            out[game_key] = {
                "pinnacle_total": pinn_total,
                "pinnacle_over_juice": pinn_over,
                "pinnacle_under_juice": pinn_under,
            }
    return out


def fetch_the_odds_api_pinnacle_totals(api_key: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Return mapping of normalized 'away @ home' -> Pinnacle totals from The Odds API."""
    if not api_key:
        print(
            "THE_ODDS_API_KEY missing; skipping The Odds API Pinnacle fallback fetch.",
            file=sys.stderr,
        )
        return {}
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
        "bookmakers": "pinnacle",
    }
    try:
        r = requests.get(
            THE_ODDS_API_MLB_ODDS_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_S,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        print(f"The Odds API Pinnacle totals fetch failed: {e}", file=sys.stderr)
        return {}

    if not isinstance(body, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for event in body:
        if not isinstance(event, dict):
            continue
        away = (event.get("away_team") or "").strip()
        home = (event.get("home_team") or "").strip()
        if not away or not home:
            continue
        game_key = f"{_normalize_team(away)} @ {_normalize_team(home)}"
        if not game_key.strip(" @ "):
            continue

        pinnacle_book = None
        for bm in event.get("bookmakers") or []:
            if isinstance(bm, dict) and str(bm.get("key") or "").strip().lower() == "pinnacle":
                pinnacle_book = bm
                break
        if pinnacle_book is None:
            continue

        pinn_total = pinn_over = pinn_under = None
        for market in pinnacle_book.get("markets") or []:
            if not isinstance(market, dict) or market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes") or []:
                if not isinstance(outcome, dict):
                    continue
                name = (outcome.get("name") or "").strip().lower()
                if name not in ("over", "under"):
                    continue
                point = _parse_float(outcome.get("point"))
                price = _parse_int(outcome.get("price"))
                if point is None or price is None:
                    continue
                pinn_total = point
                if name == "over":
                    pinn_over = price
                else:
                    pinn_under = price
        if pinn_total is not None and pinn_over is not None and pinn_under is not None:
            out[game_key] = {
                "pinnacle_total": pinn_total,
                "pinnacle_over_juice": pinn_over,
                "pinnacle_under_juice": pinn_under,
                "source": "the_odds_api_pinnacle",
                "last_update": pinnacle_book.get("last_update"),
            }
    return out


def _build_pm_query(away_team: str, home_team: str) -> str:
    return f"{away_team} {home_team} O/U".strip()


def _request_pm_markets(api_key: str, query: str) -> List[Dict[str, Any]]:
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    params = {
        "q": query,
        "sources": "polymarket",
        "min_confidence": 0,
        "sort": "balanced",
    }
    for attempt in (1, 2):
        try:
            r = requests.get(
                PARLAY_EVENT_MARKETS_URL,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT_S,
            )
        except Exception:
            if attempt == 1:
                time.sleep(RATE_LIMIT_SLEEP_S)
                continue
            raise
        if r.status_code == 429:
            if attempt == 1:
                time.sleep(RATE_LIMIT_SLEEP_S)
                continue
            r.raise_for_status()
        r.raise_for_status()
        body = r.json()
        if isinstance(body, dict):
            markets = body.get("markets")
            if isinstance(markets, list):
                return markets
        return []
    return []


def _extract_pm_total_line(title: Any) -> Optional[float]:
    if not title:
        return None
    m = re.search(r"O/U\s+(\d+\.?\d*)", str(title), re.IGNORECASE)
    if not m:
        return None
    return _parse_float(m.group(1))


def _outcome_prices(market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    prices = market.get("prices") or {}
    outcomes = prices.get("outcomes") or []
    over = under = None
    if isinstance(outcomes, list):
        for o in outcomes:
            if not isinstance(o, dict):
                continue
            name = (o.get("outcome") or "").strip().lower()
            price = _parse_float(o.get("price"))
            if name == "over":
                over = price
            elif name == "under":
                under = price
    return over, under


def _filter_pm_candidates(
    markets: List[Dict[str, Any]],
    away_team: str,
    home_team: str,
) -> List[Dict[str, Any]]:
    eligible: List[Dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue
        if str(market.get("source") or "").strip().lower() != "polymarket":
            continue
        if str(market.get("status") or "").strip().lower() != "active":
            continue
        title = str(market.get("title") or "")
        question = str(market.get("question") or "")
        if "o/u" not in title.lower() and "o/u" not in question.lower():
            continue
        over_price, under_price = _outcome_prices(market)
        if over_price is None or under_price is None:
            continue
        confidence = _parse_float(market.get("match_confidence")) or 0.0
        if confidence < MIN_PM_CONFIDENCE:
            continue
        if not _event_title_matches(str(market.get("event_title") or ""), away_team, home_team):
            continue
        prices = market.get("prices") or {}
        spread = _parse_float(prices.get("spread"))
        if spread is not None and spread > MAX_PM_SPREAD:
            continue
        liquidity = _parse_float(market.get("liquidity"))
        if liquidity is not None and liquidity < MIN_PM_LIQUIDITY:
            continue
        if _extract_pm_total_line(title) is None and _extract_pm_total_line(question) is None:
            continue
        eligible.append(market)
    return eligible


def _select_best_pm_candidate(
    candidates: List[Dict[str, Any]],
    bet_line: Optional[float],
) -> Tuple[Optional[Dict[str, Any]], str, Optional[float]]:
    if not candidates or bet_line is None:
        return None, "no_match", None

    enriched = []
    for market in candidates:
        title = market.get("title") or market.get("question") or ""
        pm_line = _extract_pm_total_line(title)
        if pm_line is None:
            continue
        distance = abs(pm_line - bet_line)
        prices = market.get("prices") or {}
        spread = _parse_float(prices.get("spread")) or float("inf")
        liquidity = _parse_float(market.get("liquidity")) or 0.0
        volume = _parse_float(market.get("volume")) or 0.0
        enriched.append({
            "market": market,
            "pm_line": pm_line,
            "distance": distance,
            "spread": spread,
            "liquidity": liquidity,
            "volume": volume,
        })

    if not enriched:
        return None, "no_match", None

    enriched.sort(key=lambda r: (r["distance"], r["spread"], -r["liquidity"], -r["volume"]))
    best = enriched[0]
    if best["distance"] <= EXACT_LINE_TOLERANCE:
        return best["market"], "exact", best["pm_line"]
    if best["distance"] <= NEAREST_LINE_TOLERANCE:
        return best["market"], "nearest_within_0_5", best["pm_line"]
    return None, "no_match", None


def _crowd_summary(pm_over: Optional[float], pm_under: Optional[float]) -> Tuple[str, str]:
    if pm_over is None or pm_under is None:
        return "", ""
    if pm_over > 0.50:
        side = "OVER"
        winning_price = pm_over
    elif pm_under > 0.50:
        side = "UNDER"
        winning_price = pm_under
    else:
        return "", "flat"
    lean = winning_price - 0.50
    if lean >= 0.15:
        strength = "strong"
    elif lean >= 0.08:
        strength = "mild"
    else:
        strength = "flat"
    return side, strength


def _signal_strength(value: float) -> str:
    if value >= 0.12:
        return "extreme"
    if value >= 0.08:
        return "strong"
    if value >= 0.05:
        return "mild"
    return "none"


def _build_signal_reason(parts: List[str]) -> str:
    return "|".join([p for p in parts if p])


def _safe(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return value


def build_row(
    csv_row: Dict[str, Any],
    pinnacle_odds_by_game: Dict[str, Dict[str, Any]],
    api_key: str,
    toa_pinnacle_odds_by_game: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {col: "" for col in OUTPUT_COLUMNS}
    for col in CARRY_FORWARD_COLUMNS:
        if col in csv_row:
            out[col] = _safe(csv_row.get(col))

    away_team, home_team = _split_game_label(csv_row.get("Game"))
    game_key = f"{_normalize_team(away_team)} @ {_normalize_team(home_team)}"

    pm_query = _build_pm_query(away_team, home_team) if away_team and home_team else ""
    out["PM_Query"] = pm_query
    out["PM_Match_Status"] = "no_pm_match"
    out["PM_Line_Match_Type"] = "no_match"

    bet_line = _parse_float(csv_row.get("Bet_Line"))

    markets: List[Dict[str, Any]] = []
    pm_error = ""
    if pm_query and api_key:
        try:
            markets = _request_pm_markets(api_key, pm_query)
        except Exception as exc:
            pm_error = str(exc)
            markets = []
        time.sleep(SEARCH_SLEEP_S)

    out["PM_Error"] = pm_error
    out["PM_Candidate_Count"] = len(markets)
    all_lines = []
    for m in markets:
        if not isinstance(m, dict):
            continue
        line = _extract_pm_total_line(m.get("title") or m.get("question") or "")
        if line is not None:
            all_lines.append(f"{line:g}")
    out["PM_All_Total_Lines_Found"] = ",".join(all_lines)

    eligible_candidates = _filter_pm_candidates(markets, away_team, home_team)
    out["PM_Eligible_Candidate_Count"] = len(eligible_candidates)

    selected_market = None
    pm_line_match_type = "no_match"
    pm_total_line: Optional[float] = None

    if eligible_candidates:
        selected_market, pm_line_match_type, pm_total_line = _select_best_pm_candidate(
            eligible_candidates, bet_line
        )
        out["PM_Line_Match_Type"] = pm_line_match_type
        if selected_market is not None:
            out["PM_Match_Status"] = "matched"
        else:
            out["PM_Match_Status"] = "no_matching_total"
    else:
        out["PM_Match_Status"] = "no_pm_match"

    pm_over_price = pm_under_price = None
    pm_spread = pm_volume = pm_liquidity = None

    if selected_market is not None:
        out["PM_Source"] = selected_market.get("source") or ""
        out["PM_Market_ID"] = selected_market.get("market_id") or ""
        out["PM_Title"] = selected_market.get("title") or ""
        out["PM_Event_Title"] = selected_market.get("event_title") or ""
        out["PM_Status"] = selected_market.get("status") or ""
        out["PM_Close_Time"] = selected_market.get("close_time") or ""
        out["PM_URL"] = selected_market.get("url") or ""
        out["PM_Match_Confidence"] = _parse_float(selected_market.get("match_confidence")) or ""
        out["PM_Total_Line"] = pm_total_line if pm_total_line is not None else ""

        prices = selected_market.get("prices") or {}
        pm_over_price, pm_under_price = _outcome_prices(selected_market)
        out["PM_Over_Price"] = pm_over_price if pm_over_price is not None else ""
        out["PM_Under_Price"] = pm_under_price if pm_under_price is not None else ""
        out["PM_Best_Bid"] = _safe(prices.get("best_bid"))
        out["PM_Best_Ask"] = _safe(prices.get("best_ask"))
        out["PM_Last"] = _safe(prices.get("last"))
        pm_spread = _parse_float(prices.get("spread"))
        out["PM_Spread"] = pm_spread if pm_spread is not None else ""
        pm_volume = _parse_float(selected_market.get("volume"))
        out["PM_Volume"] = pm_volume if pm_volume is not None else ""
        pm_liquidity = _parse_float(selected_market.get("liquidity"))
        out["PM_Liquidity"] = pm_liquidity if pm_liquidity is not None else ""

        crowd_side, crowd_strength = _crowd_summary(pm_over_price, pm_under_price)
        out["PM_Crowd_Side"] = crowd_side
        out["PM_Crowd_Lean_Strength"] = crowd_strength

    # Sharp source resolution
    sharp_source = "missing"
    pinnacle_total = pinnacle_over = pinnacle_under = None
    pinn_entry = pinnacle_odds_by_game.get(game_key)
    if pinn_entry is not None:
        pinnacle_total = pinn_entry.get("pinnacle_total")
        pinnacle_over = pinn_entry.get("pinnacle_over_juice")
        pinnacle_under = pinn_entry.get("pinnacle_under_juice")
        if pinnacle_over is not None and pinnacle_under is not None:
            sharp_source = "pinnacle"
    if sharp_source == "missing" and toa_pinnacle_odds_by_game:
        toa_entry = toa_pinnacle_odds_by_game.get(game_key)
        if toa_entry is not None:
            pinnacle_total = toa_entry.get("pinnacle_total")
            pinnacle_over = toa_entry.get("pinnacle_over_juice")
            pinnacle_under = toa_entry.get("pinnacle_under_juice")
            if pinnacle_over is not None and pinnacle_under is not None:
                sharp_source = "the_odds_api_pinnacle"
    if sharp_source == "missing":
        csv_over = _parse_int(csv_row.get("Over_Juice"))
        csv_under = _parse_int(csv_row.get("Under_Juice"))
        if csv_over is not None and csv_under is not None:
            sharp_source = "captured_book_fallback"
            pinnacle_total = bet_line
            pinnacle_over = csv_over
            pinnacle_under = csv_under

    out["Sharp_Source"] = sharp_source
    out["Pinnacle_Total"] = pinnacle_total if pinnacle_total is not None else ""
    out["Pinnacle_Over_Juice"] = pinnacle_over if pinnacle_over is not None else ""
    out["Pinnacle_Under_Juice"] = pinnacle_under if pinnacle_under is not None else ""

    sharp_over_implied = sharp_under_implied = None
    if sharp_source != "missing" and pinnacle_over is not None and pinnacle_under is not None:
        sharp_over_implied, sharp_under_implied = devig_pair(pinnacle_over, pinnacle_under)
    out["Sharp_Over_Implied"] = sharp_over_implied if sharp_over_implied is not None else ""
    out["Sharp_Under_Implied"] = sharp_under_implied if sharp_under_implied is not None else ""

    pm_signal_direction = ""
    pm_signal_strength = ""
    over_gap = under_gap = None
    if (
        out["PM_Match_Status"] == "matched"
        and sharp_source != "missing"
        and pm_over_price is not None
        and pm_under_price is not None
        and sharp_over_implied is not None
        and sharp_under_implied is not None
    ):
        over_gap = sharp_over_implied - pm_over_price
        under_gap = sharp_under_implied - pm_under_price
        out["PM_vs_Sharp_Over_Gap"] = round(over_gap, 4)
        out["PM_vs_Sharp_Under_Gap"] = round(under_gap, 4)
        if over_gap >= 0.05 and over_gap > under_gap:
            pm_signal_direction = "OVER"
        elif under_gap >= 0.05 and under_gap > over_gap:
            pm_signal_direction = "UNDER"
        winning_gap = max(over_gap, under_gap)
        pm_signal_strength = _signal_strength(winning_gap)

    out["PM_Signal_Direction"] = pm_signal_direction
    out["PM_Signal_Strength"] = pm_signal_strength

    signal_parts: List[str] = []
    if out["PM_Match_Status"] == "matched":
        signal_parts.append("pm_market_matched")
    if pm_line_match_type == "exact":
        signal_parts.append("exact_total_match")
    elif pm_line_match_type == "nearest_within_0_5":
        signal_parts.append("nearest_total_match")
    if sharp_source == "pinnacle":
        signal_parts.append("sharp_source_pinnacle")
    elif sharp_source == "the_odds_api_pinnacle":
        signal_parts.append("sharp_source_the_odds_api_pinnacle")
    elif sharp_source == "captured_book_fallback":
        signal_parts.append("sharp_source_fallback")
    if over_gap is not None:
        signal_parts.append(f"over_gap_{over_gap:.3f}")
    if under_gap is not None:
        signal_parts.append(f"under_gap_{under_gap:.3f}")
    if pm_signal_direction == "OVER":
        signal_parts.append("signal_over")
    elif pm_signal_direction == "UNDER":
        signal_parts.append("signal_under")
    elif over_gap is not None and under_gap is not None:
        signal_parts.append("no_signal")
    if pm_liquidity is not None and pm_liquidity < MIN_PM_LIQUIDITY:
        signal_parts.append("low_liquidity")
    if pm_spread is not None and pm_spread > MAX_PM_SPREAD:
        signal_parts.append("wide_pm_spread")
    if out["PM_Match_Status"] in {"no_pm_match", "no_matching_total"}:
        signal_parts.append("no_pm_match")
    if sharp_source == "missing":
        signal_parts.append("no_sharp_price")

    out["PM_Signal_Reason"] = _build_signal_reason(signal_parts)
    return out


def _print_summary(
    predictions_path: Path,
    output_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    matched_exact = sum(1 for r in rows if r.get("PM_Line_Match_Type") == "exact" and r.get("PM_Match_Status") == "matched")
    matched_nearest = sum(1 for r in rows if r.get("PM_Line_Match_Type") == "nearest_within_0_5" and r.get("PM_Match_Status") == "matched")
    no_pm_match = sum(1 for r in rows if r.get("PM_Match_Status") in {"no_pm_match", "no_matching_total"})
    sharp_pinn = sum(1 for r in rows if r.get("Sharp_Source") == "pinnacle")
    sharp_toa_pinn = sum(1 for r in rows if r.get("Sharp_Source") == "the_odds_api_pinnacle")
    sharp_fb = sum(1 for r in rows if r.get("Sharp_Source") == "captured_book_fallback")
    sharp_missing = sum(1 for r in rows if r.get("Sharp_Source") == "missing")
    sig_over = sum(1 for r in rows if r.get("PM_Signal_Direction") == "OVER")
    sig_under = sum(1 for r in rows if r.get("PM_Signal_Direction") == "UNDER")

    print(f"Input predictions file: {predictions_path}")
    print(f"Output board: {output_path}")
    print(f"Rows written: {len(rows)}")
    print(f"PM matched exact: {matched_exact}")
    print(f"PM matched nearest: {matched_nearest}")
    print(f"No PM match: {no_pm_match}")
    print(f"Sharp source pinnacle: {sharp_pinn}")
    print(f"Sharp source the_odds_api_pinnacle: {sharp_toa_pinn}")
    print(f"Sharp source fallback: {sharp_fb}")
    print(f"Sharp missing: {sharp_missing}")
    print(f"Signals OVER: {sig_over}")
    print(f"Signals UNDER: {sig_under}")


def main() -> int:
    args = parse_args()
    _load_env()
    api_key = _get_parlay_api_key()
    if not api_key:
        print("ODDS_API_KEY missing; Pinnacle and Polymarket fetches will be skipped.", file=sys.stderr)
    the_odds_api_key = _get_the_odds_api_key()

    predictions_path = _resolve_predictions_file(args.predictions_file, args.date)
    if predictions_path is None or not predictions_path.exists():
        print("No predictions CSV found.", file=sys.stderr)
        return 2

    predictions_df = pd.read_csv(predictions_path)
    pinnacle_odds_by_game = fetch_pinnacle_odds_map(api_key) if api_key else {}
    toa_pinnacle_odds_by_game = fetch_the_odds_api_pinnacle_totals(the_odds_api_key)

    rows: List[Dict[str, Any]] = []
    for _, csv_row in predictions_df.iterrows():
        record = csv_row.to_dict()
        try:
            board_row = build_row(
                record,
                pinnacle_odds_by_game,
                api_key or "",
                toa_pinnacle_odds_by_game=toa_pinnacle_odds_by_game,
            )
        except Exception as exc:
            traceback.print_exc()
            board_row = {col: "" for col in OUTPUT_COLUMNS}
            for col in CARRY_FORWARD_COLUMNS:
                if col in record:
                    board_row[col] = _safe(record.get(col))
            away_team, home_team = _split_game_label(record.get("Game"))
            board_row["PM_Query"] = _build_pm_query(away_team, home_team) if away_team and home_team else ""
            board_row["PM_Match_Status"] = "no_pm_match"
            board_row["PM_Line_Match_Type"] = "no_match"
            board_row["Sharp_Source"] = "missing"
            board_row["PM_Error"] = str(exc)
        rows.append(board_row)

    output_path = _output_path(predictions_path, args.date)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(output_path, index=False)
    _print_summary(predictions_path, output_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
