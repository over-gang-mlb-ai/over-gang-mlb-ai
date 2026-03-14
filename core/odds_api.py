"""
The Odds API integration for MLB.
Loads ODDS_API_KEY from .env; fetches totals + moneylines (American odds).
Prefer sharp book first; return safe fallbacks when unavailable.
"""
import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
MLB_SPORT_KEY = "baseball_mlb"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
# Prefer sharp / liquid books first; fall back to first available
BOOK_PRIORITY = ("pinnacle", "bovada", "betonlineag", "draftkings", "fanduel", "betmgm", "pointsbetus", "williamhill_us")

DEFAULT_TOTAL = 8.5
DEFAULT_JUICE = -110


def _game_key(away_team, home_team):
    """Build key for lookup: normalized 'away @ home' (match public_betting_loader style)."""
    try:
        from core.public_betting_loader import normalize_team_name
        a = normalize_team_name(away_team or "")
        h = normalize_team_name(home_team or "")
        return f"{a} @ {h}"
    except Exception:
        return f"{(away_team or '').lower().strip()} @ {(home_team or '').lower().strip()}"


def _parse_totals_and_h2h(book):
    """From one bookmaker, extract totals (line, over_juice, under_juice) and h2h (ml_home, ml_away)."""
    total_line = DEFAULT_TOTAL
    over_juice = under_juice = DEFAULT_JUICE
    ml_home = ml_away = None
    for market in book.get("markets") or []:
        if market.get("key") == "totals":
            outcomes = market.get("outcomes") or []
            for o in outcomes:
                name = (o.get("name") or "").lower()
                if name == "over":
                    total_line = float(o.get("point", DEFAULT_TOTAL))
                    over_juice = int(o.get("price", DEFAULT_JUICE))
                elif name == "under":
                    total_line = float(o.get("point", DEFAULT_TOTAL))
                    under_juice = int(o.get("price", DEFAULT_JUICE))
        elif market.get("key") == "h2h":
            home = book.get("_home")  # we'll set this from event
            away = book.get("_away")
            for o in market.get("outcomes") or []:
                n = (o.get("name") or "").strip()
                p = int(o.get("price", 0))
                if n == home:
                    ml_home = p
                elif n == away:
                    ml_away = p
    return total_line, over_juice, under_juice, ml_home, ml_away


def fetch_mlb_odds():
    """
    Fetch MLB odds from The Odds API (US, American odds, h2h + totals).
    Returns dict: game_key -> { total_line, over_juice, under_juice, ml_home, ml_away, book }
    """
    if not ODDS_API_KEY:
        logging.warning("⚠️ ODDS_API_KEY not set; odds API disabled.")
        return {}

    url = f"{ODDS_BASE_URL}/sports/{MLB_SPORT_KEY}/odds"
    params = {
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "apiKey": ODDS_API_KEY,
    }
    try:
        import requests
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logging.warning(f"⚠️ Odds API request failed: {e}")
        return {}

    result = {}
    for event in data:
        home = (event.get("home_team") or "").strip()
        away = (event.get("away_team") or "").strip()
        key = _game_key(away, home)
        if not key or key in result:
            continue

        best = None
        best_rank = len(BOOK_PRIORITY) + 1
        for book in event.get("bookmakers") or []:
            book_key = (book.get("key") or "").lower()
            try:
                rank = BOOK_PRIORITY.index(book_key)
            except ValueError:
                rank = len(BOOK_PRIORITY)
            if rank >= best_rank:
                continue
            has_totals = any((m.get("key") == "totals") for m in (book.get("markets") or []))
            if not has_totals:
                continue
            best_rank = rank
            best = book

        if best is None:
            for book in event.get("bookmakers") or []:
                if any((m.get("key") == "totals") for m in (book.get("markets") or [])):
                    best = book
                    break

        if best is None:
            result[key] = {
                "total_line": DEFAULT_TOTAL,
                "over_juice": DEFAULT_JUICE,
                "under_juice": DEFAULT_JUICE,
                "ml_home": None,
                "ml_away": None,
                "book": "",
            }
            continue

        # Inject home/away for h2h parsing
        best["_home"] = home
        best["_away"] = away
        total_line, over_juice, under_juice, ml_home, ml_away = _parse_totals_and_h2h(best)
        result[key] = {
            "total_line": total_line,
            "over_juice": over_juice,
            "under_juice": under_juice,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "book": (best.get("title") or best.get("key") or ""),
        }

    return result


def get_game_odds(away_team, home_team, odds_map=None):
    """
    Get odds for one game. odds_map from fetch_mlb_odds() or None to fetch now.
    Returns dict: total_line, over_juice, under_juice, ml_home, ml_away, book.
    Uses 8.5 / -110 / None / "" when missing.
    """
    if odds_map is None:
        odds_map = fetch_mlb_odds()
    key = _game_key(away_team, home_team)
    row = odds_map.get(key)
    if row:
        return {
            "total_line": float(row.get("total_line", DEFAULT_TOTAL)),
            "over_juice": int(row.get("over_juice", DEFAULT_JUICE)),
            "under_juice": int(row.get("under_juice", DEFAULT_JUICE)),
            "ml_home": row.get("ml_home"),
            "ml_away": row.get("ml_away"),
            "book": row.get("book") or "",
        }
    return {
        "total_line": DEFAULT_TOTAL,
        "over_juice": DEFAULT_JUICE,
        "under_juice": DEFAULT_JUICE,
        "ml_home": None,
        "ml_away": None,
        "book": "",
    }
