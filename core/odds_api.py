"""
The Odds API integration for MLB.
Loads ODDS_API_KEY from .env; fetches totals + moneylines (American odds).
Prefer sharp book first; return safe fallbacks when unavailable.
Target-date filtering: only keep events whose commence_time (in Mountain Time) matches target_date (YYYY-MM-DD).
"""
import os
import logging
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
MLB_SPORT_KEY = "baseball_mlb"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
# Prefer sharp / liquid books first; fall back to first available
BOOK_PRIORITY = (
    "pinnacle",
    "lowvig",  # if present in Odds API
    "betonlineag",
    "draftkings",
    "fanduel",
    "betrivers",
    "betmgm",
    "pointsbetus",
    "williamhill_us",
    "bovada",
)

DEFAULT_TOTAL = 8.5
DEFAULT_JUICE = -110
MT_ZONE = "America/Denver"


def _event_date_mt(commence_time_str):
    """
    Parse API commence_time (ISO UTC) and return date string YYYY-MM-DD in Mountain Time.
    Returns "" if parse fails.
    """
    if not commence_time_str:
        return ""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        mt = dt.astimezone(ZoneInfo(MT_ZONE))
        return mt.strftime("%Y-%m-%d")
    except Exception:
        return ""


def _commence_window_utc(target_date_yyyy_mm_dd):
    """
    For a given date in Mountain Time (YYYY-MM-DD), return (from_utc, to_utc) in ISO format
    for the API commenceTimeFrom / commenceTimeTo params. Covers full calendar day in MT.
    Returns (None, None) if target_date is invalid.
    """
    if not target_date_yyyy_mm_dd or len(target_date_yyyy_mm_dd) != 10:
        return None, None
    try:
        from zoneinfo import ZoneInfo
        y, m, d = int(target_date_yyyy_mm_dd[:4]), int(target_date_yyyy_mm_dd[5:7]), int(target_date_yyyy_mm_dd[8:10])
        day_start_mt = datetime(y, m, d, 0, 0, 0, tzinfo=ZoneInfo(MT_ZONE))
        day_end_mt = datetime(y, m, d, 23, 59, 59, tzinfo=ZoneInfo(MT_ZONE))
        utc_start = day_start_mt.astimezone(ZoneInfo("UTC"))
        utc_end = day_end_mt.astimezone(ZoneInfo("UTC"))
        from_utc = utc_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        to_utc = utc_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        return from_utc, to_utc
    except Exception:
        return None, None


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


def fetch_mlb_odds(target_date=None):
    """
    Fetch MLB odds from The Odds API (US, American odds, h2h + totals).
    target_date: optional YYYY-MM-DD string (e.g. predictor slate date in MT). If set, we pass
    commenceTimeFrom/commenceTimeTo so the API returns only events on that date (MT), and we keep
    only events whose commence_time falls on that date.
    Returns dict: game_key -> { total_line, over_juice, under_juice, ml_home, ml_away, book }.
    Note: The API returns only events that bookmakers have posted; there is no param to "request
    more events" for a date—if no odds exist for that date, the response will be empty.
    """
    print(f"[ODDS API] ODDS_API_KEY exists: {bool(ODDS_API_KEY)}")
    if target_date:
        print(f"[ODDS API] Target date filter: {target_date} (MT)")
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
    # Python-side target_date filtering only (no commenceTimeFrom/To on request) for backfill testing.
    if target_date:
        print(f"[ODDS API] Using Python-side target_date filtering only for: {target_date}")
    try:
        import requests
        r = requests.get(url, params=params, timeout=15)
        print(f"[ODDS API] Request succeeded: {r.status_code == 200}, HTTP status: {r.status_code}")
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            print("[ODDS API] Response JSON is not a list; returning empty odds_map.")
            return {}
        if len(data) == 0:
            print("[ODDS API] Empty events list from API; returning empty odds_map (no error).")
            return {}
        print(f"[ODDS API] Events returned: {len(data)}")
    except Exception as e:
        print(f"[ODDS API] Request failed: {e}")
        logging.warning(f"⚠️ Odds API request failed: {e}")
        return {}

    # Raw event inspection (first 5 events)
    events_to_inspect = data[:5] if isinstance(data, list) else []
    for i, ev in enumerate(events_to_inspect):
        away = (ev.get("away_team") or "").strip()
        home = (ev.get("home_team") or "").strip()
        commence = ev.get("commence_time", "")
        books = ev.get("bookmakers") or []
        book_names = [b.get("title") or b.get("key") or "?" for b in books]
        totals_per_book = []
        for b in books:
            has_totals = any((m.get("key") == "totals") for m in (b.get("markets") or []))
            totals_per_book.append(has_totals)
        any_totals = any(totals_per_book)
        print(f"[ODDS API] Event[{i}] away={repr(away)} home={repr(home)} commence_time={commence}")
        print(f"[ODDS API]   bookmakers ({len(books)}): {book_names}")
        print(f"[ODDS API]   totals exists per book: {totals_per_book}  any_totals={any_totals}")

    result = {}
    skip_log_cap = 3
    skip_log_count = 0
    no_totals_log_cap = 3
    no_totals_log_count = 0
    book_choice_log_cap = 3
    book_choice_log_count = 0
    date_skip_log_cap = 5
    date_skip_count = 0
    for event in data:
        commence_str = event.get("commence_time") or ""
        event_date_mt = _event_date_mt(commence_str)
        if target_date and event_date_mt != target_date:
            if date_skip_count < date_skip_log_cap:
                print(f"[ODDS API] Skip (wrong date): event_date_mt={event_date_mt} target={target_date} away={repr((event.get('away_team') or '').strip())} home={repr((event.get('home_team') or '').strip())}")
                date_skip_count += 1
            continue

        home = (event.get("home_team") or "").strip()
        away = (event.get("away_team") or "").strip()
        key = _game_key(away, home)
        if not key or key in result:
            if skip_log_count < skip_log_cap:
                reason = "empty key" if not key else "duplicate key (already in result)"
                print(f"[ODDS API] Skip reason (event {repr(away)} @ {repr(home)}): {reason}")
                skip_log_count += 1
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

        used_fallback_loop = False
        if best is None:
            for book in event.get("bookmakers") or []:
                if any((m.get("key") == "totals") for m in (book.get("markets") or [])):
                    best = book
                    used_fallback_loop = True
                    break

        if best is None:
            if no_totals_log_count < no_totals_log_cap:
                book_keys = [b.get("key") or "?" for b in (event.get("bookmakers") or [])]
                market_keys_per_book = []
                for b in (event.get("bookmakers") or []):
                    mkt = [m.get("key") for m in (b.get("markets") or [])]
                    market_keys_per_book.append((b.get("key"), mkt))
                print(f"[ODDS API] Event has no book with totals market: away={repr(away)} home={repr(home)}")
                print(f"[ODDS API]   books: {book_keys}; markets per book: {market_keys_per_book}")
                no_totals_log_count += 1
            result[key] = {
                "total_line": DEFAULT_TOTAL,
                "over_juice": DEFAULT_JUICE,
                "under_juice": DEFAULT_JUICE,
                "ml_home": None,
                "ml_away": None,
                "book": "",
            }
            continue

        if book_choice_log_count < book_choice_log_cap:
            book_name = best.get("title") or best.get("key") or "?"
            print(f"[ODDS API] Event parsed: away={repr(away)} home={repr(home)} key={repr(key)} book={book_name} (from_fallback_loop={used_fallback_loop})")
            book_choice_log_count += 1

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

    print(f"[ODDS API] Games parsed into odds_map: {len(result)} (from {len(data)} API events)")
    if target_date and len(result) == 0:
        print(f"[ODDS API] No odds found for target date {target_date}; returning empty odds_map.")
    sample_keys = list(result.keys())[:3]
    print(f"[ODDS API] Sample odds_map keys (3): {sample_keys}")
    for k in sample_keys:
        v = result.get(k, {})
        print(f"[ODDS API]   Sample value: total_line={v.get('total_line')}, over_juice={v.get('over_juice')}, under_juice={v.get('under_juice')}, book={repr(v.get('book'))}")
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
        raw_line = row.get("total_line")
        if raw_line is None or raw_line == "":
            total_line = DEFAULT_TOTAL
        else:
            try:
                total_line = float(raw_line)
            except (TypeError, ValueError):
                total_line = DEFAULT_TOTAL
        raw_over = row.get("over_juice")
        if raw_over is None or raw_over == "":
            over_juice = DEFAULT_JUICE
        else:
            try:
                over_juice = int(raw_over)
            except (TypeError, ValueError):
                over_juice = DEFAULT_JUICE
        raw_under = row.get("under_juice")
        if raw_under is None or raw_under == "":
            under_juice = DEFAULT_JUICE
        else:
            try:
                under_juice = int(raw_under)
            except (TypeError, ValueError):
                under_juice = DEFAULT_JUICE
        return {
            "total_line": total_line,
            "over_juice": over_juice,
            "under_juice": under_juice,
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
