"""
SportsDataIO MLB client.
Fetches game slate and pre-game odds for a target date, normalized to the same
dict shape as odds_api: game_key -> { total_line, over_juice, under_juice, ml_home, ml_away, book }.
"""
import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SPORTSDATAIO_API_KEY = os.getenv("SPORTSDATAIO_API_KEY", "").strip()
BASE_URL = "https://api.sportsdata.io"
FORMAT = "json"
# Games by date: /v3/mlb/scores/{format}/GamesByDate/{date}
# Pre-game odds by date: /v3/mlb/odds/{format}/GameOddsByDate/{date}
GAMES_PATH = f"/v3/mlb/scores/{FORMAT}/GamesByDate"
ODDS_PATH = f"/v3/mlb/odds/{FORMAT}/GameOddsByDate"

# When multiple books exist for a game, pick first by this order (lowercase keys for matching).
BOOK_PRIORITY = (
    "pinnacle",
    "bovada",
    "betonlineag",
    "draftkings",
    "fanduel",
    "betmgm",
    "pointsbetus",
    "williamhill_us",
    "williamhill",
    "betus",
    "lowvig",
)

DEFAULT_TOTAL = 8.5
DEFAULT_JUICE = -110


def _game_key(away_team, home_team):
    """Build key for lookup: normalized 'away @ home' (match public_betting_loader / odds_api style)."""
    try:
        from core.public_betting_loader import normalize_team_name
        a = normalize_team_name(away_team or "")
        h = normalize_team_name(home_team or "")
        return f"{a} @ {h}"
    except Exception:
        return f"{(away_team or '').lower().strip()} @ {(home_team or '').lower().strip()}"


def _date_for_api(target_date_yyyy_mm_dd):
    """
    Return the date string to use in SportsDataIO URLs.
    SportsDataIO typically expects YYYY-MM-DD for GamesByDate / GameOddsByDate.
    Pass through as-is if already YYYY-MM-DD; otherwise try to parse and format.
    """
    if not target_date_yyyy_mm_dd:
        return ""
    s = target_date_yyyy_mm_dd.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s
    try:
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    except Exception:
        pass
    return s


def fetch_mlb_games_by_date(target_date_yyyy_mm_dd):
    """
    Fetch MLB games for the given date.
    target_date_yyyy_mm_dd: YYYY-MM-DD (or parseable equivalent).
    Returns (list of game objects, dict GameId -> normalized game_key).
    On failure returns ([], {}).
    """
    if not SPORTSDATAIO_API_KEY:
        print("[SportsDataIO] SPORTSDATAIO_API_KEY exists: False")
        logging.warning("SPORTSDATAIO_API_KEY not set; SportsDataIO disabled.")
        return [], {}

    date_str = _date_for_api(target_date_yyyy_mm_dd)
    if not date_str:
        print("[SportsDataIO] Invalid target date; cannot fetch games.")
        return [], {}

    url = f"{BASE_URL.rstrip('/')}{GAMES_PATH}/{date_str}"
    params = {"key": SPORTSDATAIO_API_KEY}

    print(f"[SportsDataIO] SPORTSDATAIO_API_KEY exists: True")
    print(f"[SportsDataIO] Games URL: {url}")

    try:
        import requests
        r = requests.get(url, params=params, timeout=15)
        ok = r.status_code == 200
        print(f"[SportsDataIO] Games request succeeded: {ok}, HTTP status: {r.status_code}")
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[SportsDataIO] Games request failed: {e}")
        logging.warning(f"SportsDataIO games request failed: {e}")
        return [], {}

    if not isinstance(data, list):
        print(f"[SportsDataIO] Games response is not a list: {type(data)}")
        return [], {}

    # Build GameId -> game_key. Support common field names (PascalCase / camelCase).
    game_id_to_key = {}
    for g in data:
        if not isinstance(g, dict):
            continue
        gid = g.get("GameID") or g.get("GameId")
        away = (g.get("AwayTeam") or g.get("AwayTeamName") or "").strip()
        home = (g.get("HomeTeam") or g.get("HomeTeamName") or "").strip()
        if gid is not None and (away or home):
            key = _game_key(away, home)
            game_id_to_key[int(gid)] = key
            game_id_to_key[str(gid)] = key

    print(f"[SportsDataIO] Number of games fetched: {len(data)}")
    print(f"[SportsDataIO] GameId -> game_key mapping size: {len(set(game_id_to_key.values()))}")

    return data, game_id_to_key


def fetch_mlb_odds_by_date(target_date_yyyy_mm_dd):
    """
    Fetch MLB game slate and pre-game odds for target_date (YYYY-MM-DD).
    Returns dict: game_key -> { total_line, over_juice, under_juice, ml_home, ml_away, book }.
    Uses same key format as odds_api so predictor can use either source.
    """
    games, game_id_to_key = fetch_mlb_games_by_date(target_date_yyyy_mm_dd)
    if not game_id_to_key:
        print("[SportsDataIO] No games for date; returning empty odds map.")
        return {}

    if not SPORTSDATAIO_API_KEY:
        return {}

    date_str = _date_for_api(target_date_yyyy_mm_dd)
    if not date_str:
        return {}

    url = f"{BASE_URL.rstrip('/')}{ODDS_PATH}/{date_str}"
    params = {"key": SPORTSDATAIO_API_KEY}

    print(f"[SportsDataIO] Odds URL: {url}")

    try:
        import requests
        r = requests.get(url, params=params, timeout=15)
        ok = r.status_code == 200
        print(f"[SportsDataIO] Odds request succeeded: {ok}, HTTP status: {r.status_code}")
        r.raise_for_status()
        odds_data = r.json()
    except Exception as e:
        print(f"[SportsDataIO] Odds request failed: {e}")
        logging.warning(f"SportsDataIO odds request failed: {e}")
        return {}

    # Odds can be list of rows (one per game per book) or list of game objects with nested odds.
    # Assume list of rows with GameId, Sportsbook, HomeMoneyLine, AwayMoneyLine, OverUnder, OverPayout, UnderPayout.
    if not isinstance(odds_data, list):
        print(f"[SportsDataIO] Odds response is not a list: {type(odds_data)}")
        return {}

    print(f"[SportsDataIO] Number of odds rows fetched: {len(odds_data)}")

    # Group by GameId, then by book; pick one book per game by BOOK_PRIORITY.
    by_game = {}
    for row in odds_data:
        if not isinstance(row, dict):
            continue
        gid = row.get("GameID") or row.get("GameId")
        if gid is None:
            continue
        gid_key = int(gid) if isinstance(gid, (int, float)) else gid
        game_key = game_id_to_key.get(int(gid_key)) or game_id_to_key.get(str(gid_key))
        if not game_key:
            continue
        book_name = (row.get("Sportsbook") or row.get("SportsbookName") or "").strip()
        book_key = (book_name or str(row.get("SportsbookId", ""))).lower().replace(" ", "").replace("_", "")

        if game_key not in by_game:
            by_game[game_key] = []
        by_game[game_key].append((book_key, book_name, row))

    # For each game, choose single book by priority.
    result = {}
    for game_key, book_rows in by_game.items():
        best_row = None
        best_rank = len(BOOK_PRIORITY) + 1
        for book_key, book_name, row in book_rows:
            try:
                rank = BOOK_PRIORITY.index(book_key)
            except ValueError:
                rank = len(BOOK_PRIORITY)
            if rank < best_rank:
                best_rank = rank
                best_row = (book_name, row)

        if best_row is None:
            best_row = (book_rows[0][1], book_rows[0][2])

        book_name, row = best_row
        over_under = row.get("OverUnder")
        over_payout = row.get("OverPayout")
        under_payout = row.get("UnderPayout")
        home_ml = row.get("HomeMoneyLine")
        away_ml = row.get("AwayMoneyLine")

        try:
            total_line = float(over_under) if over_under is not None else DEFAULT_TOTAL
        except (TypeError, ValueError):
            total_line = DEFAULT_TOTAL
        try:
            over_juice = int(over_payout) if over_payout is not None else DEFAULT_JUICE
        except (TypeError, ValueError):
            over_juice = DEFAULT_JUICE
        try:
            under_juice = int(under_payout) if under_payout is not None else DEFAULT_JUICE
        except (TypeError, ValueError):
            under_juice = DEFAULT_JUICE

        result[game_key] = {
            "total_line": total_line,
            "over_juice": over_juice,
            "under_juice": under_juice,
            "ml_home": int(home_ml) if home_ml is not None else None,
            "ml_away": int(away_ml) if away_ml is not None else None,
            "book": book_name or "",
        }

    print(f"[SportsDataIO] Number of normalized games in final odds map: {len(result)}")
    sample_keys = list(result.keys())[:3]
    print(f"[SportsDataIO] Sample odds map keys (3): {sample_keys}")
    for k in sample_keys:
        v = result.get(k, {})
        print(f"[SportsDataIO]   Sample value: total_line={v.get('total_line')}, over_juice={v.get('over_juice')}, under_juice={v.get('under_juice')}, ml_home={v.get('ml_home')}, ml_away={v.get('ml_away')}, book={repr(v.get('book'))}")

    return result


def get_game_odds(away_team, home_team, odds_map=None, target_date_yyyy_mm_dd=None):
    """
    Get odds for one game from an existing odds_map, or fetch by date.
    odds_map: optional pre-fetched dict from fetch_mlb_odds_by_date().
    target_date_yyyy_mm_dd: used only when odds_map is None, to fetch by date.
    Returns dict: total_line, over_juice, under_juice, ml_home, ml_away, book.
    """
    if odds_map is None and target_date_yyyy_mm_dd:
        odds_map = fetch_mlb_odds_by_date(target_date_yyyy_mm_dd)
    if odds_map is None:
        odds_map = {}
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
