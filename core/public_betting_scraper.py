"""
Scrape Covers MLB consensus into data/public_betting.csv.

When target_date is set, only rows matching MLB's schedule for that calendar
date are written (single-day slate). Covers URL is not date-parameterized;
filtering uses statsapi.schedule for the target date.
"""
import os
import re
from datetime import date, datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from statsapi import schedule
from zoneinfo import ZoneInfo

from core.public_betting_loader import normalize_team_name

OUTPUT_FILE = "data/public_betting.csv"

_CSV_COLUMNS = [
    "Game",
    "ML_Bets_Home",
    "ML_Bets_Away",
    "OU_Bets_Over",
    "OU_Bets_Under",
    "Total_Open",
    "Total_Current",
]


def _slate_keys_for_date(target_date: date) -> set[str]:
    """Normalized 'away @ home' keys for MLB games on that calendar date."""
    ds = target_date.strftime("%Y-%m-%d")
    games = schedule(start_date=ds, end_date=ds) or []
    keys: set[str] = set()
    for g in games:
        an = normalize_team_name(str(g.get("away_name") or ""))
        hn = normalize_team_name(str(g.get("home_name") or ""))
        if an and hn:
            keys.add(f"{an} @ {hn}")
    return keys


def _row_key(game_cell: str):
    """Same normalization as loader keys; no console spam (avoid normalize_game_key prints)."""
    raw = str(game_cell).strip().lower()
    parts = raw.split(" @ ", 1)
    if len(parts) != 2:
        return None
    away = normalize_team_name(parts[0])
    home = normalize_team_name(parts[1])
    if not away or not home:
        return None
    return f"{away} @ {home}"


def scrape_public_betting(target_date: date | None = None) -> bool:
    """
    Scrape Covers matchups, keep only games on MLB schedule for target_date, write CSV.

    target_date: calendar date for the slate (use MT today or OVERGANG_TARGET_DATE from caller).
    If None, uses America/Denver today.

    Returns True if the CSV was written successfully (including empty slate).
    Returns False if scrape failed or rows could not be aligned; existing file is left unchanged.
    """
    try:
        if target_date is None:
            target_date = datetime.now(ZoneInfo("America/Denver")).date()
        elif isinstance(target_date, str):
            target_date = datetime.strptime(target_date.strip(), "%Y-%m-%d").date()

        allowed_keys = _slate_keys_for_date(target_date)
        ds = target_date.strftime("%Y-%m-%d")
        print(
            f"[PUBLIC BETTING] Target slate date: {ds} ({len(allowed_keys)} MLB game(s) on schedule)"
        )

        if not allowed_keys:
            pd.DataFrame(columns=_CSV_COLUMNS).to_csv(OUTPUT_FILE, index=False)
            print(f"✅ No MLB games on {ds}; wrote empty {OUTPUT_FILE} (headers only)")
            return True

        url = "https://www.covers.com/sports/mlb/matchups"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code != 200:
            print(
                f"❌ Public betting: Covers HTTP {response.status_code}; "
                f"not overwriting {OUTPUT_FILE}"
            )
            return False

        soup = BeautifulSoup(response.content, "html.parser")
        game_nodes = soup.select(".cmg-matchup-game")
        rows: list[dict] = []

        for game in game_nodes:
            try:
                teams = game.select_one(".cmg-matchup-team-row").text.lower()
                team_names = re.findall(r"[a-zA-Z\s\.]+", teams)
                if len(team_names) < 2:
                    continue

                away = normalize_team_name(team_names[0])
                home = normalize_team_name(team_names[1])
                matchup = f"{away} @ {home}"

                ou_data = game.select_one(".cmg-ovun-consensus")
                ou_pct_over = ou_data.select_one(".consensus-graph-bar-left .percent").text.strip("%")
                ou_pct_under = ou_data.select_one(".consensus-graph-bar-right .percent").text.strip("%")

                ml_data = game.select_one(".cmg-matchup-consensus")
                ml_home = ml_data.select(".cmg-team")[1].select_one(".consensus__percentage").text.strip("%")
                ml_away = ml_data.select(".cmg-team")[0].select_one(".consensus__percentage").text.strip("%")

                total_text = game.select_one(".cmg-total").text
                total_open = total_current = re.findall(r"\d+\.\d+", total_text)[0]

                rows.append(
                    {
                        "Game": matchup,
                        "ML_Bets_Home": int(ml_home),
                        "ML_Bets_Away": int(ml_away),
                        "OU_Bets_Over": int(ou_pct_over),
                        "OU_Bets_Under": int(ou_pct_under),
                        "Total_Open": float(total_open),
                        "Total_Current": float(total_current),
                    }
                )
            except Exception as e:
                print(f"⚠️ Skipped a Covers game due to error: {e}")
                continue

        if not rows:
            print(
                f"⚠️ Public betting: no games parsed from Covers; "
                f"not overwriting {OUTPUT_FILE}"
            )
            return False

        filtered: list[dict] = []
        for r in rows:
            k = _row_key(r["Game"])
            if k and k in allowed_keys:
                filtered.append(r)

        if not filtered:
            print(
                f"⚠️ Public betting: {len(rows)} Covers row(s) but none matched MLB schedule "
                f"for {ds}; not overwriting {OUTPUT_FILE}"
            )
            return False

        df = pd.DataFrame(filtered)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Saved {len(df)} game(s) for {ds} → {OUTPUT_FILE}")
        return True
    except Exception as e:
        print(f"❌ Public betting scrape failed: {e!r}; not overwriting {OUTPUT_FILE}")
        return False


if __name__ == "__main__":
    _tg = os.environ.get("OVERGANG_TARGET_DATE", "").strip()
    if _tg:
        try:
            scrape_public_betting(datetime.strptime(_tg, "%Y-%m-%d").date())
        except ValueError:
            print(f"⚠️ OVERGANG_TARGET_DATE={_tg!r} invalid; using MT today")
            scrape_public_betting(None)
    else:
        scrape_public_betting(None)
