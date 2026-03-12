import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

OUTPUT_FILE = "data/public_betting.csv"

def normalize_team_name(name):
    return name.strip().lower()

def scrape_public_betting():
    url = "https://www.covers.com/sports/mlb/matchups"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"❌ Failed to load Covers page. Status code: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, "html.parser")
    games = soup.select(".cmg-matchup-game")

    rows = []

    for game in games:
        try:
            teams = game.select_one(".cmg-matchup-team-row").text.lower()
            team_names = re.findall(r"[a-zA-Z\s\.]+", teams)
            if len(team_names) < 2:
                continue

            away = normalize_team_name(team_names[0])
            home = normalize_team_name(team_names[1])
            matchup = f"{away} @ {home}"

            ou_data = game.select_one(".cmg-ovun-consensus")
            ou_pct_over = ou_data.select_one(".consensus-graph-bar-left .percent").text.strip('%')
            ou_pct_under = ou_data.select_one(".consensus-graph-bar-right .percent").text.strip('%')

            ml_data = game.select_one(".cmg-matchup-consensus")
            ml_home = ml_data.select(".cmg-team")[1].select_one(".consensus__percentage").text.strip('%')
            ml_away = ml_data.select(".cmg-team")[0].select_one(".consensus__percentage").text.strip('%')

            total_text = game.select_one(".cmg-total").text
            total_open = total_current = re.findall(r"\d+\.\d+", total_text)[0]

            rows.append({
                "Game": matchup,
                "ML_Bets_Home": int(ml_home),
                "ML_Bets_Away": int(ml_away),
                "OU_Bets_Over": int(ou_pct_over),
                "OU_Bets_Under": int(ou_pct_under),
                "Total_Open": float(total_open),
                "Total_Current": float(total_current),
            })

        except Exception as e:
            print(f"⚠️ Skipped a game due to error: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} games to {OUTPUT_FILE}")

if __name__ == "__main__":
    scrape_public_betting()
