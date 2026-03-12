import pandas as pd
import os
import logging

PUBLIC_BETTING_FILE = "data/public_betting.csv"
TEAM_ALIASES = {
    "a's": "oakland athletics", "athletics": "oakland athletics",
    "guardians": "cleveland guardians", "cle": "cleveland guardians",
    "blue jays": "toronto blue jays", "tor": "toronto blue jays",
    "dbacks": "arizona diamondbacks", "diamondbacks": "arizona diamondbacks", "az": "arizona diamondbacks",
    "nats": "washington nationals", "nationals": "washington nationals", "was": "washington nationals",
    "red sox": "boston red sox", "bos": "boston red sox",
    "white sox": "chicago white sox", "chi white sox": "chicago white sox", "chw": "chicago white sox",
    "yankees": "new york yankees", "nyy": "new york yankees", "ny yankees": "new york yankees",
    "mets": "new york mets", "nym": "new york mets", "ny mets": "new york mets",
    "cubs": "chicago cubs", "chc": "chicago cubs", "chi cubs": "chicago cubs",
    "dodgers": "los angeles dodgers", "lad": "los angeles dodgers", "la dodgers": "los angeles dodgers",
    "angels": "los angeles angels", "laa": "los angeles angels", "la angels": "los angeles angels",
    "giants": "san francisco giants", "sf": "san francisco giants",
    "cardinals": "st. louis cardinals", "stl": "st. louis cardinals",
    "padres": "san diego padres", "sd": "san diego padres",
    "phillies": "philadelphia phillies", "phi": "philadelphia phillies",
    "braves": "atlanta braves", "atl": "atlanta braves",
    "marlins": "miami marlins", "mia": "miami marlins",
    "orioles": "baltimore orioles", "bal": "baltimore orioles",
    "tigers": "detroit tigers", "det": "detroit tigers",
    "pirates": "pittsburgh pirates", "pit": "pittsburgh pirates",
    "rockies": "colorado rockies", "col": "colorado rockies",
    "astros": "houston astros", "hou": "houston astros",
    "royals": "kansas city royals", "kc": "kansas city royals",
    "brewers": "milwaukee brewers", "mil": "milwaukee brewers",
    "twins": "minnesota twins", "min": "minnesota twins",
    "mariners": "seattle mariners", "sea": "seattle mariners",
    "rangers": "texas rangers", "tex": "texas rangers",
    "reds": "cincinnati reds", "cin": "cincinnati reds"
}

TEAM_ALIASES.update({
    "oakland": "oakland athletics",
    "cleveland": "cleveland guardians",
    "toronto": "toronto blue jays",
    "arizona": "arizona diamondbacks",
    "washington": "washington nationals",
    "boston": "boston red sox",
    "chicago white": "chicago white sox",
    "chicago cubs": "chicago cubs",
    "new york yankees": "new york yankees",
    "new york mets": "new york mets",
    "los angeles dodgers": "los angeles dodgers",
    "los angeles angels": "los angeles angels",
    "san francisco": "san francisco giants",
    "st. louis": "st. louis cardinals",
    "san diego": "san diego padres",
    "philadelphia": "philadelphia phillies",
    "atlanta": "atlanta braves",
    "miami": "miami marlins",
    "baltimore": "baltimore orioles",
    "detroit": "detroit tigers",
    "pittsburgh": "pittsburgh pirates",
    "colorado": "colorado rockies",
    "houston": "houston astros",
    "kansas city": "kansas city royals",
    "milwaukee": "milwaukee brewers",
    "minnesota": "minnesota twins",
    "seattle": "seattle mariners",
    "texas": "texas rangers",
    "cincinnati": "cincinnati reds"
})

def normalize_team_name(name):
    name = name.lower().strip()
    resolved = TEAM_ALIASES.get(name)
    if resolved:
        return resolved
    # Try partial match fallback if "athletics" → "oakland athletics"
    for alias, full in TEAM_ALIASES.items():
        if name in full or full in name:
            return full
    return name

def normalize_game_key(game_key):
    try:
        parts = game_key.lower().split(" @ ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid game_key format: {game_key}")
        away = normalize_team_name(parts[0])
        home = normalize_team_name(parts[1])
        print(f"🏷️ Normalized teams: {parts[0]} → {away}, {parts[1]} → {home}")
        return away, home
    except Exception as e:
        print(f"❌ normalize_game_key error: {e} ➤ Raw: {game_key}")
        return None, None

def load_public_betting_data():
    if not os.path.exists(PUBLIC_BETTING_FILE):
        logging.warning("⚠️ Public betting CSV not found.")
        return {}

    try:
        df = pd.read_csv(PUBLIC_BETTING_FILE)
    except Exception as e:
        logging.error(f"❌ Failed to load public betting CSV: {e}")
        return {}

    public_data = {}

    for _, row in df.iterrows():
        try:
            raw_key = str(row['Game']).strip().lower()
            away, home = normalize_game_key(raw_key)
            if not away or not home:
                continue
            key = f"{away} @ {home}"
            print(f"🔑 Parsed betting key: '{key}'")
            print(f"🔑 Public CSV Raw: {raw_key} → Normalized: {key}")

            try:
                total_open = float(str(row.get('Total_Open', 8.5)).strip() or 8.5)
            except Exception:
                total_open = 8.5
                print(f"⚠️ total_open fallback used for row: {row}")

            try:
                total_current = float(str(row.get('Total_Current', 8.5)).strip() or 8.5)
            except Exception:
                total_current = 8.5
                print(f"⚠️ total_current fallback used for row: {row}")

            public_data[key] = {
                "ml_bets_pct_home": int(float(row.get('ML_Bets_Home', 50) or 50)),
                "ml_bets_pct_away": int(float(row.get('ML_Bets_Away', 50) or 50)),
                "ou_bets_pct_over": int(float(row.get('OU_Bets_Over', 50) or 50)),
                "ou_bets_pct_under": int(float(row.get('OU_Bets_Under', 50) or 50)),
                "total_open": total_open,
                "total_current": total_current,
            }
        except Exception as e:
            logging.warning(f"⚠️ Skipped row due to parsing error: {e} | Row: {row}")

    logging.info(f"✅ Loaded public betting data for {len(public_data)} games")
    print("✅ Final betting keys loaded:")
    for k in public_data.keys():
        print(f"  ➤ {k}") 
    return public_data

def split_game_key(game_key: str) -> tuple[str, str] | None:
    try:
        parts = game_key.strip().lower().split(" @ ", 1)
        if len(parts) != 2:
            print(f"⚠️ split_game_key() failed: {game_key} → {parts}")
            return None
        return parts[0].strip(), parts[1].strip()
    except Exception as e:
        print(f"❌ split_game_key error: {e} → '{game_key}'")
        return None
