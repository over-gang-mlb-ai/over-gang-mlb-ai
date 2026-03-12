from rapidfuzz import process, fuzz
import pandas as pd

# Load your files
df1 = pd.read_csv("data/pitcher_stats_with_xera_with_collumns.csv")
df_fallback = pd.read_csv("data/fallback_pitchers_with_collumns.csv")

# Fuzzy WHIP match
whip_lookup = dict(zip(df_fallback["Name"], df_fallback["WHIP"]))

def get_fuzzy_whip(name):
    match, score, _ = process.extractOne(name, whip_lookup.keys(), scorer=fuzz.token_sort_ratio)
    return whip_lookup[match] if score >= 85 else None

df1["WHIP"] = df1.apply(
    lambda row: get_fuzzy_whip(row["Name"]), axis=1
)

# Clean and round
df1["WHIP"] = pd.to_numeric(df1["WHIP"], errors="coerce").round(2)
df1["IP"] = pd.to_numeric(df1["IP"], errors="coerce").round(1)
df1["xERA"] = pd.to_numeric(df1["xERA"], errors="coerce").round(2)
df1["K/9"] = pd.to_numeric(df1["K/9"], errors="coerce").round(2)

# Map full team names
team_name_map = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
    "ATH": "Oakland Athletics", "- - -": "League Average"
}
df1["Team"] = df1["Team"].map(team_name_map).fillna(df1["Team"])

# Final export
df_final = df1[["Name", "Team", "IP", "WHIP", "xERA", "K/9"]].dropna()
df_final.to_csv("data/pitcher_stats.csv", index=False)
