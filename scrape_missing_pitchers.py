from pybaseball import playerid_lookup, pitching_stats
import pandas as pd
from datetime import datetime
import time
import json


# Load existing fallback to avoid duplicates
try:
    existing_fallback = pd.read_csv("data/manual_fallback_pitchers.csv")
    existing_names = set(existing_fallback["Name"].str.lower())
except FileNotFoundError:
    existing_names = set()

# Load names from JSON file
with open("data/missing_pitchers.json") as f:
    all_missing = json.load(f)

# Remove names already in fallback
missing = [name for name in all_missing if name.lower() not in existing_names]

fallback_stats = []

stats = pitching_stats(datetime.now().year)

for name in missing:
    try:
        first, last = name.split(" ", 1)
        lookup = playerid_lookup(last, first)
        if lookup.empty:
            print(f"❌ Not found: {name}")
            continue
        row = stats[stats['Name'].str.lower() == name.lower()]
        if not row.empty:
            record = row[['Name', 'xERA', 'WHIP', 'IP']].iloc[0]
            fallback_stats.append(record)
            print(f"✅ Collected: {name}")
        else:
            print(f"⚠️ No match in stats: {name}")
        time.sleep(1.5)
    except Exception as e:
        print(f"❌ Error for {name}: {e}")

df_fallback = pd.DataFrame(fallback_stats)
df_fallback.to_csv("data/manual_fallback_pitchers.csv", index=False)
print("\n💾 Saved to data/manual_fallback_pitchers.csv")
