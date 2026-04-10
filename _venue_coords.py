"""Get all MLB venue names and coordinates from the Stats API — read-only."""
from statsapi import schedule

# Collect unique venues from a wide date range
venues = {}
for d in [
    "2026-04-03","2026-04-04","2026-04-05","2026-04-06","2026-04-07",
    "2026-04-08","2026-04-09","2026-04-10",
]:
    games = schedule(start_date=d, end_date=d) or []
    for g in games:
        vn = g.get("venue_name", "")
        vid = g.get("venue_id", "")
        if vn and vn not in venues:
            venues[vn] = vid

print(f"{len(venues)} unique venues found")
print()

# Now use the MLB API to get coordinates for each venue
import requests

for vn in sorted(venues.keys()):
    vid = venues[vn]
    try:
        r = requests.get(f"https://statsapi.mlb.com/api/v1/venues/{vid}", timeout=10)
        data = r.json().get("venues", [{}])[0]
        loc = data.get("location", {})
        coords = loc.get("defaultCoordinates", {})
        lat = coords.get("latitude", "?")
        lon = coords.get("longitude", "?")
        tz = loc.get("timeZone", {}).get("id", "?")
        dome = data.get("fieldInfo", {}).get("roofType", "?")
    except Exception as e:
        lat = lon = tz = dome = f"ERR:{e}"

    print(f"{vn:<45} vid={vid:<6} lat={lat:<12} lon={lon:<12} roof={dome:<15} tz={tz}")
