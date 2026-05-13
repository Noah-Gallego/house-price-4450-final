import os
import sys
import time
import json
import urllib.parse
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import CITIES_GEO

API = "https://geocode.maps.co/search"
SLEEP = 1.05

# SoCal bounding box (generous)
LAT_LO, LAT_HI = 32.0, 37.0
LON_LO, LON_HI = -121.0, -114.0


def load_env_key():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(repo, ".env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MAPS_CO_API_KEY="):
                return line.split("=", 1)[1]
    raise SystemExit("MAPS_CO_API_KEY not in .env")


def geocode_strict(city_name, key):
    # city_name is like "Alpine, CA"; we strip the suffix and use structured search
    base = city_name.split(",")[0].strip()
    qs = urllib.parse.urlencode({
        "city": base, "state": "California", "country": "USA",
        "api_key": key,
    })
    url = f"{API}?{qs}"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    # filter to socal box
    for item in data:
        lat = float(item["lat"]); lon = float(item["lon"])
        if LAT_LO <= lat <= LAT_HI and LON_LO <= lon <= LON_HI:
            return lat, lon
    # fallback: free-form with USA suffix
    free = urllib.parse.quote_plus(f"{base}, California, USA")
    with urllib.request.urlopen(f"{API}?q={free}&api_key={key}", timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    for item in data:
        lat = float(item["lat"]); lon = float(item["lon"])
        if LAT_LO <= lat <= LAT_HI and LON_LO <= lon <= LON_HI:
            return lat, lon
    return None, None


def main():
    key = load_env_key()
    geo = pd.read_csv(CITIES_GEO)
    bad_mask = ~(geo["lat"].between(LAT_LO, LAT_HI) & geo["lon"].between(LON_LO, LON_HI))
    bad = geo[bad_mask].copy()
    print(f"{len(bad)} cities to re-fix")
    for _, row in bad.iterrows():
        city = row["citi"]
        try:
            lat, lon = geocode_strict(city, key)
            if lat is None:
                print(f"  could not place {city} inside SoCal box, leaving as-is")
                time.sleep(SLEEP); continue
            geo.loc[geo["citi"] == city, ["lat", "lon"]] = [lat, lon]
            print(f"  fixed {city}: ({lat:.4f}, {lon:.4f})")
        except Exception as e:
            print(f"  failed {city}: {e}")
        time.sleep(SLEEP)
    geo.to_csv(CITIES_GEO, index=False)
    still_bad = (~(geo["lat"].between(LAT_LO, LAT_HI) & geo["lon"].between(LON_LO, LON_HI))).sum()
    print(f"saved {CITIES_GEO}. {still_bad} still out of box.")


if __name__ == "__main__":
    main()
