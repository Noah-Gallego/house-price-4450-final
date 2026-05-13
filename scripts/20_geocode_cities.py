import os
import sys
import time
import json
import urllib.parse
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import RAW_CSV, CLEAN_DIR

CACHE = os.path.join(CLEAN_DIR, "cities_geo.csv")
API   = "https://geocode.maps.co/search"
SLEEP = 1.05  # service is 1 req/sec, give a small buffer


def load_env_key():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(repo, ".env")
    if not os.path.exists(env_path):
        raise SystemExit(f"missing {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("MAPS_CO_API_KEY="):
                return line.split("=", 1)[1]
    raise SystemExit("MAPS_CO_API_KEY not found in .env")


def geocode(city, key):
    q = urllib.parse.quote_plus(city)
    url = f"{API}?q={q}&api_key={key}"
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    if not data:
        return None, None
    top = data[0]
    return float(top["lat"]), float(top["lon"])


def main():
    key = load_env_key()
    df = pd.read_csv(RAW_CSV)
    cities = sorted(df["citi"].dropna().unique().tolist())
    print(f"{len(cities)} unique cities to geocode")

    os.makedirs(CLEAN_DIR, exist_ok=True)
    if os.path.exists(CACHE):
        cache_df = pd.read_csv(CACHE)
        done = set(cache_df["citi"].tolist())
        print(f"{len(done)} cached, {len(cities) - len(done)} to fetch")
    else:
        cache_df = pd.DataFrame(columns=["citi", "lat", "lon"])
        done = set()

    todo = [c for c in cities if c not in done]
    if not todo:
        print("nothing to do, cache complete")
        return

    new_rows = []
    t0 = time.time()
    for i, city in enumerate(todo):
        try:
            lat, lon = geocode(city, key)
            new_rows.append({"citi": city, "lat": lat, "lon": lon})
        except Exception as e:
            print(f"  failed {city}: {e}", flush=True)
            new_rows.append({"citi": city, "lat": None, "lon": None})
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - (i + 1)) / max(rate, 1e-6)
            print(f"  {i+1}/{len(todo)}  elapsed={elapsed:.0f}s  eta={eta/60:.1f}min", flush=True)
            # periodic checkpoint write
            merged = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
            merged.to_csv(CACHE, index=False)
        time.sleep(SLEEP)

    merged = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["citi"], keep="last")
    merged.to_csv(CACHE, index=False)
    print(f"saved {CACHE} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
