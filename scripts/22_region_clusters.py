import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, CITIES_GEO, CLEAN_DIR, FIGURES_DIR,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)

CA_GEOJSON = os.path.join(CLEAN_DIR, "california.geojson")


def load_ca_polygons():
    with open(CA_GEOJSON) as f:
        d = json.load(f)
    coords = d["geometry"]["coordinates"]
    polys = []
    for poly in coords:
        ring = poly[0]
        polys.append(np.asarray(ring, dtype=float))
    return polys

K = 10
SEED = 42


def kmeans(X, k, max_iter=300, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(X)
    cent = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        d2 = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                cent[j] = X[mask].mean(axis=0)
    return labels, cent


def main():
    geo = pd.read_csv(CITIES_GEO)
    LAT_LO, LAT_HI = 32.0, 37.0
    LON_LO, LON_HI = -121.0, -114.0
    geo = geo[geo["lat"].between(LAT_LO, LAT_HI) & geo["lon"].between(LON_LO, LON_HI)].reset_index(drop=True)

    X = geo[["lat", "lon"]].to_numpy(dtype=float)
    labels, centers = kmeans(X, k=K)

    # order clusters by mean training price so region_id has weak monotonic meaning
    df = load_features().reset_index(drop=True)
    tr, _, _ = load_splits()
    train_prices = df["price"].iloc[tr]
    train_cities = df["citi"].iloc[tr]
    city_to_idx = {c: i for i, c in enumerate(geo["citi"])}
    city_train_mean = train_prices.groupby(train_cities).mean()
    cluster_price = {}
    for j in range(K):
        cities_in_j = geo.loc[labels == j, "citi"].tolist()
        prices = [city_train_mean.get(c, np.nan) for c in cities_in_j]
        prices = [p for p in prices if not np.isnan(p)]
        cluster_price[j] = float(np.mean(prices)) if prices else 0.0
    order = sorted(range(K), key=lambda j: cluster_price[j])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(l)] for l in labels])
    centers = centers[order]

    geo["region_id"] = labels.astype(int)
    geo.to_csv(CITIES_GEO, index=False)

    palette = ["#003594", "#1B4FB5", "#3A6BCC", "#5C87DC", "#7AAEFF",
               "#FFD566", "#FDB913", "#E89B00", "#C77B00", "#8F5500"]
    cmap = ListedColormap(palette)

    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=160)
    polys = load_ca_polygons()
    for ring in polys:
        ax.fill(ring[:, 0], ring[:, 1], facecolor="#F2F4F8",
                edgecolor="#9AA3B2", linewidth=0.9, zorder=1)

    sc = ax.scatter(geo["lon"], geo["lat"], c=geo["region_id"], cmap=cmap,
                    s=45, edgecolor=INK, linewidth=0.4, alpha=0.9, zorder=3)
    ax.scatter(centers[:, 1], centers[:, 0], c="white", s=200, marker="X",
               edgecolor=INK, linewidth=1.5, zorder=5)
    for j in range(K):
        ax.annotate(f"R{j}", (centers[j, 1], centers[j, 0]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=INK, zorder=6)

    ax.set_xlim(-122.0, -114.0)
    ax.set_ylim(32.2, 36.0)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"K-means on city coordinates, k={K}",
                 fontsize=14, color=CSUB_BLUE, fontweight="bold", pad=12)
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig22_regions.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
    print(f"region_id added to {CITIES_GEO} ({len(geo)} cities)")
    for j in range(K):
        n_cities = int((labels == j).sum())
        print(f"  region {j}: {n_cities} cities, mean train price = ${cluster_price[order[j]]/1000:.0f}k")


if __name__ == "__main__":
    main()
