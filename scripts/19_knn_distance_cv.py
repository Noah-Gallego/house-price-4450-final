import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, FIGURES_DIR, RESULTS_DIR, CLEAN_DIR, CITIES_GEO,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)
from models import knn_predict, fit_scaler, apply_scaler

N_TRIALS = 200
K_VALUES = [1, 3, 5, 7, 10, 15, 20, 30, 50]
P_TRAIN, P_VAL = 0.65, 0.15
METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]


def one_shuffle(n, rng):
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(round(n * P_TRAIN)); n_va = int(round(n * P_VAL))
    return idx[:n_tr], idx[n_tr:n_tr + n_va]


def main():
    df = load_features().reset_index(drop=True)
    y = df["price"].to_numpy(dtype=float)
    citi = df["citi"].to_numpy()
    spec_num = df[["bed", "bath", "sqft"]].to_numpy(dtype=float)
    n = len(df)

    geo = pd.read_csv(CITIES_GEO)
    lat_map = dict(zip(geo["citi"], geo["lat"]))
    lon_map = dict(zip(geo["citi"], geo["lon"]))
    city_lat_all = np.asarray([lat_map.get(c, np.nan) for c in citi], dtype=float)
    city_lon_all = np.asarray([lon_map.get(c, np.nan) for c in citi], dtype=float)

    out = {m: {k: [] for k in K_VALUES} for m in METRICS}

    rng = np.random.default_rng(42)
    t0 = time.time()
    for t in range(N_TRIALS):
        tr, va = one_shuffle(n, rng)
        tr_price = y[tr]
        gm = float(tr_price.mean())
        tr_c = pd.Series(tr_price).groupby(citi[tr]).mean()
        ctm = np.asarray([tr_c.get(c, gm) for c in citi], dtype=float)
        # fill missing lat/lon with train mean of available coords
        gl_lat = float(np.nanmean(city_lat_all[tr])) if np.isfinite(city_lat_all[tr]).any() else 34.0
        gl_lon = float(np.nanmean(city_lon_all[tr])) if np.isfinite(city_lon_all[tr]).any() else -118.0
        lat_col = np.where(np.isnan(city_lat_all), gl_lat, city_lat_all)
        lon_col = np.where(np.isnan(city_lon_all), gl_lon, city_lon_all)
        X = np.hstack([spec_num, ctm.reshape(-1, 1), lat_col.reshape(-1, 1), lon_col.reshape(-1, 1)])
        mu, sd = fit_scaler(X[tr])
        Xtr = apply_scaler(X[tr], mu, sd)
        Xva = apply_scaler(X[va], mu, sd)
        for m in METRICS:
            for k in K_VALUES:
                pred = knn_predict(Xtr, y[tr], Xva, k=k, metric=m, p=3)
                out[m][k].append(float(np.mean(np.abs(pred - y[va]))))
        if (t + 1) % 25 == 0 or t == N_TRIALS - 1:
            print(f"trial {t+1}/{N_TRIALS}  elapsed={time.time()-t0:.1f}s", flush=True)

    # save raw
    os.makedirs(CLEAN_DIR, exist_ok=True)
    flat = {}
    for m in METRICS:
        for k in K_VALUES:
            flat[f"{m}_k{k}"] = np.asarray(out[m][k])
    np.savez(os.path.join(CLEAN_DIR, "knn_distance_cv.npz"), **flat)

    # summary
    summary = {}
    for m in METRICS:
        per_k = {k: float(np.median(out[m][k])) for k in K_VALUES}
        best_k = min(per_k, key=per_k.get)
        summary[m] = {"best_k": int(best_k),
                       "best_median_mae": float(per_k[best_k]),
                       "per_k_median": {str(k): v for k, v in per_k.items()}}
    with open(os.path.join(RESULTS_DIR, "knn_distance_cv.json"), "w") as f:
        json.dump(summary, f, indent=2)

    for m, s in summary.items():
        print(f"  {m:11s} best k = {s['best_k']:>2d}   median MAE = ${s['best_median_mae']/1000:.1f}k")

    # 2x2 grid of boxplots
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlecolor": CSUB_BLUE,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.0), dpi=160, sharey=True)
    for ax, m in zip(axes.ravel(), METRICS):
        data = [np.asarray(out[m][k]) / 1000 for k in K_VALUES]
        bp = ax.boxplot(data, tick_labels=[str(k) for k in K_VALUES],
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color=INK, linewidth=1.4),
                        flierprops=dict(marker="o", markersize=2.0, alpha=0.3,
                                        markeredgecolor=MUTED, markerfacecolor=MUTED),
                        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                        boxprops=dict(facecolor="#B7C5E2", edgecolor=CSUB_BLUE, linewidth=1.0))
        medians = [float(np.median(a)) for a in data]
        ax.plot(range(1, len(K_VALUES) + 1), medians, color=CSUB_GOLD,
                linewidth=1.8, marker="o", markersize=5,
                markerfacecolor=CSUB_GOLD, markeredgecolor=CSUB_GOLD, zorder=5)
        ax.set_title(m, color=CSUB_BLUE, fontsize=13)
        ax.set_xlabel("k")
        ax.set_ylabel("validation MAE ($k)")
        style_axes(ax)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig20_knn_distances.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
