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
from models import knn_predict, fit_minmax, apply_minmax

N_TRIALS = 500
K_VALUES = [1, 3, 5, 7, 10, 15, 20, 30, 50]
P_TRAIN, P_VAL = 0.65, 0.15


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
    reg_map = dict(zip(geo["citi"], geo["region_id"]))
    lat_col = np.asarray([lat_map.get(c, np.nan) for c in citi], dtype=float)
    lon_col = np.asarray([lon_map.get(c, np.nan) for c in citi], dtype=float)
    reg_col = np.asarray([reg_map.get(c, -1)     for c in citi], dtype=float)

    out = {k: [] for k in K_VALUES}

    rng = np.random.default_rng(42)
    t0 = time.time()
    for t in range(N_TRIALS):
        tr, va = one_shuffle(n, rng)
        gl_lat = float(np.nanmean(lat_col[tr])) if np.isfinite(lat_col[tr]).any() else 34.0
        gl_lon = float(np.nanmean(lon_col[tr])) if np.isfinite(lon_col[tr]).any() else -118.0
        lat_clean = np.where(np.isnan(lat_col), gl_lat, lat_col)
        lon_clean = np.where(np.isnan(lon_col), gl_lon, lon_col)
        reg_clean = np.where(reg_col < 0, 0, reg_col)
        X = np.hstack([spec_num,
                       lat_clean.reshape(-1, 1),
                       lon_clean.reshape(-1, 1),
                       reg_clean.reshape(-1, 1)])
        lo, rng_w = fit_minmax(X[tr])
        Xtr = apply_minmax(X[tr], lo, rng_w)
        Xva = apply_minmax(X[va], lo, rng_w)
        for k in K_VALUES:
            pred = knn_predict(Xtr, y[tr], Xva, k=k, metric="euclidean")
            out[k].append(float(np.mean(np.abs(pred - y[va]))))
        if (t + 1) % 50 == 0 or t == N_TRIALS - 1:
            print(f"trial {t+1}/{N_TRIALS}  elapsed={time.time()-t0:.1f}s", flush=True)

    flat = {f"k{k}": np.asarray(out[k]) for k in K_VALUES}
    np.savez(os.path.join(CLEAN_DIR, "knn_euclidean_cv.npz"), **flat)

    per_k_med = {k: float(np.median(out[k])) for k in K_VALUES}
    best_k = min(per_k_med, key=per_k_med.get)
    with open(os.path.join(RESULTS_DIR, "knn_euclidean_cv.json"), "w") as f:
        json.dump({"best_k": int(best_k),
                   "median_mae_by_k": {str(k): v for k, v in per_k_med.items()},
                   "n_trials": N_TRIALS}, f, indent=2)
    print(f"best k = {best_k}, median MAE = ${per_k_med[best_k]/1000:.0f}k")

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=160)
    data = [np.asarray(out[k]) / 1000 for k in K_VALUES]
    bp = ax.boxplot(data, tick_labels=[str(k) for k in K_VALUES],
                    patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.4),
                    flierprops=dict(marker="o", markersize=2.0, alpha=0.3,
                                    markeredgecolor=MUTED, markerfacecolor=MUTED),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    boxprops=dict(facecolor="#B7C5E2", edgecolor=CSUB_BLUE, linewidth=1.0))
    medians = [float(np.median(a)) for a in data]
    ax.plot(range(1, len(K_VALUES) + 1), medians, color=CSUB_GOLD,
            linewidth=2.2, marker="o", markersize=7,
            markerfacecolor=CSUB_GOLD, markeredgecolor=CSUB_GOLD, zorder=5)
    ax.set_xlabel("k", fontsize=13)
    ax.set_ylabel("validation MAE ($k)", fontsize=13)
    style_axes(ax)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig24_knn_euclidean.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
