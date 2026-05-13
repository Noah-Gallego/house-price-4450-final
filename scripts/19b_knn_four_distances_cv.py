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
from models import fit_minmax, apply_minmax

N_TRIALS = 100
K_VALUES = [1, 3, 5, 7, 10, 15, 20, 30, 50]
P_TRAIN, P_VAL = 0.65, 0.15
METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]
P_MINKOWSKI = 3


def one_shuffle(n, rng):
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(round(n * P_TRAIN)); n_va = int(round(n * P_VAL))
    return idx[:n_tr], idx[n_tr:n_tr + n_va]


def distance_matrix(Xq, Xt, metric):
    if metric == "euclidean":
        tn = (Xt * Xt).sum(axis=1)
        qn = (Xq * Xq).sum(axis=1)
        d = qn[:, None] + tn[None, :] - 2.0 * (Xq @ Xt.T)
        np.maximum(d, 0, out=d)
        return d
    n_q = Xq.shape[0]; n_t = Xt.shape[0]
    d = np.empty((n_q, n_t), dtype=np.float64)
    # chunk so peak memory stays bounded
    CHUNK = max(1, 30_000_000 // max(n_t * Xt.shape[1], 1))
    for s in range(0, n_q, CHUNK):
        e = min(n_q, s + CHUNK)
        diff = np.abs(Xq[s:e, None, :] - Xt[None, :, :])
        if metric == "manhattan":
            d[s:e] = diff.sum(axis=2)
        elif metric == "chebyshev":
            d[s:e] = diff.max(axis=2)
        elif metric == "minkowski":
            d[s:e] = (diff ** P_MINKOWSKI).sum(axis=2) ** (1.0 / P_MINKOWSKI)
    return d


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

    out = {m: {k: [] for k in K_VALUES} for m in METRICS}

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
        for m in METRICS:
            D = distance_matrix(Xva, Xtr, m)
            for k in K_VALUES:
                idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
                pred = y[tr][idx].mean(axis=1)
                out[m][k].append(float(np.mean(np.abs(pred - y[va]))))
        if (t + 1) % 10 == 0 or t == N_TRIALS - 1:
            print(f"trial {t+1}/{N_TRIALS}  elapsed={time.time()-t0:.1f}s", flush=True)

    flat = {}
    for m in METRICS:
        for k in K_VALUES:
            flat[f"{m}_k{k}"] = np.asarray(out[m][k])
    np.savez(os.path.join(CLEAN_DIR, "knn_4dist_cv.npz"), **flat)

    summary = {}
    for m in METRICS:
        per_k = {k: float(np.median(out[m][k])) for k in K_VALUES}
        best_k = min(per_k, key=per_k.get)
        summary[m] = {"best_k": int(best_k),
                       "best_median_mae": float(per_k[best_k]),
                       "per_k_median": {str(k): v for k, v in per_k.items()}}
    with open(os.path.join(RESULTS_DIR, "knn_4dist_cv.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for m, s in summary.items():
        print(f"  {m:11s} best k = {s['best_k']:>2d}   median MAE = ${s['best_median_mae']/1000:.1f}k", flush=True)

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
