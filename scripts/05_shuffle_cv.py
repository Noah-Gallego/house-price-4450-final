import os
import sys
import time
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    FIGURES_DIR, CLEAN_DIR, RESULTS_DIR,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, PANEL,
    SPEC_COLS, IMAGE_COLS, style_axes,
)
from models import (
    fit_linear, predict_linear,
    knn_predict, fit_scaler, apply_scaler,
)

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlecolor": CSUB_BLUE,
})

N_TRIALS = 500
P_TRAIN, P_VAL = 0.65, 0.15
KNN_K = 15
WORKERS = os.cpu_count() or 4
CACHE = os.path.join(CLEAN_DIR, "shuffle_cv.npz")


def one_shuffle_split(n, rng):
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(round(n * P_TRAIN)); n_va = int(round(n * P_VAL))
    return idx[:n_tr], idx[n_tr:n_tr + n_va]


def city_target_mean_vec(prices_all, citi_all, tr_idx):
    tr_p = prices_all[tr_idx]
    tr_c = citi_all[tr_idx]
    grouped = pd.Series(tr_p).groupby(tr_c).mean()
    g = float(tr_p.mean())
    return np.asarray([grouped.get(c, g) for c in citi_all], dtype=float)


def trial_mae(X, y, tr, va, kind):
    if kind == "linear":
        w = fit_linear(X[tr], y[tr]); p = predict_linear(w, X[va])
    elif kind == "knn":
        mu, sd = fit_scaler(X[tr])
        Xtr = apply_scaler(X[tr], mu, sd); Xva = apply_scaler(X[va], mu, sd)
        p = knn_predict(Xtr, y[tr], Xva, KNN_K)
    return float(np.mean(np.abs(p - y[va])))


def _run_trial(args):
    t, spec_num, img, y, citi, n = args
    rng = np.random.default_rng(42 + t)
    tr, va = one_shuffle_split(n, rng)
    ctm = city_target_mean_vec(y, citi, tr).reshape(-1, 1)
    X_specs = np.hstack([spec_num, ctm])
    X_full  = np.hstack([X_specs, img])
    result = {}
    for k in ("linear", "knn"):
        result[k] = {
            "specs": trial_mae(X_specs, y, tr, va, k),
            "specs_image": trial_mae(X_full,  y, tr, va, k),
        }
    return t, result


def main():
    if os.path.exists(CACHE):
        print(f"{CACHE} exists, skipping shuffle CV")
        return

    df = load_features().reset_index(drop=True)
    y = df["price"].to_numpy(dtype=float)
    citi = df["citi"].to_numpy()
    spec_num = df[["bed", "bath", "sqft"]].to_numpy(dtype=float)
    img = df[IMAGE_COLS].to_numpy(dtype=float)
    n = len(df)

    kinds = ["linear", "knn"]
    out = {k: {"specs": [], "specs_image": []} for k in kinds}

    t0 = time.time()
    task_args = [(t, spec_num, img, y, citi, n) for t in range(N_TRIALS)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_run_trial, a) for a in task_args]
        done = 0
        for fut in futures:
            t_idx, result = fut.result()
            for k in kinds:
                out[k]["specs"].append(result[k]["specs"])
                out[k]["specs_image"].append(result[k]["specs_image"])
            done += 1
            if done % 50 == 0 or done == N_TRIALS:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (N_TRIALS - done) / rate
                print(f"trial {done}/{N_TRIALS}  elapsed={elapsed:.1f}s  eta={eta/60:.1f}min", flush=True)

    arrs = {k: {f: np.asarray(v) for f, v in d.items()} for k, d in out.items()}
    os.makedirs(CLEAN_DIR, exist_ok=True)
    np.savez(CACHE,
             linear_specs=arrs["linear"]["specs"],
             linear_specs_image=arrs["linear"]["specs_image"],
             knn_specs=arrs["knn"]["specs"],
             knn_specs_image=arrs["knn"]["specs_image"])

    summary = {}
    for k in kinds:
        for f in ("specs", "specs_image"):
            a = arrs[k][f]
            summary[f"{k}__{f}"] = {
                "mean_mae": float(a.mean()),
                "median_mae": float(np.median(a)),
                "p05": float(np.percentile(a, 5)),
                "p95": float(np.percentile(a, 95)),
                "n_trials": int(len(a)),
            }
    print("\nshuffle CV summary (MAE in dollars):")
    for k, v in summary.items():
        print(f"  {k:25s}  mean={v['mean_mae']:>10,.0f}  median={v['median_mae']:>10,.0f}  p05={v['p05']:>10,.0f}  p95={v['p95']:>10,.0f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "shuffle_cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=170)
    data = [arrs[k]["specs"] / 1000 for k in kinds]
    bp = ax.boxplot(data, labels=kinds, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.4),
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.35, markeredgecolor=MUTED, markerfacecolor=MUTED),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    boxprops=dict(facecolor="#B7C5E2", edgecolor=CSUB_BLUE, linewidth=1.2))
    medians_k = [float(np.median(a)) for a in data]
    ax.plot(range(1, len(kinds) + 1), medians_k, color=CSUB_GOLD, linewidth=2.2,
            marker="o", markersize=7, markerfacecolor=CSUB_GOLD, markeredgecolor=CSUB_GOLD, zorder=5)
    ax.set_ylabel("validation MAE ($k)")
    ax.set_title("")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig05_baseline_boxplot.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.6), dpi=170)
    positions, labels, data, colors = [], [], [], []
    for i, k in enumerate(kinds):
        positions += [i * 3 + 1, i * 3 + 2]
        labels += [f"{k}\nspecs", f"{k}\nspecs+image"]
        data += [arrs[k]["specs"] / 1000, arrs[k]["specs_image"] / 1000]
        colors += ["#B7C5E2", CSUB_GOLD]
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.4),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3, markeredgecolor=MUTED, markerfacecolor=MUTED),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_edgecolor(CSUB_BLUE)
    ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("validation MAE ($k)")
    ax.set_title("")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig11_specs_vs_image_boxplot.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("saved fig05 and fig11.")


if __name__ == "__main__":
    main()
