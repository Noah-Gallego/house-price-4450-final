import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    compute_metrics, save_metrics, RESULTS_DIR, FIGURES_DIR,
    SPEC_COLS, IMAGE_COLS,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)
from models import (
    fit_linear, predict_linear,
    fit_tree, predict_tree,
    knn_predict, fit_scaler, apply_scaler,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlecolor": CSUB_BLUE,
})

TREE_DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20]
KNN_KS = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100]


def tune_tree(X_tr, y_tr, X_va, y_va):
    results = []
    for d in TREE_DEPTHS:
        nodes = fit_tree(X_tr, y_tr, max_depth=d, min_leaf=10)
        m = compute_metrics(y_va, predict_tree(nodes, X_va))
        m["depth"] = d
        results.append(m)
    best = min(results, key=lambda r: r["mae"])
    return best, results


def tune_knn(X_tr, y_tr, X_va, y_va):
    mu, sd = fit_scaler(X_tr)
    Xtr = apply_scaler(X_tr, mu, sd); Xva = apply_scaler(X_va, mu, sd)
    results = []
    for k in KNN_KS:
        pred = knn_predict(Xtr, y_tr, Xva, k)
        m = compute_metrics(y_va, pred)
        m["k"] = k
        results.append(m)
    best = min(results, key=lambda r: r["mae"])
    return best, results


def run(name, X, y, tr, va):
    out = {}
    w = fit_linear(X[tr], y[tr])
    out["linear"] = compute_metrics(y[va], predict_linear(w, X[va]))

    tree_best, tree_all = tune_tree(X[tr], y[tr], X[va], y[va])
    out["tree"] = tree_best
    out["tree_sweep"] = tree_all

    knn_best, knn_all = tune_knn(X[tr], y[tr], X[va], y[va])
    out["knn"] = knn_best
    out["knn_sweep"] = knn_all

    print(f"\n=== {name} ===")
    print(f"linear            MAE=${out['linear']['mae']:>10,.0f}   <100k={out['linear']['within_100k']:.1%}")
    print(f"tree  depth={tree_best['depth']:<2d}     MAE=${tree_best['mae']:>10,.0f}   <100k={tree_best['within_100k']:.1%}")
    print(f"knn   k={knn_best['k']:<3d}        MAE=${knn_best['mae']:>10,.0f}   <100k={knn_best['within_100k']:.1%}")
    return out


def main():
    df = load_features().reset_index(drop=True)
    tr, va, te = load_splits()
    df, _, _ = attach_city_target_mean(df, tr)
    y = df["price"].to_numpy(dtype=float)

    X_specs = df[SPEC_COLS].to_numpy(dtype=float)
    X_full  = df[SPEC_COLS + IMAGE_COLS].to_numpy(dtype=float)

    specs_only = run("specs only",    X_specs, y, tr, va)
    multimodal = run("specs + image", X_full,  y, tr, va)

    save_metrics({
        "tuned_specs_only_val":  specs_only,
        "tuned_multimodal_val":  multimodal,
    })

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=170)
    for label, sw, color in [
        ("specs only",  specs_only["tree_sweep"], CSUB_BLUE),
        ("specs+image", multimodal["tree_sweep"], CSUB_GOLD),
    ]:
        d_arr = [r["depth"] for r in sw]; m_arr = [r["mae"] / 1000 for r in sw]
        ax.plot(d_arr, m_arr, marker="o", color=color, linewidth=2, label=label)
    ax.set_xlabel("tree max depth"); ax.set_ylabel("validation MAE ($k)")
    ax.set_title("decision tree, depth swept on validation")
    style_axes(ax); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig09_tree_depth.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=170)
    for label, sw, color in [
        ("specs only",  specs_only["knn_sweep"], CSUB_BLUE),
        ("specs+image", multimodal["knn_sweep"], CSUB_GOLD),
    ]:
        ks = [r["k"] for r in sw]; m_arr = [r["mae"] / 1000 for r in sw]
        ax.plot(ks, m_arr, marker="o", color=color, linewidth=2, label=label)
    ax.set_xlabel("k (neighbors)"); ax.set_ylabel("validation MAE ($k)")
    ax.set_title("knn, k swept on validation")
    style_axes(ax); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig10_knn_k.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nsaved tuned-model metrics and depth/k sweep figures.")


if __name__ == "__main__":
    main()
