import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    compute_metrics, SPEC_COLS, CLEAN_DIR, RESULTS_DIR,
)
from models import (
    fit_linear, predict_linear,
    fit_tree, predict_tree,
    knn_predict, fit_scaler, apply_scaler,
)


def eval_combo(X, y, tr, va, *, kind, tree_depth=15, knn_k=15):
    if kind == "linear":
        w = fit_linear(X[tr], y[tr]); p = predict_linear(w, X[va])
    elif kind == "tree":
        nodes = fit_tree(X[tr], y[tr], max_depth=tree_depth, min_leaf=10)
        p = predict_tree(nodes, X[va])
    else:
        mu, sd = fit_scaler(X[tr])
        Xtr = apply_scaler(X[tr], mu, sd); Xva = apply_scaler(X[va], mu, sd)
        p = knn_predict(Xtr, y[tr], Xva, k=knn_k)
    return compute_metrics(y[va], p)


def main():
    df = load_features().reset_index(drop=True)
    sp = pd.read_parquet(os.path.join(CLEAN_DIR, "features_spatial.parquet"))
    df = df.merge(sp, on="image_id", how="left")
    tr, va, te = load_splits()
    df, _, _ = attach_city_target_mean(df, tr)
    y = df["price"].to_numpy(dtype=float)

    spatial_cols = [c for c in sp.columns if c != "image_id"]
    sets = {
        "specs only":            df[SPEC_COLS].to_numpy(float),
        "specs + spatial (96)":  df[SPEC_COLS + spatial_cols].to_numpy(float),
        "spatial only (no specs)": df[spatial_cols].to_numpy(float),
    }

    rows = []
    for name, X in sets.items():
        for kind in ("linear", "tree", "knn"):
            depths = [10, 15, 20] if kind == "tree" else [None]
            ks     = [10, 15, 25, 50] if kind == "knn" else [None]
            best = None
            for d in depths:
                for k in ks:
                    m = eval_combo(X, y, tr, va, kind=kind,
                                   tree_depth=(d or 15), knn_k=(k or 15))
                    tag = (d, k)
                    if best is None or m["mae"] < best[1]["mae"]:
                        best = (tag, m)
            rows.append({"variant": name, "model": kind,
                         "tree_depth": best[0][0], "knn_k": best[0][1],
                         **best[1]})

    out = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(RESULTS_DIR, "spatial_comparison.csv"), index=False)

    print(f"{'variant':28s}{'model':8s}{'depth':>8s}{'k':>6s}{'MAE':>12s}{'within 50k':>14s}{'within 100k':>14s}")
    print("-" * 90)
    for _, r in out.iterrows():
        depth = "-" if pd.isna(r["tree_depth"]) else int(r["tree_depth"])
        kk    = "-" if pd.isna(r["knn_k"])      else int(r["knn_k"])
        print(f"{r['variant']:28s}{r['model']:8s}{str(depth):>8}{str(kk):>6}{r['mae']:>12,.0f}{r['within_50k']:>14.1%}{r['within_100k']:>14.1%}")

    payload = {}
    for _, r in out.iterrows():
        key = f"{r['variant'].replace(' ', '_').replace('(', '').replace(')','')}_{r['model']}"
        payload[key] = {"mae": float(r["mae"]),
                        "within_50k": float(r["within_50k"]),
                        "within_100k": float(r["within_100k"]),
                        "rmse": float(r["rmse"]),
                        "tree_depth": (None if pd.isna(r["tree_depth"]) else int(r["tree_depth"])),
                        "knn_k":      (None if pd.isna(r["knn_k"])      else int(r["knn_k"]))}
    with open(os.path.join(RESULTS_DIR, "spatial_comparison.json"), "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
