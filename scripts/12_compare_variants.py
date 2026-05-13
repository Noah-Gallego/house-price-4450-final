import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    compute_metrics, SPEC_COLS, IMAGE_COLS, CLEAN_DIR, RESULTS_DIR,
)
from models import (
    fit_linear, predict_linear,
    fit_tree, predict_tree,
    knn_predict, fit_scaler, apply_scaler,
)


def eval_combo(X, y, tr, va, *, kind, tree_depth=15, knn_k=15):
    if kind == "linear":
        w = fit_linear(X[tr], y[tr])
        p = predict_linear(w, X[va])
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
    v2 = pd.read_parquet(os.path.join(CLEAN_DIR, "features_v2.parquet"))
    df = df.merge(v2, on="image_id", how="left")
    tr, va, te = load_splits()
    df, _, _ = attach_city_target_mean(df, tr)
    y = df["price"].to_numpy(dtype=float)

    # variant feature sets
    raw_image = IMAGE_COLS
    crop_image = [f"c_{c}" for c in IMAGE_COLS]
    norm_image = [f"n_{c}" for c in IMAGE_COLS]
    combo_image = [f"x_{c}" for c in IMAGE_COLS] + ["x_hue_mean", "x_hue_std", "x_sat_mean", "x_val_mean"]

    sets = {
        "specs only":           df[SPEC_COLS].to_numpy(float),
        "specs + raw image":    df[SPEC_COLS + raw_image].to_numpy(float),
        "specs + crop":         df[SPEC_COLS + crop_image].to_numpy(float),
        "specs + per-img norm": df[SPEC_COLS + norm_image].to_numpy(float),
        "specs + crop+norm+HSV": df[SPEC_COLS + combo_image].to_numpy(float),
    }

    rows = []
    for name, X in sets.items():
        for kind, depth, k in (("linear", None, None), ("tree", 15, None), ("knn", None, 15)):
            m = eval_combo(X, y, tr, va, kind=kind,
                           tree_depth=(depth or 15), knn_k=(k or 15))
            rows.append({"variant": name, "model": kind, **m})
    out = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(RESULTS_DIR, "variant_comparison.csv"), index=False)

    print(f"{'variant':28s}{'model':8s}{'MAE':>12s}{'within 50k':>14s}{'within 100k':>14s}")
    print("-" * 80)
    for _, r in out.iterrows():
        print(f"{r['variant']:28s}{r['model']:8s}{r['mae']:>12,.0f}{r['within_50k']:>14.1%}{r['within_100k']:>14.1%}")

    payload = {}
    for _, r in out.iterrows():
        key = f"{r['variant'].replace(' ', '_')}_{r['model']}"
        payload[key] = {"mae": float(r["mae"]), "within_50k": float(r["within_50k"]),
                        "within_100k": float(r["within_100k"]), "rmse": float(r["rmse"])}
    with open(os.path.join(RESULTS_DIR, "variant_comparison.json"), "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
