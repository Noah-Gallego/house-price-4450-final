import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    compute_metrics, save_metrics, SPEC_COLS, CLEAN_DIR, RESULTS_DIR,
)
from models import (
    fit_linear, predict_linear,
    fit_tree, predict_tree,
    knn_predict, fit_scaler, apply_scaler,
)


def main():
    df = load_features().reset_index(drop=True)
    sp = pd.read_parquet(os.path.join(CLEAN_DIR, "features_spatial.parquet"))
    df = df.merge(sp, on="image_id", how="left")
    tr, va, te = load_splits()
    df, _, _ = attach_city_target_mean(df, tr)
    spatial_cols = [c for c in sp.columns if c != "image_id"]

    tv = np.concatenate([tr, va])
    y = df["price"].to_numpy(dtype=float)

    X_specs = df[SPEC_COLS].to_numpy(float)
    X_full  = df[SPEC_COLS + spatial_cols].to_numpy(float)
    X_spatial_only = df[spatial_cols].to_numpy(float)

    const = float(y[tv].mean())
    m_const = compute_metrics(y[te], np.full(len(te), const))

    w = fit_linear(X_specs[tv], y[tv])
    m_lin_s = compute_metrics(y[te], predict_linear(w, X_specs[te]))

    w = fit_linear(X_full[tv], y[tv])
    m_lin_f = compute_metrics(y[te], predict_linear(w, X_full[te]))

    nodes = fit_tree(X_specs[tv], y[tv], max_depth=15, min_leaf=10)
    m_tree_s = compute_metrics(y[te], predict_tree(nodes, X_specs[te]))

    nodes = fit_tree(X_full[tv], y[tv], max_depth=10, min_leaf=10)
    m_tree_f = compute_metrics(y[te], predict_tree(nodes, X_full[te]))

    mu, sd = fit_scaler(X_specs[tv])
    pred_knn_s = knn_predict(apply_scaler(X_specs[tv], mu, sd), y[tv],
                             apply_scaler(X_specs[te], mu, sd), k=15)
    m_knn_s = compute_metrics(y[te], pred_knn_s)

    mu, sd = fit_scaler(X_full[tv])
    pred_knn_f = knn_predict(apply_scaler(X_full[tv], mu, sd), y[tv],
                             apply_scaler(X_full[te], mu, sd), k=15)
    m_knn_f = compute_metrics(y[te], pred_knn_f)

    mu, sd = fit_scaler(X_spatial_only[tv])
    pred_knn_so = knn_predict(apply_scaler(X_spatial_only[tv], mu, sd), y[tv],
                               apply_scaler(X_spatial_only[te], mu, sd), k=50)
    m_knn_so = compute_metrics(y[te], pred_knn_so)

    payload = {
        "test_constant_predictor":     m_const,
        "test_linear_specs":           m_lin_s,
        "test_linear_specs_spatial":   m_lin_f,
        "test_tree_specs_depth15":     m_tree_s,
        "test_tree_specs_spatial_d10": m_tree_f,
        "test_knn_specs_k15":          m_knn_s,
        "test_knn_specs_spatial_k15":  m_knn_f,
        "test_knn_spatial_only_k50":   m_knn_so,
        "test_set_size":     int(len(te)),
        "trainval_set_size": int(len(tv)),
    }
    save_metrics(payload)
    with open(os.path.join(RESULTS_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    rows = [
        ("constant (train+val mean)",  "none",            m_const),
        ("linear regression",          "specs",           m_lin_s),
        ("linear regression",          "specs+spatial",   m_lin_f),
        ("decision tree (d=15)",       "specs",           m_tree_s),
        ("decision tree (d=10)",       "specs+spatial",   m_tree_f),
        ("KNN (k=15)",                 "specs",           m_knn_s),
        ("KNN (k=15)",                 "specs+spatial",   m_knn_f),
    ]
    print(f"{'model':32s}{'features':16s}{'MAE':>12s}{'RMSE':>14s}{'<$50k':>10s}{'<$100k':>10s}")
    print("-" * 100)
    for name, feat, m in rows:
        print(f"{name:32s}{feat:16s}{m['mae']:>12,.0f}{m['rmse']:>14,.0f}"
              f"{m['within_50k']:>10.2%}{m['within_100k']:>10.2%}")


if __name__ == "__main__":
    main()
