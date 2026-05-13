import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    compute_metrics, save_metrics,
    SPEC_COLS, IMAGE_COLS, RESULTS_DIR,
)
from models import (
    fit_linear, predict_linear,
    fit_tree, predict_tree,
    knn_predict, fit_scaler, apply_scaler,
)


def main():
    df = load_features().reset_index(drop=True)
    tr, va, te = load_splits()
    df, _, _ = attach_city_target_mean(df, tr)

    # use train + val together for the final fit, since val decisions are locked
    tv = np.concatenate([tr, va])
    y = df["price"].to_numpy(dtype=float)

    X_specs = df[SPEC_COLS].to_numpy(dtype=float)
    X_full  = df[SPEC_COLS + IMAGE_COLS].to_numpy(dtype=float)

    # constant baseline, predicts train+val mean on test
    const = float(y[tv].mean())
    pred_const = np.full(len(te), const)
    m_const = compute_metrics(y[te], pred_const)

    # specs-only linear (MLR)
    w = fit_linear(X_specs[tv], y[tv])
    pred_lin_s = predict_linear(w, X_specs[te])
    m_lin_s = compute_metrics(y[te], pred_lin_s)

    # specs+image linear
    w_full = fit_linear(X_full[tv], y[tv])
    pred_lin_f = predict_linear(w_full, X_full[te])
    m_lin_f = compute_metrics(y[te], pred_lin_f)

    # specs-only tree, depth chosen on val = 15
    nodes_tree_s = fit_tree(X_specs[tv], y[tv], max_depth=15, min_leaf=10)
    pred_tree_s = predict_tree(nodes_tree_s, X_specs[te])
    m_tree_s = compute_metrics(y[te], pred_tree_s)

    # specs+image tree, depth chosen on val = 8
    nodes_tree_f = fit_tree(X_full[tv], y[tv], max_depth=8, min_leaf=10)
    pred_tree_f = predict_tree(nodes_tree_f, X_full[te])
    m_tree_f = compute_metrics(y[te], pred_tree_f)

    # specs-only KNN, k = 15
    mu_s, sd_s = fit_scaler(X_specs[tv])
    Xs_tv = apply_scaler(X_specs[tv], mu_s, sd_s)
    Xs_te = apply_scaler(X_specs[te], mu_s, sd_s)
    pred_knn_s = knn_predict(Xs_tv, y[tv], Xs_te, k=15)
    m_knn_s = compute_metrics(y[te], pred_knn_s)

    # specs+image KNN, k = 10
    mu_f, sd_f = fit_scaler(X_full[tv])
    Xf_tv = apply_scaler(X_full[tv], mu_f, sd_f)
    Xf_te = apply_scaler(X_full[te], mu_f, sd_f)
    pred_knn_f = knn_predict(Xf_tv, y[tv], Xf_te, k=10)
    m_knn_f = compute_metrics(y[te], pred_knn_f)

    payload = {
        "test_constant_predictor": m_const,
        "test_linear_specs":        m_lin_s,
        "test_linear_specs_image":  m_lin_f,
        "test_tree_specs_depth15":  m_tree_s,
        "test_tree_specs_image_depth8": m_tree_f,
        "test_knn_specs_k15":       m_knn_s,
        "test_knn_specs_image_k10": m_knn_f,
        "test_set_size": int(len(te)),
        "trainval_set_size": int(len(tv)),
    }
    save_metrics(payload)
    with open(os.path.join(RESULTS_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    rows = [
        ("constant (train+val mean)",  "none",         m_const),
        ("linear regression",          "specs",        m_lin_s),
        ("linear regression",          "specs+image",  m_lin_f),
        ("decision tree (d=15)",       "specs",        m_tree_s),
        ("decision tree (d=8)",        "specs+image",  m_tree_f),
        ("KNN (k=15)",                 "specs",        m_knn_s),
        ("KNN (k=10)",                 "specs+image",  m_knn_f),
    ]
    print(f"{'model':32s}{'features':14s}{'MAE':>12s}{'RMSE':>14s}{'<$50k':>10s}{'<$100k':>10s}{'<$200k':>10s}")
    print("-" * 102)
    for name, feat, m in rows:
        print(f"{name:32s}{feat:14s}"
              f"{m['mae']:>12,.0f}{m['rmse']:>14,.0f}"
              f"{m['within_50k']:>10.2%}{m['within_100k']:>10.2%}{m['within_200k']:>10.2%}")


if __name__ == "__main__":
    main()
