import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    compute_metrics, save_metrics, SPEC_COLS,
)
from models import fit_linear, predict_linear


def main():
    df = load_features()
    tr, va, te = load_splits()
    df, _, _ = attach_city_target_mean(df, tr)

    y = df["price"].to_numpy(dtype=float)
    X_specs = df[SPEC_COLS].to_numpy(dtype=float)

    y_tr = y[tr]; y_va = y[va]
    train_mean = float(y_tr.mean())

    const_pred = np.full_like(y_va, train_mean, dtype=float)
    m_const = compute_metrics(y_va, const_pred)

    # simple linear regression: sqft only
    sqft_tr = X_specs[tr, 2:3]; sqft_va = X_specs[va, 2:3]
    w_slr = fit_linear(sqft_tr, y_tr)
    m_slr = compute_metrics(y_va, predict_linear(w_slr, sqft_va))

    # multiple linear regression: all four specs
    w_mlr = fit_linear(X_specs[tr], y_tr)
    pred_mlr = predict_linear(w_mlr, X_specs[va])
    m_mlr = compute_metrics(y_va, pred_mlr)

    payload = {
        "baseline_constant_val": {"features": "none", **m_const, "train_mean": train_mean},
        "baseline_slr_val":      {"features": "sqft only", **m_slr,
                                  "intercept": float(w_slr[0]),
                                  "coef_sqft": float(w_slr[1])},
        "baseline_mlr_val":      {"features": "bed+bath+sqft+city_target_mean", **m_mlr,
                                  "intercept": float(w_mlr[0]),
                                  "coef": dict(zip(SPEC_COLS, [float(c) for c in w_mlr[1:]]))},
    }
    save_metrics(payload)

    def line(name, m):
        return (f"{name:30s}  MAE=${m['mae']:>10,.0f}  RMSE=${m['rmse']:>10,.0f}  "
                f"med=${m['median_abs']:>10,.0f}  <50k={m['within_50k']:.1%}  "
                f"<100k={m['within_100k']:.1%}  <200k={m['within_200k']:.1%}")

    print(line("constant (train mean)", m_const))
    print(line("SLR  (sqft)",          m_slr))
    print(line("MLR  (4 specs)",       m_mlr))
    print(f"\nMLR coef: {dict(zip(SPEC_COLS, [round(c, 2) for c in w_mlr[1:]]))}")
    print(f"intercept: {w_mlr[0]:,.0f}")


if __name__ == "__main__":
    main()
