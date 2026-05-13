import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    compute_metrics, SPEC_COLS, FIGURES_DIR, RESULTS_DIR,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)
from models import fit_linear, predict_linear, fit_minmax, apply_minmax


def main():
    df = load_features().reset_index(drop=True)
    tr, va, te = load_splits()
    df, _, _ = attach_city_target_mean(df, tr)
    y = df["price"].to_numpy(dtype=float)
    X = df[SPEC_COLS].to_numpy(dtype=float)

    lo, rng = fit_minmax(X[tr])
    Xtr = apply_minmax(X[tr], lo, rng)
    Xva = apply_minmax(X[va], lo, rng)

    w = fit_linear(Xtr, y[tr])
    pred = predict_linear(w, Xva)

    m = compute_metrics(y[va], pred)
    print("MLR on validation (6 features, min-max scaled):")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:14s} {v:>14,.2f}")
        else:
            print(f"  {k:14s} {v}")

    coefs = {"intercept": float(w[0])}
    for name, val in zip(SPEC_COLS, w[1:]):
        coefs[name] = float(val)
    payload = {"validation_metrics": m, "coefficients_minmax_space": coefs}
    with open(os.path.join(RESULTS_DIR, "mlr_v3.json"), "w") as f:
        json.dump(payload, f, indent=2)

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=160)
    actual_k = y[va] / 1000
    pred_k = pred / 1000
    ax.scatter(actual_k, pred_k, s=8, alpha=0.35,
               color=CSUB_BLUE, edgecolor="none")
    lim_lo = min(actual_k.min(), pred_k.min()) - 50
    lim_hi = max(actual_k.max(), pred_k.max()) + 50
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color=CSUB_GOLD, linewidth=2.0, label="perfect prediction")
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("actual price ($k)")
    ax.set_ylabel("predicted price ($k)")
    ax.set_title("MLR predictions on validation",
                 fontsize=14, color=CSUB_BLUE, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig23_mlr_pred_vs_actual.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
