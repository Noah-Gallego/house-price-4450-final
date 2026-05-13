import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    SPEC_COLS, IMAGE_COLS, FIGURES_DIR, RESULTS_DIR,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)
from models import fit_tree, predict_tree

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlecolor": CSUB_BLUE,
})


def main():
    df = load_features().reset_index(drop=True)
    tr, va, te = load_splits()
    df, _, _ = attach_city_target_mean(df, tr)
    y = df["price"].to_numpy(dtype=float)

    X_specs = df[SPEC_COLS].to_numpy(dtype=float)
    X_full  = df[SPEC_COLS + IMAGE_COLS].to_numpy(dtype=float)

    nodes_specs = fit_tree(X_specs[tr], y[tr], max_depth=15, min_leaf=10)
    nodes_full  = fit_tree(X_full [tr], y[tr], max_depth=8,  min_leaf=10)

    pred_s = predict_tree(nodes_specs, X_specs[va])
    pred_f = predict_tree(nodes_full,  X_full [va])
    val = df.iloc[va].copy()
    val["abs_s"] = np.abs(pred_s - y[va])
    val["abs_f"] = np.abs(pred_f - y[va])

    top = val["citi"].value_counts().head(10).index.tolist()
    rows = []
    for c in top:
        sub = val[val["citi"] == c]
        rows.append({"city": c.split(",")[0], "n": len(sub),
                     "mae_specs": float(sub["abs_s"].mean()),
                     "mae_image": float(sub["abs_f"].mean()),
                     "delta":     float(sub["abs_f"].mean() - sub["abs_s"].mean())})
    out = pd.DataFrame(rows).sort_values("mae_specs", ascending=False).reset_index(drop=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(RESULTS_DIR, "subgroup_top10_cities.csv"), index=False)
    print(out.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=170)
    y_pos = np.arange(len(out))[::-1]
    h = 0.36
    ax.barh(y_pos + h/2, out["mae_specs"] / 1000, height=h, color=CSUB_BLUE, label="specs only")
    ax.barh(y_pos - h/2, out["mae_image"] / 1000, height=h, color=CSUB_GOLD, label="specs + image")
    ax.set_yticks(y_pos); ax.set_yticklabels(out["city"])
    ax.set_xlabel("validation MAE ($k)")
    ax.set_title("per-city MAE, decision tree, specs vs specs + image")
    style_axes(ax); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig12_per_city_delta.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=170)
    deltas = out["delta"] / 1000
    colors = [CSUB_BLUE if d < 0 else CSUB_GOLD for d in deltas]
    ax.barh(np.arange(len(out))[::-1], deltas, color=colors)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(np.arange(len(out))[::-1]); ax.set_yticklabels(out["city"])
    ax.set_xlabel("MAE change with image features ($k, negative = better)")
    ax.set_title("per-city delta: blue = image features helped, gold = image features hurt")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig13_per_city_delta_signed.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved subgroup figures.")


if __name__ == "__main__":
    main()
