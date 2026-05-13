import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, attach_city_target_mean,
    PICS_DIR, FIGURES_DIR, SPEC_COLS,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)
from models import fit_linear, predict_linear

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
    X = df[SPEC_COLS].to_numpy(dtype=float)

    w = fit_linear(X[tr], y[tr])
    pred_va = predict_linear(w, X[va])
    err = pred_va - y[va]
    abs_err = np.abs(err)
    val = df.iloc[va].copy()
    val["pred"] = pred_va
    val["err"]  = err
    val["abs_err"] = abs_err

    fig, ax = plt.subplots(figsize=(9.5, 4.5), dpi=170)
    ax.hist(err / 1000, bins=60, color=CSUB_BLUE, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=CSUB_GOLD, linewidth=2)
    ax.set_xlabel("residual = predicted minus actual ($k)")
    ax.set_ylabel("count")
    ax.set_title("MLR residuals on validation set")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig06_baseline_residuals.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cap_rows    = val[val["price"] >= 1_900_000].sort_values("abs_err", ascending=False).head(3).reset_index(drop=True)
    noncap_rows = val[val["price"] <  1_900_000].sort_values("abs_err", ascending=False).head(3).reset_index(drop=True)

    fig = plt.figure(figsize=(12, 7.6), dpi=170)
    # 4 rows: caption, photos, caption, photos
    gs = fig.add_gridspec(4, 3, height_ratios=[0.15, 1.0, 0.15, 1.0], hspace=0.25, wspace=0.08)

    def caption(row, text):
        ax = fig.add_subplot(gs[row, :])
        ax.axis("off")
        ax.text(0.0, 0.5, text, transform=ax.transAxes,
                fontsize=13, color=CSUB_BLUE, fontweight="bold", va="center", ha="left")

    def photo_row(grid_row, rows):
        for i, row in enumerate(rows.itertuples(index=False)):
            ax = fig.add_subplot(gs[grid_row, i])
            img = Image.open(os.path.join(PICS_DIR, f"{int(row.image_id)}.jpg")).convert("RGB")
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color(CSUB_GOLD); sp.set_linewidth(2)
            sign = "over" if row.err > 0 else "under"
            ax.set_title(f"{row.citi.split(',')[0]}, {row.bed}bd/{row.bath:g}ba/{int(row.sqft):,}sf",
                         fontsize=10, color=INK, loc="left", pad=4)
            ax.text(0.02, 0.06,
                    f"actual ${row.price/1000:,.0f}k\npred   ${row.pred/1000:,.0f}k\n{sign} by ${abs(row.err)/1000:,.0f}k",
                    transform=ax.transAxes, fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=CSUB_BLUE, edgecolor="none"))

    caption(0, "at the $1,995k dataset cap  —  38 listings share this exact price")
    photo_row(1, cap_rows)
    caption(2, "worst misses below the cap  —  real model failures")
    photo_row(3, noncap_rows)

    fig.savefig(os.path.join(FIGURES_DIR, "fig07_worst_misses.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top_cities = val["citi"].value_counts().head(10).index.tolist()
    rows = []
    for c in top_cities:
        sub = val[val["citi"] == c]
        rows.append({"city": c.split(",")[0], "n": len(sub), "mae": float(sub["abs_err"].mean())})
    city_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=170)
    y_pos = np.arange(len(city_df))[::-1]
    ax.barh(y_pos, city_df["mae"] / 1000, color=CSUB_BLUE)
    ax.set_yticks(y_pos); ax.set_yticklabels(city_df["city"])
    for yp, m, n in zip(y_pos, city_df["mae"] / 1000, city_df["n"]):
        ax.text(m + max(city_df["mae"] / 1000) * 0.01, yp, f"${m:.0f}k  (n={n})",
                va="center", color=INK, fontsize=10)
    ax.set_xlabel("validation MAE ($k)")
    ax.set_title("per-city MAE for the top 10 cities, MLR baseline")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig08_per_city_baseline.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    out_dir = os.path.join(os.path.dirname(FIGURES_DIR), "results")
    os.makedirs(out_dir, exist_ok=True)
    val.to_parquet(os.path.join(out_dir, "val_baseline_residuals.parquet"), index=False)
    print("error-analysis figures saved.")
    worst_all = val.sort_values("abs_err", ascending=False).head(6).reset_index(drop=True)
    print("\nworst 5 misses:")
    for r in worst_all.head(5).itertuples(index=False):
        print(f"  id={int(r.image_id):>5d}  {r.citi:25s}  actual=${r.price:>10,.0f}  pred=${r.pred:>10,.0f}  err=${r.err:>+12,.0f}")


if __name__ == "__main__":
    main()
