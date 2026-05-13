import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, FIGURES_DIR, CITIES_GEO,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)


def main():
    df = load_features().reset_index(drop=True)
    tr, va, te = load_splits()
    geo = pd.read_csv(CITIES_GEO).dropna(subset=["lat", "lon"])

    train_df = df.iloc[tr]
    city_means = train_df.groupby("citi")["price"].mean().to_dict()
    city_counts = train_df.groupby("citi").size().to_dict()

    geo["price"] = geo["citi"].map(city_means)
    geo["n"]     = geo["citi"].map(city_counts).fillna(0).astype(int)
    geo = geo.dropna(subset=["price"])

    cmap = LinearSegmentedColormap.from_list("csub_blue_gold", ["#003594", "#7AAEFF", "#FDB913"])

    fig, ax = plt.subplots(figsize=(11, 7.0), dpi=160)
    sc = ax.scatter(geo["lon"], geo["lat"],
                    s=np.sqrt(geo["n"]) * 5 + 8,
                    c=geo["price"] / 1000,
                    cmap=cmap, edgecolor=INK, linewidth=0.5, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("train-mean price ($k)", color=INK, fontsize=11)
    cbar.ax.tick_params(colors=INK)

    big = geo.nlargest(8, "n")
    for _, r in big.iterrows():
        ax.annotate(r["citi"].split(",")[0], (r["lon"], r["lat"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8.5, color=INK, alpha=0.9)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("415 SoCal cities, plotted at their geocoded coordinates",
                 fontsize=13, color=CSUB_BLUE, fontweight="bold", pad=12)
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig21_city_map.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}  ({len(geo)} cities plotted)")


if __name__ == "__main__":
    main()
