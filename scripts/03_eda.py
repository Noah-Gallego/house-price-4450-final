import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_raw, PICS_DIR, FIGURES_DIR, ASSETS_DIR,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes, fmt_dollars,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlecolor": CSUB_BLUE,
    "axes.labelcolor": INK,
})


def fig_price_hist(df, out):
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    ax.hist(df["price"] / 1000, bins=60, color=CSUB_BLUE, edgecolor="white", linewidth=0.5)
    mean_k = df["price"].mean() / 1000
    median_k = df["price"].median() / 1000
    ax.axvline(median_k, color=CSUB_GOLD, linewidth=2.2, label=f"median ${median_k:.0f}k")
    ax.axvline(mean_k, color=INK, linewidth=1.6, linestyle="--", label=f"mean ${mean_k:.0f}k")
    ax.set_xlabel("price ($k)")
    ax.set_ylabel("count")
    ax.set_title("price distribution, 15,474 listings")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_sample_houses(df, out, n=6, seed=7):
    rng = np.random.default_rng(seed)
    sample = df.sample(n=n, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), dpi=160)
    for ax, row in zip(axes.ravel(), sample.itertuples(index=False)):
        img = Image.open(os.path.join(PICS_DIR, f"{row.image_id}.jpg")).convert("RGB")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(CSUB_BLUE); spine.set_linewidth(1.5)
        ax.set_title(f"{row.citi.split(',')[0]}\n{row.bed}bd / {row.bath:g}ba / {row.sqft:,}sf",
                     fontsize=10, color=INK, loc="left", pad=4)
        price_k = row.price / 1000
        label = f"${price_k:.0f}k" if price_k < 1000 else f"${price_k/1000:.2f}M"
        ax.text(0.98, 0.05, label, transform=ax.transAxes, fontsize=12,
                color="white", weight="bold", ha="right",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=CSUB_BLUE, edgecolor="none"))
    fig.suptitle("six houses, drawn at random", color=CSUB_BLUE, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_top_cities(df, out, top=15):
    counts = df["citi"].value_counts().head(top)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts.values, color=CSUB_BLUE)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace(", CA", "") for c in counts.index])
    for yi, v in zip(y, counts.values):
        ax.text(v + max(counts.values) * 0.01, yi, str(v), va="center", color=INK, fontsize=10)
    ax.set_xlabel("listings")
    ax.set_title(f"top {top} cities by listing count")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_sqft_vs_price(df, out):
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=160)
    sub = df.sample(n=min(5000, len(df)), random_state=1)
    ax.scatter(sub["sqft"], sub["price"] / 1000, s=8, alpha=0.35, color=CSUB_BLUE, edgecolors="none")
    ax.set_xlabel("sqft")
    ax.set_ylabel("price ($k)")
    ax.set_title("price vs sqft, 5k sampled listings")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.0f}k"))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    df = load_raw()
    fig_price_hist(df, os.path.join(FIGURES_DIR, "fig01_price_hist.png"))
    fig_sample_houses(df, os.path.join(FIGURES_DIR, "fig02_sample_houses.png"))
    fig_top_cities(df, os.path.join(FIGURES_DIR, "fig03_top_cities.png"))
    fig_sqft_vs_price(df, os.path.join(FIGURES_DIR, "fig04_sqft_vs_price.png"))
    print("EDA figures saved.")


if __name__ == "__main__":
    main()
