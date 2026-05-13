import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, PICS_DIR, IMAGE_COLS, FIGURES_DIR,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlecolor": CSUB_BLUE,
})


def fig_bad_photos():
    # use the same image_ids from the worst-misses figure
    picks = [
        (18,    "date watermark",     "Tehachapi"),
        (13532, "harbor, not house",  "Morro Bay"),
        (5288,  "tree blocks house",  "San Jacinto"),
        (14757, "carport, not house", "Oxnard"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), dpi=170)
    for ax, (iid, cap, city) in zip(axes, picks):
        p = os.path.join(PICS_DIR, f"{iid}.jpg")
        img = np.asarray(Image.open(p).convert("RGB"))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{city}: {cap}", fontsize=11, color=INK, loc="left", pad=6)
        for spine in ax.spines.values():
            spine.set_color(CSUB_BLUE); spine.set_linewidth(2.0)
    fig.subplots_adjust(wspace=0.15, left=0.02, right=0.98, top=0.85, bottom=0.05)
    fig.savefig(os.path.join(FIGURES_DIR, "fig16_bad_photos.png"), dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_feature_scales():
    df = load_features().reset_index(drop=True)
    tr, va, te = load_splits()
    img = df[IMAGE_COLS].to_numpy(float)
    stats = []
    for i, c in enumerate(IMAGE_COLS):
        v = img[tr, i]
        stats.append((c, float(v.std())))
    stats.sort(key=lambda x: x[1], reverse=True)
    names = [s[0] for s in stats]
    stds  = [s[1] for s in stats]

    fig, ax = plt.subplots(figsize=(11.5, 5.5), dpi=170)
    y = np.arange(len(names))[::-1]
    bars = ax.barh(y, stds, color=CSUB_BLUE, edgecolor=CSUB_BLUE)
    # highlight biggest offenders in gold
    big = [i for i, s in enumerate(stds) if s > 100]
    for i in big:
        bars[i].set_color(CSUB_GOLD)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("standard deviation across training photos (log scale)")
    ax.set_title("raw image features sit on wildly different scales")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig17_feature_scales.png"), dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_cap_cluster():
    df = load_features().reset_index(drop=True)
    sub = df[df["price"] >= 1_500_000]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=170)
    bins = np.arange(1_500_000, 2_005_000, 10_000)
    ax.hist(sub["price"], bins=bins, color=CSUB_BLUE, edgecolor=CSUB_BLUE)
    cap_n = int((df["price"] == 1_995_000).sum())
    ax.axvline(1_995_000, color=CSUB_GOLD, linewidth=2.4, linestyle="--",
               label=f"$1,995,000  ({cap_n} listings stacked here)")
    ax.set_xlabel("price")
    ax.set_ylabel("count")
    ax.set_title("dataset cap at $1,995k creates a synthetic cluster of houses")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
    ax.legend(frameon=False, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig18_cap_cluster.png"), dpi=170,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    fig_bad_photos()
    fig_feature_scales()
    fig_cap_cluster()
    print("saved fig16, fig17, fig18")


if __name__ == "__main__":
    main()
