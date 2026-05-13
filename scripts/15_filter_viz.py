import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, PICS_DIR, FIGURES_DIR,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED,
)

TARGET = 128


def load_resized(image_id):
    p = os.path.join(PICS_DIR, f"{image_id}.jpg")
    with Image.open(p) as im:
        im = im.convert("RGB").resize((TARGET, TARGET), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def conv3(img, k):
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = (
        k[0, 0] * img[:-2, :-2] + k[0, 1] * img[:-2, 1:-1] + k[0, 2] * img[:-2, 2:]
        + k[1, 0] * img[1:-1, :-2] + k[1, 1] * img[1:-1, 1:-1] + k[1, 2] * img[1:-1, 2:]
        + k[2, 0] * img[2:,  :-2] + k[2, 1] * img[2:,  1:-1] + k[2, 2] * img[2:,  2:]
    )
    return out


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T


def pick_house():
    # find a clean exterior photo: medium price, common city
    df = load_features().reset_index(drop=True)
    sub = df[(df["citi"].str.contains("Lancaster", case=False, na=False))
             & (df["price"].between(300_000, 500_000))
             & (df["bed"] == 3)
             & (df["bath"].between(1.5, 2.5))]
    if len(sub) == 0:
        sub = df[df["price"].between(400_000, 600_000)]
    return int(sub.iloc[0]["image_id"])


def filter_maps(arr):
    r = arr[..., 0]; g = arr[..., 1]; b = arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    rgb_sum = np.maximum(r + g + b, 1e-6)
    return [
        ("sobel-x\nvertical edges",          np.abs(conv3(gray, SOBEL_X)), "Blues"),
        ("sobel-y\nhorizontal edges",        np.abs(conv3(gray, SOBEL_Y)), "Blues"),
        ("brightness\nlight vs dark",        gray,                          "gray"),
        ("blue dominance\nsky / pool",       b / rgb_sum,                   "Blues"),
        ("green dominance\nvegetation",      g / rgb_sum,                   "Greens"),
        ("red dominance\nroof / brick",      r / rgb_sum,                   "Reds"),
    ]


def main():
    iid = pick_house()
    arr = load_resized(iid)
    maps = filter_maps(arr)

    fig, axes = plt.subplots(1, 7, figsize=(15.5, 3.0), dpi=170)
    axes[0].imshow(arr.astype(np.uint8))
    axes[0].set_title("original", fontsize=11, color=INK, pad=6)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for sp in axes[0].spines.values(): sp.set_color(CSUB_BLUE); sp.set_linewidth(2)
    for ax, (name, m, cmap) in zip(axes[1:], maps):
        ax.imshow(m, cmap=cmap)
        ax.set_title(name, fontsize=10, color=INK, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(CSUB_BLUE); sp.set_linewidth(2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig19_filter_bank.png"),
                dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved fig19_filter_bank.png (sample house image_id={iid})")


if __name__ == "__main__":
    main()
