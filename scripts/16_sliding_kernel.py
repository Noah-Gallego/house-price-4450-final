import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, PICS_DIR, FIGURES_DIR, ASSETS_DIR,
    CSUB_BLUE, CSUB_GOLD, INK,
)

TARGET = 64                 # smaller image for tractable GIF
STRIDE = 4                  # kernel hops by this many pixels
KERNEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)


def load_resized(image_id, size=TARGET):
    p = os.path.join(PICS_DIR, f"{image_id}.jpg")
    with Image.open(p) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def pick_house():
    df = load_features().reset_index(drop=True)
    sub = df[(df["citi"].str.contains("Lancaster", case=False, na=False))
             & (df["price"].between(300_000, 500_000))
             & (df["bed"] == 3)
             & (df["bath"].between(1.5, 2.5))]
    if len(sub) == 0:
        sub = df[df["price"].between(400_000, 600_000)]
    return int(sub.iloc[0]["image_id"])


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    iid = pick_house()
    arr = load_resized(iid)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    h, w = gray.shape

    # walk the kernel across every (STRIDE)-th valid pixel position
    positions = [(i, j)
                 for i in range(1, h - 1, STRIDE)
                 for j in range(1, w - 1, STRIDE)]

    output_so_far = np.zeros_like(gray)

    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=140)
    ax_in.imshow(arr.astype(np.uint8))
    ax_in.set_title("photo, sobel-y kernel sliding", color=INK, fontsize=12, pad=8)
    ax_in.set_xticks([]); ax_in.set_yticks([])
    for sp in ax_in.spines.values(): sp.set_color(CSUB_BLUE); sp.set_linewidth(2)

    out_im = ax_out.imshow(output_so_far, cmap="Blues", vmin=0, vmax=500)
    ax_out.set_title("filter response, building up", color=INK, fontsize=12, pad=8)
    ax_out.set_xticks([]); ax_out.set_yticks([])
    for sp in ax_out.spines.values(): sp.set_color(CSUB_BLUE); sp.set_linewidth(2)

    rect = Rectangle((0, 0), 3, 3, fill=False, edgecolor=CSUB_GOLD, linewidth=2.0)
    ax_in.add_patch(rect)

    def update(frame_idx):
        i, j = positions[frame_idx]
        patch = gray[i - 1:i + 2, j - 1:j + 2]
        output_so_far[i, j] = abs(float((patch * KERNEL).sum()))
        rect.set_xy((j - 1.5, i - 1.5))
        out_im.set_data(output_so_far)
        return rect, out_im

    anim = FuncAnimation(fig, update, frames=len(positions), interval=40, blit=False)
    out_path = os.path.join(ASSETS_DIR, "sobel_sliding.gif")
    anim.save(out_path, writer=PillowWriter(fps=18))
    plt.close(fig)
    print(f"saved {out_path}  frames={len(positions)}")


if __name__ == "__main__":
    main()
