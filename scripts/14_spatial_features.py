import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import RAW_CSV, PICS_DIR, CLEAN_DIR

TARGET = 128
GRID   = 4          # 4 x 4 spatial pool
CELL   = TARGET // GRID
OUT    = os.path.join(CLEAN_DIR, "features_spatial.parquet")


def load_resized(image_id):
    p = os.path.join(PICS_DIR, f"{image_id}.jpg")
    with Image.open(p) as im:
        im = im.convert("RGB").resize((TARGET, TARGET), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def conv3(img, k):
    # generic 3x3 convolution via slicing on a single-channel image
    out = np.zeros_like(img)
    out[1:-1, 1:-1] = (
        k[0, 0] * img[:-2, :-2] + k[0, 1] * img[:-2, 1:-1] + k[0, 2] * img[:-2, 2:]
        + k[1, 0] * img[1:-1, :-2] + k[1, 1] * img[1:-1, 1:-1] + k[1, 2] * img[1:-1, 2:]
        + k[2, 0] * img[2:,  :-2] + k[2, 1] * img[2:,  1:-1] + k[2, 2] * img[2:,  2:]
    )
    return out


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T


def pool_4x4(resp):
    # mean-pool an arbitrary 2D map down to a 4x4 grid of cells
    h, w = resp.shape
    cy = h // GRID; cx = w // GRID
    out = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(GRID):
        for j in range(GRID):
            patch = resp[i * cy:(i + 1) * cy, j * cx:(j + 1) * cx]
            out[i, j] = float(patch.mean())
    return out


def features_for_image(arr):
    r = arr[..., 0]; g = arr[..., 1]; b = arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    rgb_sum = np.maximum(r + g + b, 1e-6)

    maps = {
        "sobel_x":  np.abs(conv3(gray, SOBEL_X)),
        "sobel_y":  np.abs(conv3(gray, SOBEL_Y)),
        "bright":   gray,
        "blue_dom": b / rgb_sum,
        "green_dom": g / rgb_sum,
        "red_dom":  r / rgb_sum,
    }

    out = {}
    for name, m in maps.items():
        cells = pool_4x4(m)
        for i in range(GRID):
            for j in range(GRID):
                out[f"{name}_{i}{j}"] = float(cells[i, j])
    return out


def main():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    n = len(df)
    rows = []
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            arr = load_resized(row.image_id)
            feats = features_for_image(arr)
        except Exception:
            feats = {}
        feats["image_id"] = int(row.image_id)
        rows.append(feats)
        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - (i + 1)) / rate
            print(f"{i+1}/{n}  rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)
    feat_df = pd.DataFrame(rows)
    feat_df.to_parquet(OUT, index=False)
    print(f"saved {OUT}  shape={feat_df.shape}")


if __name__ == "__main__":
    main()
