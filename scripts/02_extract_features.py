import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import RAW_CSV, PICS_DIR, FEATS_PARQUET, CLEAN_DIR, IMAGE_COLS, IMAGE_BASIC_COLS, IMAGE_EXTRA_COLS

TARGET = 128


def load_resized(image_id):
    p = os.path.join(PICS_DIR, f"{image_id}.jpg")
    with Image.open(p) as im:
        im = im.convert("RGB").resize((TARGET, TARGET), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def sobel_edge_density(gray):
    g = gray.astype(np.float32)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    h, w = g.shape
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    # 3x3 conv via slicing
    gx[1:-1, 1:-1] = (
        kx[0, 0] * g[:-2, :-2] + kx[0, 1] * g[:-2, 1:-1] + kx[0, 2] * g[:-2, 2:]
        + kx[1, 0] * g[1:-1, :-2] + kx[1, 1] * g[1:-1, 1:-1] + kx[1, 2] * g[1:-1, 2:]
        + kx[2, 0] * g[2:, :-2] + kx[2, 1] * g[2:, 1:-1] + kx[2, 2] * g[2:, 2:]
    )
    gy[1:-1, 1:-1] = (
        ky[0, 0] * g[:-2, :-2] + ky[0, 1] * g[:-2, 1:-1] + ky[0, 2] * g[:-2, 2:]
        + ky[1, 0] * g[1:-1, :-2] + ky[1, 1] * g[1:-1, 1:-1] + ky[1, 2] * g[1:-1, 2:]
        + ky[2, 0] * g[2:, :-2] + ky[2, 1] * g[2:, 1:-1] + ky[2, 2] * g[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    return float(mag.mean())


def features_for_image(arr):
    r = arr[..., 0]; g = arr[..., 1]; b = arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    out = {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "brightness": float(gray.mean()),
        "contrast":   float(gray.std()),
        "edge_density": sobel_edge_density(gray),
    }
    edges = np.linspace(0, 256, 5)
    for name, chan in (("r", r), ("g", g), ("b", b)):
        h, _ = np.histogram(chan, bins=edges)
        h = h / h.sum() if h.sum() > 0 else h
        for i in range(4):
            out[f"hist_{name}_{i}"] = float(h[i])

    h, w = gray.shape
    top    = slice(0, h // 4)
    bottom = slice(h * 3 // 4, h)
    cy0, cy1 = h * 3 // 10, h * 7 // 10
    cx0, cx1 = w * 3 // 10, w * 7 // 10

    sky = gray[top]
    sky_b = arr[top, ..., 2]; sky_r = arr[top, ..., 0]
    out["sky_brightness"] = float(sky.mean())
    out["sky_blue_ratio"] = float(sky_b.mean() / (sky_r.mean() + 1e-6))

    green_mask = (g > r + 8) & (g > b + 8)
    out["green_fraction"] = float(green_mask.mean())

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    out["saturation_mean"] = float(sat.mean())

    out["pixel_variance"] = float(gray.var())

    center = gray[cy0:cy1, cx0:cx1]
    out["center_dark_ratio"] = float((center < 80).mean())

    out["edge_top"]    = sobel_edge_density(gray[top])
    out["edge_bottom"] = sobel_edge_density(gray[bottom])
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
        except Exception as e:
            feats = {c: float("nan") for c in IMAGE_COLS}
        feats["image_id"] = int(row.image_id)
        rows.append(feats)
        if (i + 1) % 500 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - (i + 1)) / rate
            print(f"{i+1}/{n}  rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)
    feat_df = pd.DataFrame(rows)
    out = df.merge(feat_df, on="image_id", how="left")
    out.to_parquet(FEATS_PARQUET, index=False)
    print(f"saved {FEATS_PARQUET}  shape={out.shape}")


if __name__ == "__main__":
    main()
