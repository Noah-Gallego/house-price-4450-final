import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import RAW_CSV, PICS_DIR, CLEAN_DIR

TARGET = 128
OUT = os.path.join(CLEAN_DIR, "features_v2.parquet")


def load_resized(image_id):
    p = os.path.join(PICS_DIR, f"{image_id}.jpg")
    with Image.open(p) as im:
        im = im.convert("RGB").resize((TARGET, TARGET), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def sobel_edge_density(gray):
    g = gray
    h, w = g.shape
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[1:-1, 1:-1] = (
        -g[:-2, :-2] + g[:-2, 2:]
        - 2 * g[1:-1, :-2] + 2 * g[1:-1, 2:]
        - g[2:, :-2]  + g[2:, 2:]
    )
    gy[1:-1, 1:-1] = (
        -g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:]
        + g[2:,  :-2] + 2 * g[2:,  1:-1] + g[2:,  2:]
    )
    return float(np.sqrt(gx * gx + gy * gy).mean())


def center_crop(arr, frac=0.6):
    h, w = arr.shape[:2]
    dy = int(h * (1 - frac) / 2)
    dx = int(w * (1 - frac) / 2)
    return arr[dy:h - dy, dx:w - dx]


def per_image_brightness_normalize(arr, target=128.0):
    g = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    m = max(float(g.mean()), 1e-6)
    out = arr * (target / m)
    return np.clip(out, 0, 255)


def hsv_from_rgb(arr):
    r = arr[..., 0] / 255.0; g = arr[..., 1] / 255.0; b = arr[..., 2] / 255.0
    mx = np.max(arr, axis=-1) / 255.0
    mn = np.min(arr, axis=-1) / 255.0
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    # hue, 0..1 (representing 0..360 deg)
    diff = np.maximum(mx - mn, 1e-6)
    h = np.zeros_like(v)
    mask_r = (mx == r) & (mx - mn > 1e-6)
    mask_g = (mx == g) & (mx - mn > 1e-6)
    mask_b = (mx == b) & (mx - mn > 1e-6)
    h = np.where(mask_r, ((g - b) / diff) % 6, h)
    h = np.where(mask_g, ((b - r) / diff) + 2, h)
    h = np.where(mask_b, ((r - g) / diff) + 4, h)
    h = (h / 6.0) % 1.0
    return h, s, v


def features_27(arr, prefix=""):
    r = arr[..., 0]; g = arr[..., 1]; b = arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    out = {
        f"{prefix}mean_r": float(r.mean()),
        f"{prefix}mean_g": float(g.mean()),
        f"{prefix}mean_b": float(b.mean()),
        f"{prefix}brightness": float(gray.mean()),
        f"{prefix}contrast":   float(gray.std()),
        f"{prefix}edge_density": sobel_edge_density(gray),
    }
    edges = np.linspace(0, 256, 5)
    for name, chan in (("r", r), ("g", g), ("b", b)):
        h, _ = np.histogram(chan, bins=edges)
        s = h.sum()
        h = h / s if s > 0 else h
        for i in range(4):
            out[f"{prefix}hist_{name}_{i}"] = float(h[i])

    h, w = gray.shape
    top    = slice(0, h // 4)
    bottom = slice(h * 3 // 4, h)
    cy0, cy1 = h * 3 // 10, h * 7 // 10
    cx0, cx1 = w * 3 // 10, w * 7 // 10

    out[f"{prefix}sky_brightness"] = float(gray[top].mean())
    sky_b = arr[top, ..., 2]; sky_r = arr[top, ..., 0]
    out[f"{prefix}sky_blue_ratio"] = float(sky_b.mean() / (sky_r.mean() + 1e-6))

    green_mask = (g > r + 8) & (g > b + 8)
    out[f"{prefix}green_fraction"] = float(green_mask.mean())

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)
    out[f"{prefix}saturation_mean"] = float(sat.mean())

    out[f"{prefix}pixel_variance"] = float(gray.var())

    center = gray[cy0:cy1, cx0:cx1]
    out[f"{prefix}center_dark_ratio"] = float((center < 80).mean())

    out[f"{prefix}edge_top"]    = sobel_edge_density(gray[top])
    out[f"{prefix}edge_bottom"] = sobel_edge_density(gray[bottom])
    return out


def hsv_features(arr, prefix=""):
    h, s, v = hsv_from_rgb(arr)
    return {
        f"{prefix}hue_mean":   float(h.mean()),
        f"{prefix}hue_std":    float(h.std()),
        f"{prefix}sat_mean":   float(s.mean()),
        f"{prefix}val_mean":   float(v.mean()),
    }


def all_variants_for_image(arr):
    out = {}
    # variant A: center crop, same 27 features
    cropped = center_crop(arr, frac=0.6)
    out.update(features_27(cropped, prefix="c_"))
    # variant B: per-image brightness normalize, same 27 features
    norm = per_image_brightness_normalize(arr, target=128.0)
    out.update(features_27(norm, prefix="n_"))
    # variant C: combined (center crop, then per-image normalize), + HSV
    combo = per_image_brightness_normalize(cropped, target=128.0)
    out.update(features_27(combo, prefix="x_"))
    out.update(hsv_features(combo, prefix="x_"))
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
            feats = all_variants_for_image(arr)
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
