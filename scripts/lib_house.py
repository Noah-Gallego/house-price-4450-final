import os
import json
import numpy as np
import pandas as pd

RAW_CSV   = "/home/noah-gallego/house-data/raw/socal2.csv"
PICS_DIR  = "/home/noah-gallego/house-data/raw/socal2/socal_pics"
CLEAN_DIR = "/home/noah-gallego/house-data/clean"
FEATS_PARQUET = os.path.join(CLEAN_DIR, "features.parquet")
SPLITS_NPZ    = os.path.join(CLEAN_DIR, "splits.npz")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO, "figures")
RESULTS_DIR = os.path.join(REPO, "results")
ASSETS_DIR  = os.path.join(REPO, "slides_assets")

CSUB_BLUE = "#003594"
CSUB_GOLD = "#FDB913"
INK       = "#202535"
MUTED     = "#606878"
PANEL     = "#F5F7FB"

SPEC_COLS  = ["bed", "bath", "sqft", "city_target_mean", "city_lat", "city_lon"]
CITIES_GEO = os.path.join(CLEAN_DIR, "cities_geo.csv")
IMAGE_BASIC_COLS = (
    ["mean_r", "mean_g", "mean_b", "brightness", "contrast", "edge_density"]
    + [f"hist_r_{i}" for i in range(4)]
    + [f"hist_g_{i}" for i in range(4)]
    + [f"hist_b_{i}" for i in range(4)]
)
IMAGE_EXTRA_COLS = [
    "sky_brightness", "sky_blue_ratio", "green_fraction",
    "saturation_mean", "pixel_variance", "center_dark_ratio",
    "edge_top", "edge_bottom",
]
IMAGE_COLS = IMAGE_BASIC_COLS + IMAGE_EXTRA_COLS


def load_raw():
    return pd.read_csv(RAW_CSV)


def load_features():
    return pd.read_parquet(FEATS_PARQUET)


def make_split_indices(n, seed=42, p_train=0.65, p_val=0.15):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * p_train))
    n_val   = int(round(n * p_val))
    train = idx[:n_train]
    val   = idx[n_train:n_train + n_val]
    test  = idx[n_train + n_val:]
    return train, val, test


def save_splits(n=15474, seed=42):
    os.makedirs(CLEAN_DIR, exist_ok=True)
    train, val, test = make_split_indices(n, seed=seed)
    np.savez(SPLITS_NPZ, train=train, val=val, test=test)
    return train, val, test


def load_splits():
    z = np.load(SPLITS_NPZ)
    return z["train"], z["val"], z["test"]


def attach_city_target_mean(df, train_idx):
    train_df = df.iloc[train_idx]
    city_means = train_df.groupby("citi")["price"].mean()
    global_mean = train_df["price"].mean()
    out = df.copy()
    out["city_target_mean"] = out["citi"].map(city_means).fillna(global_mean).astype(float)

    if os.path.exists(CITIES_GEO):
        geo = pd.read_csv(CITIES_GEO)
        lat_map = dict(zip(geo["citi"], geo["lat"]))
        lon_map = dict(zip(geo["citi"], geo["lon"]))
        train_lat = train_df["citi"].map(lat_map)
        train_lon = train_df["citi"].map(lon_map)
        global_lat = float(train_lat.dropna().mean()) if train_lat.notna().any() else 34.0
        global_lon = float(train_lon.dropna().mean()) if train_lon.notna().any() else -118.0
        out["city_lat"] = out["citi"].map(lat_map).fillna(global_lat).astype(float)
        out["city_lon"] = out["citi"].map(lon_map).fillna(global_lon).astype(float)
    else:
        out["city_lat"] = 34.0
        out["city_lon"] = -118.0
    return out, city_means, global_mean


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "mae":         float(abs_err.mean()),
        "rmse":        float(np.sqrt((err ** 2).mean())),
        "sse":         float((err ** 2).sum()),
        "median_abs":  float(np.median(abs_err)),
        "max_abs":     float(abs_err.max()),
        "within_50k":  float((abs_err <= 50_000).mean()),
        "within_100k": float((abs_err <= 100_000).mean()),
        "within_200k": float((abs_err <= 200_000).mean()),
        "n":           int(len(y_true)),
    }


def save_metrics(payload, path=None):
    if path is None:
        path = os.path.join(RESULTS_DIR, "metrics.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(payload)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    return path


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK)
    ax.grid(True, axis="y", color="#E4E7EE", linewidth=0.8)
    ax.set_axisbelow(True)


def fmt_dollars(x, _=None):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}k"
    return f"${x:.0f}"
