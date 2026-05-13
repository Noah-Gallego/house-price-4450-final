import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_house import (
    load_features, load_splits, IMAGE_COLS, FIGURES_DIR, RESULTS_DIR,
    CSUB_BLUE, CSUB_GOLD, INK, MUTED, style_axes,
)
from models import fit_scaler, apply_scaler

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlecolor": CSUB_BLUE,
})


def kmeans(X, k, max_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    init = rng.choice(n, size=k, replace=False)
    cent = X[init].copy()
    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        d2 = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2) if X.shape[1] < 32 else _pairwise_sq(X, cent)
        new_labels = np.argmin(d2, axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                cent[j] = X[mask].mean(axis=0)
    sse = float(((X - cent[labels]) ** 2).sum())
    return labels, cent, sse


def _pairwise_sq(A, B):
    a2 = (A * A).sum(axis=1)[:, None]
    b2 = (B * B).sum(axis=1)[None, :]
    return np.maximum(a2 + b2 - 2 * A @ B.T, 0)


def hierarchical(X, method="average"):
    n = len(X)
    D = _pairwise_sq(X, X) ** 0.5
    np.fill_diagonal(D, np.inf)
    cluster_size = np.ones(n, dtype=float)
    parent = list(range(n))
    active = list(range(n))
    Z = np.zeros((n - 1, 4))
    next_id = n
    label_map = {i: i for i in range(n)}
    for step in range(n - 1):
        flat = np.argmin(D)
        i, j = divmod(int(flat), D.shape[0])
        if i > j:
            i, j = j, i
        d_ij = float(D[i, j])
        ci = label_map[active[i]]
        cj = label_map[active[j]]
        si = cluster_size[i]; sj = cluster_size[j]
        Z[step] = [ci, cj, d_ij, si + sj]
        if method == "average":
            new_row = (D[i] * si + D[j] * sj) / (si + sj)
        elif method == "single":
            new_row = np.minimum(D[i], D[j])
        else:
            new_row = np.maximum(D[i], D[j])
        new_row[i] = np.inf; new_row[j] = np.inf
        D[i] = new_row; D[:, i] = new_row
        D[j] = np.inf; D[:, j] = np.inf
        cluster_size[i] = si + sj
        label_map[active[i]] = next_id
        next_id += 1
        active.pop(j)
        cluster_size = np.delete(cluster_size, j) if False else cluster_size  # keep array size; mark inf instead
    return Z


def _build_linkage(D0, labels, method="average"):
    # full agglomerative on a small precomputed distance matrix
    n = D0.shape[0]
    D = D0.copy().astype(float)
    np.fill_diagonal(D, np.inf)
    sizes = np.ones(n, dtype=float)
    ids = np.arange(n)
    Z = []
    for step in range(n - 1):
        flat = int(np.argmin(D))
        i, j = divmod(flat, D.shape[0])
        if i > j: i, j = j, i
        d_ij = float(D[i, j])
        Z.append([ids[i], ids[j], d_ij, sizes[i] + sizes[j]])
        si = sizes[i]; sj = sizes[j]
        if method == "average":
            new_row = (D[i] * si + D[j] * sj) / (si + sj)
        elif method == "single":
            new_row = np.minimum(D[i], D[j])
        else:
            new_row = np.maximum(D[i], D[j])
        new_row[i] = np.inf; new_row[j] = np.inf
        D[i, :] = new_row; D[:, i] = new_row
        D = np.delete(D, j, axis=0); D = np.delete(D, j, axis=1)
        sizes[i] = si + sj
        sizes = np.delete(sizes, j)
        ids[i] = n + step
        ids = np.delete(ids, j)
    return np.asarray(Z, dtype=float)


def _draw_dendrogram(Z, names, ax):
    n = len(names)
    leaf_x = {i: float(i) for i in range(n)}
    leaf_y = {i: 0.0 for i in range(n)}
    for k, row in enumerate(Z):
        a, b, h, _ = row
        a = int(a); b = int(b)
        xa, ya = leaf_x[a], leaf_y[a]
        xb, yb = leaf_x[b], leaf_y[b]
        ax.plot([xa, xa, xb, xb], [ya, h, h, yb], color=CSUB_BLUE, linewidth=1.4)
        nid = n + k
        leaf_x[nid] = 0.5 * (xa + xb)
        leaf_y[nid] = h
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=9, color=INK)
    ax.set_ylabel("linkage distance")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main():
    df = load_features().reset_index(drop=True)
    tr, va, te = load_splits()
    y = df["price"].to_numpy(dtype=float)
    img = df[IMAGE_COLS].to_numpy(dtype=float)
    mu, sd = fit_scaler(img[tr])
    X = apply_scaler(img, mu, sd)

    k = 4
    labels, cent, sse = kmeans(X[tr], k=k, max_iter=100, seed=42)
    train_prices = y[tr]
    grp_means = [float(train_prices[labels == j].mean()) for j in range(k)]
    grp_med   = [float(np.median(train_prices[labels == j])) for j in range(k)]
    grp_n     = [int((labels == j).sum()) for j in range(k)]
    order = np.argsort(grp_means)

    fig, ax = plt.subplots(figsize=(10, 5.0), dpi=170)
    data = [train_prices[labels == j] / 1000 for j in order]
    box_labels = [f"cluster {i+1}\nn={grp_n[j]}" for i, j in enumerate(order)]
    bp = ax.boxplot(data, labels=box_labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, linewidth=1.4),
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.35,
                                    markeredgecolor=MUTED, markerfacecolor=MUTED),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    boxprops=dict(facecolor="#B7C5E2", edgecolor=CSUB_BLUE, linewidth=1.2))
    for j_idx, j in enumerate(order):
        ax.scatter(j_idx + 1, grp_means[j] / 1000, marker="D", color=CSUB_GOLD,
                   s=42, zorder=5, edgecolor=INK, linewidth=0.8)
    ax.set_ylabel("price ($k)")
    ax.set_title(f"K-means on image features only (k={k}), training-price by cluster")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig14_kmeans_image.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top_cities = df["citi"].value_counts().head(20).index.tolist()
    sub = df[df["citi"].isin(top_cities)].copy()
    sub_tr = sub.loc[sub.index.isin(tr)]
    city_summary = sub_tr.groupby("citi").agg(
        mean_price=("price", "mean"),
        **{c: (c, "mean") for c in IMAGE_COLS}
    ).loc[top_cities]
    feats = np.hstack([
        (city_summary["mean_price"].to_numpy().reshape(-1, 1) - city_summary["mean_price"].mean())
        / city_summary["mean_price"].std(),
        ((city_summary[IMAGE_COLS].to_numpy() - city_summary[IMAGE_COLS].to_numpy().mean(axis=0))
         / (city_summary[IMAGE_COLS].to_numpy().std(axis=0) + 1e-9)),
    ])
    D = _pairwise_sq(feats, feats) ** 0.5
    Z = _build_linkage(D, [c.split(",")[0] for c in top_cities], method="average")

    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=170)
    _draw_dendrogram(Z, [c.split(",")[0] for c in top_cities], ax)
    ax.set_title("hierarchical clustering of top 20 cities, train price + image features")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig15_hierarchical_cities.png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "kmeans_k": k,
        "kmeans_sse_on_scaled_image_train": sse,
        "cluster_mean_prices_ordered_low_to_high": [grp_means[j] for j in order],
        "cluster_sizes_ordered": [grp_n[j] for j in order],
        "hierarchical_cities": [c.split(",")[0] for c in top_cities],
    }
    with open(os.path.join(RESULTS_DIR, "clustering_summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print("kmeans cluster means (low to high):", [round(grp_means[j], 0) for j in order])
    print("kmeans cluster sizes:               ", [grp_n[j] for j in order])
    print("saved kmeans + hierarchical figures.")


if __name__ == "__main__":
    main()
