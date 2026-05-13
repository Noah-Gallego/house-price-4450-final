import numpy as np


def fit_linear(X, y, ridge=1e-6):
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    A = Xb.T @ Xb
    A.flat[::A.shape[0] + 1] += ridge
    w = np.linalg.solve(A, Xb.T @ y)
    return w


def predict_linear(w, X):
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    return Xb @ w


def fit_scaler(X):
    # legacy z-score scaler, kept for older scripts
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def apply_scaler(X, mu, sd):
    return (X - mu) / sd


def fit_minmax(X):
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return lo, rng


def apply_minmax(X, lo, rng):
    return (X - lo) / rng


def knn_predict(X_train, y_train, X_query, k, metric="euclidean", p=3):
    if metric == "euclidean":
        tn = (X_train * X_train).sum(axis=1)
        qn = (X_query * X_query).sum(axis=1)
        cross = X_query @ X_train.T
        d = qn[:, None] + tn[None, :] - 2.0 * cross
        np.maximum(d, 0, out=d)
    else:
        # diff: (n_query, n_train, n_features). build in chunks to control memory
        n_q = X_query.shape[0]
        n_t = X_train.shape[0]
        d = np.empty((n_q, n_t), dtype=np.float64)
        CHUNK = max(1, 20_000_000 // max(n_t * X_train.shape[1], 1))
        for s in range(0, n_q, CHUNK):
            e = min(n_q, s + CHUNK)
            diff = np.abs(X_query[s:e, None, :] - X_train[None, :, :])
            if metric == "manhattan":
                d[s:e] = diff.sum(axis=2)
            elif metric == "chebyshev":
                d[s:e] = diff.max(axis=2)
            elif metric == "minkowski":
                d[s:e] = np.power((diff ** p).sum(axis=2), 1.0 / p)
            else:
                raise ValueError(f"unknown metric: {metric}")
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    return y_train[idx].mean(axis=1)


def _best_split(X_col, y, sse_parent, min_leaf):
    n = len(y)
    if n < 2 * min_leaf:
        return 0.0, None
    order = np.argsort(X_col, kind="stable")
    xs = X_col[order]; ys = y[order]
    csum = np.cumsum(ys)
    csum2 = np.cumsum(ys * ys)
    total_sum = csum[-1]; total_sum2 = csum2[-1]
    lo = min_leaf - 1
    hi = n - min_leaf
    if hi <= lo:
        return 0.0, None
    i = np.arange(lo, hi)
    n_l = (i + 1).astype(float)
    n_r = (n - n_l)
    s_l = csum[i]; s2_l = csum2[i]
    s_r = total_sum - s_l; s2_r = total_sum2 - s2_l
    sse_l = s2_l - (s_l * s_l) / n_l
    sse_r = s2_r - (s_r * s_r) / n_r
    gain = sse_parent - (sse_l + sse_r)
    same = xs[i] == xs[i + 1]
    gain = np.where(same, -np.inf, gain)
    j = int(np.argmax(gain))
    g = float(gain[j])
    if g <= 0 or not np.isfinite(g):
        return 0.0, None
    pos = i[j]
    thr = 0.5 * (xs[pos] + xs[pos + 1])
    return g, float(thr)


def fit_tree(X, y, max_depth=10, min_leaf=5):
    n, d = X.shape
    nodes = []
    stack = [(np.arange(n), 0, -1, None)]
    while stack:
        idx, depth, parent, side = stack.pop()
        ys = y[idx]
        leaf_val = float(ys.mean())
        sse_parent = float(((ys - leaf_val) ** 2).sum())
        node = {"is_leaf": True, "value": leaf_val, "feat": -1, "thr": 0.0, "left": -1, "right": -1}
        my_idx = len(nodes)
        nodes.append(node)
        if parent >= 0:
            if side == "L": nodes[parent]["left"] = my_idx
            else:           nodes[parent]["right"] = my_idx
        if depth >= max_depth or len(idx) < 2 * min_leaf or sse_parent < 1e-9:
            continue
        best_gain = 0.0
        best_feat = -1
        best_thr  = None
        for f in range(d):
            gain, thr = _best_split(X[idx, f], ys, sse_parent, min_leaf)
            if gain > best_gain:
                best_gain = gain; best_feat = f; best_thr = thr
        if best_feat < 0 or best_thr is None:
            continue
        left_mask  = X[idx, best_feat] <= best_thr
        right_mask = ~left_mask
        if left_mask.sum() < min_leaf or right_mask.sum() < min_leaf:
            continue
        node["is_leaf"] = False
        node["feat"] = best_feat; node["thr"] = float(best_thr)
        stack.append((idx[right_mask], depth + 1, my_idx, "R"))
        stack.append((idx[left_mask],  depth + 1, my_idx, "L"))
    return nodes


def predict_tree(nodes, X):
    n = X.shape[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        node = 0
        while not nodes[node]["is_leaf"]:
            if X[i, nodes[node]["feat"]] <= nodes[node]["thr"]:
                node = nodes[node]["left"]
            else:
                node = nodes[node]["right"]
        out[i] = nodes[node]["value"]
    return out
