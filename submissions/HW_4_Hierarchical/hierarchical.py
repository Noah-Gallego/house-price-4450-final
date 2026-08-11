# Imports
import os
import sys
import numpy as np

# Data loader
def load(path):
    pts = []
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                pts.append((float(parts[0]), float(parts[1])))
                labels.append(int(float(parts[2])))
            elif len(parts) == 2:
                pts.append((float(parts[0]), float(parts[1])))
                labels.append(0)
    return np.array(pts, dtype=float), np.array(labels, dtype=int)

# Run all merges down to a single cluster
def merge_history(points, method='min'):
    n = len(points)
    diff = points[:, None, :] - points[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(D, np.inf)

    sizes = np.ones(n, dtype=float)
    centroids = points.copy()
    parent = np.arange(n)
    active = np.ones(n, dtype=bool)

    distances = []
    snapshots = [parent.copy()]

    for _ in range(n - 1):
        flat = int(np.argmin(D))
        i, j = flat // n, flat % n
        if i > j:
            i, j = j, i
        distances.append(float(D[i, j]))

        ni, nj = sizes[i], sizes[j]
        if method == 'min':
            new = np.minimum(D[i], D[j])
        elif method == 'max':
            new = np.maximum(D[i], D[j])
        elif method == 'avg':
            new = (ni * D[i] + nj * D[j]) / (ni + nj)
        elif method == 'centroid':
            centroids[i] = (ni * centroids[i] + nj * centroids[j]) / (ni + nj)
            new = np.sqrt(np.sum((centroids - centroids[i]) ** 2, axis=1))
        else:
            raise ValueError(method)

        active[j] = False
        new[~active] = np.inf
        new[i] = np.inf

        D[i] = new
        D[:, i] = new
        D[j, :] = np.inf
        D[:, j] = np.inf
        sizes[i] = ni + nj
        parent[parent == j] = i
        snapshots.append(parent.copy())

    return distances, snapshots

# Within-cluster sum of squared distances to centroid
def wcss(points, parent):
    total = 0.0
    for c in set(parent.tolist()):
        mask = parent == c
        pts = points[mask]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        total += float(np.sum((pts - center) ** 2))
    return total

# Elbow method
def elbow_k(points, snapshots, max_k=10):
    n = len(points)
    ks = list(range(1, min(max_k, n - 1) + 1))
    ys = [wcss(points, snapshots[n - k]) for k in ks]
    x = np.array(ks, dtype=float)
    y = np.array(ys, dtype=float)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line = p2 - p1
    line_n = line / np.linalg.norm(line)
    rel = np.column_stack([x, y]) - p1
    proj = rel @ line_n
    perp = rel - np.outer(proj, line_n)
    dist = np.linalg.norm(perp, axis=1)
    return ks[int(np.argmax(dist))]

# Convert a parent snapshot into 0-indexed cluster labels
def labels_from(parent):
    roots = sorted(set(parent.tolist()))
    label_map = {r: idx for idx, r in enumerate(roots)}
    return np.array([label_map[p] for p in parent.tolist()], dtype=int)

# Auto-k clustering using the elbow method
def hierarchical(points, method='min', max_k=10):
    _, snapshots = merge_history(points, method)
    k = elbow_k(points, snapshots, max_k=max_k)
    n = len(points)
    return labels_from(snapshots[n - k]), k

# Evaluation
def purity(pred, truth):
    n = len(pred)
    total = 0
    for c in set(pred.tolist()):
        mask = pred == c
        if not mask.any():
            continue
        vals, counts = np.unique(truth[mask], return_counts=True)
        total += counts.max()
    return total / n

# Run benchmark across all data files
if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    files = sorted(f for f in os.listdir(data_dir) if f.startswith('Hierarchical_TestCase') and f.endswith('.txt'))
    methods = ['min', 'max', 'avg', 'centroid']

    header = f'{"file":<30}{"n":>6}  ' + '  '.join(f'{m:>14}' for m in methods)
    print(header)
    print(' ' * 38 + '  '.join(f'{"k  purity":>14}' for _ in methods))
    for fname in files:
        path = os.path.join(data_dir, fname)
        pts, truth = load(path)
        cells = []
        for m in methods:
            pred, k = hierarchical(pts, m)
            cells.append(f'{k:>3} {purity(pred, truth):>9.4f}')
        short = fname.replace('Hierarchical_TestCaseClustering', '').replace('.txt', '')
        print(f'{short:<30}{len(pts):>6}  ' + '  '.join(f'{c:>14}' for c in cells))
