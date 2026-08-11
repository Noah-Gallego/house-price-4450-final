import csv
import math
import random
import time
from collections import Counter

random.seed(42)


def load(path, labeled=True):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    if labeled:
        return [([float(r[0]), float(r[1])], r[2]) for r in rows]
    return [([float(r[0]), float(r[1])], None) for r in rows]


def fit_normalize(data):
    cols = list(zip(*[x for x, _ in data]))
    return [min(c) for c in cols], [max(c) for c in cols]


def apply_normalize(data, mins, maxs):
    return [([(x[i] - mins[i]) / (maxs[i] - mins[i]) for i in range(len(x))], y)
            for x, y in data]


def distance(a, b, name):
    if name == "Euclidean":
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    elif name == "Manhattan":
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    elif name == "Chebyshev":
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    elif name == "Minkowski":
        return (abs(a[0] - b[0]) ** 3 + abs(a[1] - b[1]) ** 3) ** (1 / 3)


DISTANCE_NAMES = ["Euclidean", "Manhattan", "Chebyshev", "Minkowski"]
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]


def knn_predict(known_xs, known_ys, query, k, dist_name):
    dists = [(distance(known_xs[i], query, dist_name), known_ys[i]) for i in range(len(known_xs))]
    dists.sort(key=lambda p: p[0])
    neighbors = [label for _, label in dists[:k]]
    counts = Counter(neighbors)
    return sorted(counts.items(), key=lambda t: (-t[1], t[0]))[0][0]


def grid_search(known_norm, n_trials, split_ratio=0.75):
    xs = [x for x, _ in known_norm]
    ys = [y for _, y in known_norm]
    n = len(xs)
    n_train = int(round(n * split_ratio))

    totals = {(dname, k): 0.0 for dname in DISTANCE_NAMES for k in K_VALUES}

    for t in range(n_trials):
        idx = list(range(n))
        random.shuffle(idx)
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:]

        for dname in DISTANCE_NAMES:
            for k in K_VALUES:
                correct = 0
                for vi in valid_idx:
                    pred = knn_predict(
                        [xs[ti] for ti in train_idx],
                        [ys[ti] for ti in train_idx],
                        xs[vi], k, dname,
                    )
                    if pred == ys[vi]:
                        correct += 1
                totals[(dname, k)] += correct / len(valid_idx)

    return {key: totals[key] / n_trials for key in totals}


def run(tag, train_path, test_path, ans_path, n_trials=10000):
    print(f"\n=== {tag} ===")
    known = load(train_path, labeled=True)
    unknown = load(test_path, labeled=False)
    mins, maxs = fit_normalize(known)
    known_n = apply_normalize(known, mins, maxs)
    unknown_n = apply_normalize(unknown, mins, maxs)

    t0 = time.time()
    avg = grid_search(known_n, n_trials=n_trials)
    elapsed = time.time() - t0
    print(f"  grid search ({n_trials} trials) took {elapsed:.1f}s")

    best_key = max(avg, key=avg.get)
    print(f"  best: distance={best_key[0]}  k={best_key[1]}  validation accuracy={avg[best_key]:.4f}")

    top5 = sorted(avg.items(), key=lambda kv: -kv[1])[:5]
    print("  top 5:")
    for (dn, k), a in top5:
        print(f"    {dn:12s} k={k:2d}  avg acc={a:.4f}")

    train_xs = [x for x, _ in known_n]
    train_ys = [y for _, y in known_n]
    preds = [knn_predict(train_xs, train_ys, x, best_key[1], best_key[0]) for x, _ in unknown_n]
    with open(ans_path, "w") as f:
        f.write("".join(preds) + "\n")
    print(f"  wrote {ans_path} ({len(preds)} predictions)")


if __name__ == "__main__":
    run("Data 1  (in / lb)", "KNN_armyTraining1.csv", "KNN_armyTesting1.csv", "YourAnsForData1.txt")
    run("Data 2  (cm / kg)", "KNN_armyTraining2.csv", "KNN_armyTesting2.csv", "YourAnsForData2.txt")
