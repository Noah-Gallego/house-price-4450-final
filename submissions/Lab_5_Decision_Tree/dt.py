import csv
from math import log2

def load(path, labeled=True):
    with open(path) as f:
        return [([float(x) for x in r[:4]], int(r[4]) if labeled else None)
                for r in csv.reader(f) if r]

def build(rows, max_depth=None, min_gain=0.0, depth=0):
    # entropy of current set
    counts = {}
    for _, c in rows:
        counts[c] = counts.get(c, 0) + 1
    n = len(rows)
    base = -sum((v/n) * log2(v/n) for v in counts.values())
    majority = max(counts, key=counts.get)

    if base == 0 or (max_depth is not None and depth >= max_depth):
        return ("leaf", majority)

    # find best split across all features/thresholds
    best = (0.0, None, None, None, None)
    for feat in range(4):
        vals = sorted(set(r[0][feat] for r in rows))
        for i in range(len(vals) - 1):
            thr = (vals[i] + vals[i+1]) / 2
            l = [r for r in rows if r[0][feat] <= thr]
            r_ = [r for r in rows if r[0][feat] > thr]
            if not l or not r_: continue
            def h(s):
                cc = {}
                for _, c in s: cc[c] = cc.get(c, 0) + 1
                return -sum((v/len(s)) * log2(v/len(s)) for v in cc.values())
            gain = base - (len(l)/n)*h(l) - (len(r_)/n)*h(r_)
            if gain > best[0]:
                best = (gain, feat, thr, l, r_)

    gain, feat, thr, l, r_ = best
    if feat is None or gain <= min_gain:
        return ("leaf", majority)
    return ("node", feat, thr,
            build(l, max_depth, min_gain, depth+1),
            build(r_, max_depth, min_gain, depth+1))

def predict(tree, feats):
    while tree[0] == "node":
        _, feat, thr, l, r = tree
        tree = l if feats[feat] <= thr else r
    return tree[1]

def show(tree, depth=0):
    pad = "  " * depth
    names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    if tree[0] == "leaf":
        print(f"{pad}class {tree[1]}")
    else:
        _, feat, thr, l, r = tree
        print(f"{pad}{names[feat]} <= {thr}")
        show(l, depth+1)
        print(f"{pad}{names[feat]} > {thr}")
        show(r, depth+1)

train = load("DT_irisTraining.csv")
test = load("DT_irisTesting.csv", labeled=False)

trees = [
    ("A (perfect)", build(train)),
    ("B (over-pruned)", build(train, max_depth=1)),
    ("C (pruned)", build(train, max_depth=3, min_gain=0.1)),
]

for name, t in trees:
    print(f"=== Tree {name} ===")
    show(t)
    acc = sum(1 for f, c in train if predict(t, f) == c) / len(train)
    print(f"training accuracy: {acc:.4f}")
    print(f"test predictions: {[predict(t, f) for f, _ in test]}")
    print()
