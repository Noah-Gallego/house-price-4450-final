import os
import sys
import numpy as np

# K-means
def kmeans(X, k, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    C = X[rng.choice(n, k, replace=False)].astype(float)
    labels = np.zeros(n, dtype=int)
    for _ in range(100):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        new_labels = D.argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                C[c] = X[m].mean(0)
    wcss = float(((X - C[labels]) ** 2).sum())
    return labels, wcss

# Best of N random restarts
def best_kmeans(X, k, restarts=20):
    best = None
    for s in range(restarts):
        labels, wcss = kmeans(X, k, seed=s)
        if best is None or wcss < best[1]:
            best = (labels, wcss)
    return best

# Iris
data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
iris = np.loadtxt(os.path.join(data_dir, 'Kmeans_irisTesting.csv'), delimiter=',')

print('=== Iris (k=3) ===')
labels, wcss = best_kmeans(iris, 3)
ans = ''.join(str(l + 1) for l in labels)
with open('YourAns.txt', 'w') as f:
    f.write(ans)
print(f'WCSS={wcss:.2f}, written to YourAns.txt')
print(ans)

print()
print('=== Iris: try different k ===')
for k in range(1, 8):
    _, w = best_kmeans(iris, k)
    print(f'  k={k}  WCSS={w:.2f}')

adj = np.loadtxt(os.path.join(data_dir, 'Kmeans_KarateMatrix.txt'))
print()
print('=== Karate (adjacency rows as features) ===')
for k in [2, 3, 4, 5]:
    labels, w = best_kmeans(adj, k)
    print(f'  k={k}  WCSS={w:.2f}')
    if k == 2:
        for c in [0, 1]:
            members = [i + 1 for i in range(len(labels)) if labels[i] == c]
            print(f'    group {c+1} ({len(members)}): {members}')
