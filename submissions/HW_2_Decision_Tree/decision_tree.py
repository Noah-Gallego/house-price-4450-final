import math
import json

data = [
    ("Sunny",    "Hot",  "High",   "False", "No"),
    ("Sunny",    "Hot",  "High",   "True",  "No"),
    ("Overcast", "Hot",  "High",   "False", "Yes"),
    ("Rainy",    "Mild", "High",   "False", "Yes"),
    ("Rainy",    "Cool", "Normal", "False", "Yes"),
    ("Rainy",    "Cool", "Normal", "True",  "No"),
    ("Overcast", "Cool", "Normal", "True",  "Yes"),
    ("Sunny",    "Mild", "High",   "False", "No"),
    ("Sunny",    "Cool", "Normal", "False", "Yes"),
    ("Rainy",    "Mild", "Normal", "False", "Yes"),
    ("Sunny",    "Mild", "Normal", "True",  "Yes"),
    ("Overcast", "Mild", "High",   "True",  "Yes"),
    ("Overcast", "Hot",  "Normal", "False", "Yes"),
    ("Rainy",    "Mild", "High",   "True",  "No"),
]

attrs = ["Outlook", "Temp", "Humidity", "Windy"]


def entropy(rows):
    labels = [r[-1] for r in rows]
    n = len(labels)
    return -sum((labels.count(c) / n) * math.log2(labels.count(c) / n) for c in set(labels))


def build(rows, available):
    labels = [r[-1] for r in rows]
    if len(set(labels)) == 1:
        return labels[0]
    if not available:
        return max(set(labels), key=labels.count)

    base = entropy(rows)
    best, best_gain = None, -1
    for a in available:
        idx = attrs.index(a)
        groups = {}
        for r in rows:
            groups.setdefault(r[idx], []).append(r)
        gain = base - sum((len(g) / len(rows)) * entropy(g) for g in groups.values())
        if gain > best_gain:
            best, best_gain, best_groups = a, gain, groups

    return {best: {v: build(g, [a for a in available if a != best]) for v, g in best_groups.items()}}


tree = build(data, attrs)
print(json.dumps(tree, indent=2))
