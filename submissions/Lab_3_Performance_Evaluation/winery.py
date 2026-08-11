#%%
import matplotlib.pyplot as plt

# Open and parse file
with open('PerforEva_grape_data.txt') as file:
    lines = file.readlines()

diameters = [float(x) for x in lines[0].strip().split(',')]
labels = [int(x) for x in lines[1].strip().split(',')]

print(len(diameters), len(labels))
# %%
blueberry = [(i, d) for i, (d, l) in enumerate(zip(diameters, labels)) if l == 0]
merlot = [(i, d) for i, (d, l) in enumerate(zip(diameters, labels)) if l == 1]

plt.scatter([b[0] for b in blueberry], [b[1] for b in blueberry], color='blue', label='Blueberry', alpha=0.5, s=10)
plt.scatter([m[0] for m in merlot], [m[1] for m in merlot], color='red', label='Merlot', alpha=0.5, s=10)
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Diameter')
plt.show()
# %%

#%%
def MCC(TP, FP, TN, FN):
    n = (TP * TN - FP * FN)
    d = ( ( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))** 0.5) 
    return n / d

def metrics(TP, FP, TN, FN, mode='all'):
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if mode == 'accuracy': return accuracy
    elif mode == 'precision': return precision
    elif mode == 'recall': return recall
    elif mode == 'f1': return f1
    else: return accuracy, precision, recall, f1

def predict(diameters, threshold):
    preds = []

    # Generate Predictions
    for example in diameters:
        preds.append(1 if example > threshold else 0)

    return preds 

def evaluate(predictions, labels):
    TP = FP = TN = FN = 0

    for pred, label in zip(predictions, labels):
        if pred == 1 and label == 1:
            TP += 1
        elif pred == 1 and label == 0:
            FP += 1
        elif pred == 0 and label == 1:
            FN += 1
        elif pred == 0 and label == 0:
            TN += 1

    return TP, FP, TN, FN

def evaluate_on_thresholds(thresholds):
    results = []

    for t in thresholds:
        preds = predict(diameters, t)
        TP, FP, TN, FN = evaluate(preds, labels)
        acc, prec, rec, f1 = metrics(TP, FP, TN, FN)
        mcc = MCC(TP, FP, TN, FN)
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        tpr = rec

        results.append({
            't': t, 'acc': acc, 'prec': prec, 'rec': rec,
            'f1': f1, 'mcc': mcc, 'fpr': fpr, 'tpr': tpr
        })

    return results

# %%

# Obtain midpoints of each diameter
thresholds = [(a + b) / 2 for a, b in zip(sorted(set(diameters)), sorted(set(diameters))[1:])]
results = list(evaluate_on_thresholds(thresholds))

# Plot Each Metric
results = evaluate_on_thresholds(thresholds)
ts = [r['t'] for r in results]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

plots = [
    ('acc', 'Accuracy'),
    ('prec', 'Precision'),
    ('rec', 'Recall'),
    ('f1', 'F1 Score'),
    ('mcc', 'MCC'),
    ('tpr', 'TPR'),
]

for ax, (key, title) in zip(axes.flatten(), plots):
    vals = [r[key] for r in results]
    ax.plot(ts, vals)
    ax.set_xlabel('Threshold')
    ax.set_ylabel(title)
    ax.set_title(f'{title} vs Threshold')

plt.tight_layout()
plt.show()
# %%

# Additional Plots (From class)
precision_vals = [r['prec'] for r in results]
recall_vals = [r['rec'] for r in results]
fpr_vals = [r['fpr'] for r in results]
tpr_vals = [r['tpr'] for r in results]

# Precision-Recall Plot
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# ROC Curve
plt.plot(fpr_vals, tpr_vals)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show

#%%

# Find AUC (via Reimann Sum)
auc_score = 0

for i in range(len(fpr_vals) - 1):
    delta_x = (fpr_vals[i+1] - fpr_vals[i]) # (b - a) / n
    auc_score += delta_x * tpr_vals[i]

print(abs(auc_score))


#%%