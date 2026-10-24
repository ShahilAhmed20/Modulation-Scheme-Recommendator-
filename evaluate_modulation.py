# Modulation Scheme Classifier - Evaluation
# Generates: Confusion Matrix, Precision Matrix, F1 score -> saved to "F:\Mini project"
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
matplotlib.use("Agg")

OUT = r"F:\Mini project"

df = pd.read_csv(OUT + r"\modulation_dataset_balanced.csv")

feature_cols = ['snr_db', 'effective_snr_db', 'distance_m', 'tx_power_dbm',
                'bandwidth_mhz', 'frequency_ghz', 'target_rate_mbps',
                'velocity_kmh', 'spectral_efficiency_needed']

X = df[feature_cols]
y = df['recommended_modulation']

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)
class_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Train ----------------
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                          random_state=42)
model.fit(X_train, y_train)

# ---------------- Predictions ----------------
y_pred = model.predict(X_test)

# ---------------- Evaluation ----------------
from sklearn.metrics import (accuracy_score, precision_score, f1_score,
                             recall_score, confusion_matrix,
                             classification_report,
                             precision_recall_fscore_support)

accuracy  = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro    = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro        = f1_score(y_test, y_pred, average='macro', zero_division=0)
f1_weighted     = f1_score(y_test, y_pred, average='weighted', zero_division=0)
cm  = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

report = classification_report(y_test, y_pred,
                               target_names=class_names, zero_division=0, digits=4)

print("Accuracy:", accuracy)

# ---------------- Save text report ----------------
with open(OUT + r"\evaluation_results.txt", "w") as f:
    f.write("MODULATION SCHEME CLASSIFIER - EVALUATION RESULTS\n")
    f.write("Model: XGBoost (300 trees, depth 8) | Dataset: modulation_dataset_balanced.csv\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Accuracy (macro)      : {accuracy:.4f}\n")
    f.write(f"Precision (macro)     : {precision_macro:.4f}\n")
    f.write(f"Recall    (macro)     : {recall_macro:.4f}\n")
    f.write(f"F1 Score  (macro)     : {f1_macro:.4f}\n")
    f.write(f"F1 Score  (weighted)  : {f1_weighted:.4f}\n\n")
    f.write("Confusion Matrix (counts):\n")
    f.write(np.array2string(cm) + "\n\n")
    f.write("Confusion Matrix (normalized):\n")
    f.write(np.array2string(cm_normalized, precision=4) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

# ---------------- Confusion matrix figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
            xticklabels=class_names, yticklabels=class_names)
ax[0].set_title("Confusion Matrix (Counts)")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=ax[1],
            xticklabels=class_names, yticklabels=class_names)
ax[1].set_title("Confusion Matrix (Normalized)")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(OUT + r"\confusion_matrix.png", dpi=150)
plt.close()

# ---------------- Precision matrix figure ----------------
cm_sum = cm.sum(axis=0, keepdims=True)   # predicted totals
with np.errstate(divide='ignore', invalid='ignore'):
    precision_matrix = cm / cm_sum
per_class_p, per_class_r, per_class_f, support = \
    precision_recall_fscore_support(y_test, y_pred, zero_division=0)

precision_matrix = np.nan_to_num(precision_matrix)

# ---------------- Recall matrix figure ----------------
# Recall matrix: rows = actual class, cols = predicted class,
# value = fraction of that actual class predicted as each class
cm_rowsum = cm.sum(axis=1, keepdims=True)       # row sums (actual totals)
with np.errstate(divide='ignore', invalid='ignore'):
    recall_matrix = cm / cm_rowsum
recall_matrix = np.nan_to_num(recall_matrix)

n = len(class_names)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(recall_matrix, annot=True, fmt='.3f', cmap='Oranges', ax=ax[0],
            xticklabels=class_names, yticklabels=class_names)
ax[0].set_title("Recall Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

x = np.arange(n)
ax[1].bar(x - 0.20, per_class_r, 0.4, label='Recall', color='#e65100')
ax[1].bar(x + 0.20, per_class_f, 0.4, label='F1 score', color='#1565c0')
ax[1].set_xticks(x)
ax[1].set_xticklabels(class_names)
ax[1].set_ylim(0, 1.15)
ax[1].set_title("Per-class Recall & F1")
ax[1].legend()
for b in ax[1].containers:
    ax[1].bar_label(b, fmt='%.2f', fontsize=8)

plt.tight_layout()
plt.savefig(OUT + r"\recall_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(precision_matrix, annot=True, fmt='.3f', cmap='Greens', ax=ax[0],
            xticklabels=class_names, yticklabels=class_names)
ax[0].set_title("Precision Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

per_class_p, per_class_r, per_class_f, support = \
    precision_recall_fscore_support(y_test, y_pred, zero_division=0)
x = np.arange(n)
ax[1].bar(x - 0.20, per_class_p, 0.4, label='Precision', color='#2e7d32')
ax[1].bar(x + 0.20, per_class_f, 0.4, label='F1 score', color='#1565c0')
ax[1].set_xticks(x)
ax[1].set_xticklabels(class_names)
ax[1].set_ylim(0, 1.15)
ax[1].set_title("Per-class Precision & F1")
ax[1].legend()
for b in ax[1].containers:
    ax[1].bar_label(b, fmt='%.2f', fontsize=8)

plt.tight_layout()
plt.savefig(OUT + r"\precision_matrix.png", dpi=150)
plt.close()

print(f"\nSaved to {OUT}:")
print("  - evaluation_results.txt")
print("  - confusion_matrix.png")
print("  - precision_matrix.png")
print(f"Accuracy={accuracy:.4f}  Precision(macro)={precision_macro:.4f}  F1(macro)={f1_macro:.4f}")
