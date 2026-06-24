import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ─────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────

print("Loading predictions...")

with open("outputs/final_predictions.json", "r") as f:
    data = json.load(f)

y_true = np.array(data["ground_truth"])
y_pred = np.array(data["predictions"])


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ─────────────────────────────────────────────
# OPTIONAL: SAVE METRICS
# ─────────────────────────────────────────────

results = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist()
}

with open("outputs/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nSaved metrics → outputs/metrics.json")