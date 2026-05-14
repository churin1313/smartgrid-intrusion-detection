"""
Module 4 – Multi-Modal Fusion & Risk Analytics
File: performance_metrics.py
Author: Saanvi
Description:
This script evaluates model performance using accuracy,
precision, recall, F1 score, confusion matrix, and ROC curve.
"""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(input_file):
    """
    Evaluate fusion model performance.
    """

    data = pd.read_csv(input_file)

    y_true = data["label"]

    # Convert severity to binary prediction
    def convert_prediction(severity):
        return 0 if severity == "Low" else 1

    y_pred = data["severity_level"].apply(convert_prediction)

    # Basic Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nEvaluation Metrics:")
    print("Accuracy :", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall   :", round(recall, 3))
    print("F1 Score :", round(f1, 3))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    y_scores = data["fusion_score"]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_model("fused_risk_score.csv")
