"""
Module 4 – Multi-Modal Fusion & Risk Analytics
File: score_normalization.py
Author: Saanvi
Description:
This script performs Min-Max normalization on anomaly scores
from signal, AI, and IDS modules before fusion.
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_scores(input_file, output_file):
    """
    Normalize anomaly scores using Min-Max scaling.
    """

    data = pd.read_csv(input_file)

    scaler = MinMaxScaler()

    score_columns = ["signal", "ai", "ids"]

    data[score_columns] = scaler.fit_transform(data[score_columns])

    data.to_csv(output_file, index=False)

    print("Normalization completed and saved to:", output_file)


if __name__ == "__main__":
    normalize_scores("dummy_scores.csv", "normalized_scores.csv")
