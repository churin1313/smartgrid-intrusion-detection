"""
Module 4 – Multi-Modal Fusion & Risk Analytics
File: fusion_engine.py
Author: Saanvi

Description:
This script performs weighted fusion of anomaly scores
to generate a unified fusion anomaly score.
"""

import pandas as pd


def perform_fusion(input_file, output_file):
    """
    Perform weighted fusion of anomaly scores.
    """

    # Load AI anomaly scores from teammate module
    data = pd.read_csv(input_file)
    # Load IDS / Network anomaly scores
    network_data = pd.read_csv(
    "module_4_fusion_analytics/network_anomaly_scores.csv"
    )

# Merge AI and IDS scores using window_id
    data = pd.merge(
        data,
        network_data,
        on="window_id"
    )

# Fusion Formula
    data["fused_score"] = (
        0.6 * data["ai_anomaly_score"] +
        0.4 * data["network_anomaly_score"]
    )

    # Save fused output
    data.to_csv(output_file, index=False)

    print("Fusion completed and saved to:", output_file)


if __name__ == "__main__":
    perform_fusion(
        "module_4_fusion_analytics/ai_anomaly_scores.csv",
        "module_4_fusion_analytics/fused_output.csv"
    )