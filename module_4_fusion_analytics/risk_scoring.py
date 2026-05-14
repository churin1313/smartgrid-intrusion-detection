"""
Module 4 – Risk Scoring & Severity Classification
"""

import pandas as pd


def classify_risk(score):

    if score < 0.3:
        return "Low"

    elif score < 0.7:
        return "Medium"

    else:
        return "Critical"


def perform_risk_scoring(input_file, output_file):

    # Load fused scores
    data = pd.read_csv(input_file)

    # Create risk score
    data["risk_score"] = data["fused_score"]

    # Severity classification
    data["severity_level"] = data["risk_score"].apply(classify_risk)

    # Save final output
    data.to_csv(output_file, index=False)

    print("Risk scoring completed and saved to:", output_file)


if __name__ == "__main__":

    perform_risk_scoring(
        "module_4_fusion_analytics/fused_output.csv",
        "module_4_fusion_analytics/fused_risk_score.csv"
    )
