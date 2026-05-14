Module 4 – Multi-Modal Fusion & Risk Analytics

Author: Saanvi
Specialization: Data Science

PROJECT OBJECTIVE

This module implements a multi-modal anomaly score fusion system for smart grid cybersecurity.
It combines anomaly scores from signal-based detection, AI-based detection, and IDS detection to generate a unified risk assessment.

SYSTEM WORKFLOW

Score Normalization

Min-Max scaling is applied to bring all anomaly scores into the range 0 to 1.

Weighted Fusion

Normalized scores are combined using weighted aggregation.

Risk Index Calculation

The fusion score is used as the Grid Risk Index.

Severity Classification

Risk levels are classified as Low, Medium, or Critical.

Performance Evaluation

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

ROC Curve

FUSION FORMULA

Fusion Score =
0.3 × Signal Score

0.4 × AI Score

0.3 × IDS Score

SEVERITY LEVELS

0.0 – 0.3 → Low
0.3 – 0.7 → Medium

0.7 → Critical

TECHNOLOGIES USED

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

HOW TO RUN

Activate virtual environment:
source venv/bin/activate

Run the pipeline:
python score_normalization.py
python fusion_engine.py
python risk_scoring.py
python performance_metrics.py
