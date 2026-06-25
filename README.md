# SmartGrid-CyberPhysical-GameTheoretic-IDS

**Game-Theoretic Multi-Model Cyber-Physical Attack Detection Framework for Smart Grid**  
VIT SCORE | Guide: Dr. Priya V, Sr Associate Professor

---

## Student 3 – Core CSE – Varun (25BCE2455)
**Role: System Integrator & Technical Lead**

This repository contains the complete system architecture, backend integration framework, and final executable pipeline for the multidisciplinary project.

---

## Project Architecture

```
Smart Grid Data (PMU + SCADA + IoT + Network Traffic)
        ↓
  [ECE – Pranav]        Signal Processing & Feature Extraction
        ↓
  [AI  – Shashwat]      GNN + LSTM Spatio-Temporal Detection
        ↓
  [IT  – Nithila]       Network IDS (RF/XGBoost/Autoencoder)
        ↓
  [DS  – Saanvi]        Multi-Modal Fusion + Risk Analytics
        ↓
  [IT  – Nithila]       Stackelberg Game-Theoretic Defense
        ↓
  [CSE – Varun] ★       System Integration & Final Pipeline
        ↓
     OUTPUT: Attack Type + Severity + Defense Action + Resilience Score
```

---

## Module 3 Files (Varun's Deliverables)

| File | Purpose |
|------|---------|
| `main.py` | Final orchestration entry point – runs end-to-end pipeline |
| `integration_core/pipeline_manager.py` | Manages 6-step execution sequence |
| `integration_core/api_interface.py` | Standardised CSV I/O + mock data generation |
| `integration_core/system_controller.py` | Logging, timing, error handling, report generation |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python main.py
```

---

## Data Flow

All modules exchange data as CSV files in `data/processed/`:

| CSV File | Owner | Description |
|----------|-------|-------------|
| `processed_signal_features.csv` | ECE – Pranav | PMU/SCADA signal anomaly scores |
| `ai_anomaly_scores.csv` | AI – Shashwat | GNN+LSTM prediction scores |
| `network_anomaly_scores.csv` | IT – Nithila | IDS network anomaly scores |
| `fused_risk_score.csv` | DS – Saanvi | Weighted fusion + severity |
| `optimal_defense_action.csv` | IT – Nithila | Stackelberg defense strategy |
| `results/final_outputs.csv` | CSE – Varun | Final consolidated output |

> **Note:** If a teammate's CSV is not yet available, the system auto-generates realistic mock data so the pipeline always runs.

---

## Integration Notes for Teammates

- Use the **same column names** as defined in `api_interface.py` (`_SCHEMA` dict)
- Save your output CSV to `data/processed/` with the exact filename shown above
- Once your real CSV is in place, the mock generator will **not overwrite it**
- All scores should be **normalised to [0, 1]**
