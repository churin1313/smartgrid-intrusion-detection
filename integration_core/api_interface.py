# integration_core/api_interface.py
# Handles: Reading / Writing CSVs between modules + Mock Data Generation
# Student 3 – Core CSE – Varun

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger("SmartGrid")

# ──────────────────────────────────────────────
#  Path constants  (relative to project root)
# ──────────────────────────────────────────────

DATA_PROCESSED = os.path.join("data", "processed")
RESULTS_DIR    = os.path.join("results")

# Standard filenames agreed across the team
FILENAMES = {
    "signal":   "processed_signal_features.csv",   # ECE  – Pranav
    "ai":       "ai_anomaly_scores.csv",            # AI   – Shashwat
    "ids":      "network_anomaly_scores.csv",       # IT   – Nithila
    "fusion":   "fused_risk_score.csv",             # DS   – Saanvi
    "defense":  "optimal_defense_action.csv",       # IT   – Nithila (game theory)
    "output":   "final_outputs.csv",                # CSE  – Varun (this module)
}

N_SAMPLES = 50   # rows of simulated data per CSV


# ──────────────────────────────────────────────
#  Mock data generators
#  (stand-ins until teammates push their CSVs)
# ──────────────────────────────────────────────

def _mock_signal_features() -> pd.DataFrame:
    """
    Simulates processed_signal_features.csv from the ECE module.
    Columns: timestamp, bus_id, voltage_magnitude, phase_angle,
             frequency, residual, is_attack
    """
    rng = np.random.default_rng(42)
    timestamps = [
        (datetime(2024, 1, 1) + timedelta(seconds=i * 0.1)).isoformat()
        for i in range(N_SAMPLES)
    ]
    attack_mask = rng.integers(0, 2, size=N_SAMPLES)   # 0 = normal, 1 = attack
    return pd.DataFrame({
        "timestamp":         timestamps,
        "bus_id":            rng.integers(1, 14, size=N_SAMPLES),
        "voltage_magnitude": np.where(
            attack_mask,
            rng.uniform(0.85, 1.05, N_SAMPLES),   # anomalous range
            rng.uniform(0.95, 1.05, N_SAMPLES),   # normal range
        ).round(4),
        "phase_angle":       rng.uniform(-30, 30, size=N_SAMPLES).round(4),
        "frequency":         rng.normal(50.0, 0.05, size=N_SAMPLES).round(4),
        "residual":          np.where(
            attack_mask,
            rng.uniform(0.4, 1.0, N_SAMPLES),
            rng.uniform(0.0, 0.15, N_SAMPLES),
        ).round(4),
        "signal_anomaly_score": np.where(
            attack_mask,
            rng.uniform(0.55, 0.99, N_SAMPLES),
            rng.uniform(0.01, 0.30, N_SAMPLES),
        ).round(4),
        "is_attack":         attack_mask,
    })


def _mock_ai_scores() -> pd.DataFrame:
    """
    Simulates ai_anomaly_scores.csv from the AI/ML module (GNN + LSTM).
    Columns: timestamp, bus_id, ai_anomaly_score, predicted_attack_type
    """
    rng = np.random.default_rng(7)
    attack_types = ["FDI", "Sensor Spoofing", "Stealth", "Cascading", "Normal"]
    df_signal = _mock_signal_features()
    scores = np.where(
        df_signal["is_attack"],
        rng.uniform(0.50, 0.99, N_SAMPLES),
        rng.uniform(0.01, 0.35, N_SAMPLES),
    ).round(4)
    predicted = np.where(
        df_signal["is_attack"],
        rng.choice(attack_types[:-1], size=N_SAMPLES),
        "Normal",
    )
    return pd.DataFrame({
        "timestamp":         df_signal["timestamp"],
        "bus_id":            df_signal["bus_id"],
        "ai_anomaly_score":  scores,
        "predicted_attack_type": predicted,
    })


def _mock_ids_scores() -> pd.DataFrame:
    """
    Simulates network_anomaly_scores.csv from the IT/IDS module.
    Columns: timestamp, src_ip, dst_ip, ids_anomaly_score, attack_category
    """
    rng = np.random.default_rng(13)
    df_signal = _mock_signal_features()
    categories = ["DDoS", "Replay", "Command Injection", "Malware", "Normal"]
    scores = np.where(
        df_signal["is_attack"],
        rng.uniform(0.45, 0.98, N_SAMPLES),
        rng.uniform(0.01, 0.25, N_SAMPLES),
    ).round(4)
    cat = np.where(
        df_signal["is_attack"],
        rng.choice(categories[:-1], size=N_SAMPLES),
        "Normal",
    )
    src_ips = [f"192.168.1.{rng.integers(1, 255)}" for _ in range(N_SAMPLES)]
    dst_ips = [f"10.0.0.{rng.integers(1, 50)}"     for _ in range(N_SAMPLES)]
    return pd.DataFrame({
        "timestamp":          df_signal["timestamp"],
        "src_ip":             src_ips,
        "dst_ip":             dst_ips,
        "ids_anomaly_score":  scores,
        "attack_category":    cat,
    })


def _mock_fusion_scores(
    df_signal: pd.DataFrame,
    df_ai:     pd.DataFrame,
    df_ids:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulates fused_risk_score.csv from the Data Science module.
    Uses weighted average: w_signal=0.30, w_ai=0.45, w_ids=0.25
    """
    w_signal, w_ai, w_ids = 0.30, 0.45, 0.25
    fused = (
        w_signal * df_signal["signal_anomaly_score"].values +
        w_ai     * df_ai["ai_anomaly_score"].values         +
        w_ids    * df_ids["ids_anomaly_score"].values
    ).round(4)

    risk_index = np.clip(fused * 10, 0, 10).round(2)

    severity = np.select(
        [fused < 0.35, fused < 0.65, fused < 0.85],
        ["LOW",        "MEDIUM",      "HIGH"],
        default="CRITICAL"
    )

    return pd.DataFrame({
        "timestamp":       df_signal["timestamp"],
        "signal_score":    df_signal["signal_anomaly_score"].values,
        "ai_score":        df_ai["ai_anomaly_score"].values,
        "ids_score":       df_ids["ids_anomaly_score"].values,
        "fused_score":     fused,
        "risk_index":      risk_index,
        "severity_level":  severity,
        "attack_type":     df_ai["predicted_attack_type"].values,
    })


def _mock_defense_actions(df_fusion: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates optimal_defense_action.csv from the game-theoretic engine.
    Maps severity → Stackelberg-optimal defender strategy.
    """
    strategy_map = {
        "LOW":      "Monitor",
        "MEDIUM":   "Trigger Alert",
        "HIGH":     "Isolate Node",
        "CRITICAL": "Reconfigure Grid",
    }
    actions = df_fusion["severity_level"].map(strategy_map).fillna("Monitor")
    return pd.DataFrame({
        "timestamp":      df_fusion["timestamp"],
        "fused_score":    df_fusion["fused_score"],
        "severity_level": df_fusion["severity_level"],
        "defense_action": actions,
        "attacker_utility": (1 - df_fusion["fused_score"].values).round(4),
        "defender_utility": df_fusion["fused_score"].values.round(4),
    })


# ──────────────────────────────────────────────
#  Initialiser – generates all mock CSVs once
# ──────────────────────────────────────────────

def initialise_mock_data() -> None:
    """
    Called at pipeline startup.
    Writes mock CSV files to data/processed/ ONLY if they don't already
    exist (i.e., real teammates' files take priority).
    """
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)

    def _write_if_missing(df: pd.DataFrame, fname: str) -> None:
        path = os.path.join(DATA_PROCESSED, fname)
        if not os.path.exists(path):
            df.to_csv(path, index=False)
            logger.debug(f"  [MOCK] Generated → {path}  ({len(df)} rows)")
        else:
            logger.debug(f"  [REAL] Found existing → {path}")

    logger.info("Initialising data layer (mock stubs for missing module outputs)…")

    df_sig = _mock_signal_features()
    df_ai  = _mock_ai_scores()
    df_ids = _mock_ids_scores()
    df_fus = _mock_fusion_scores(df_sig, df_ai, df_ids)
    df_def = _mock_defense_actions(df_fus)

    _write_if_missing(df_sig, FILENAMES["signal"])
    _write_if_missing(df_ai,  FILENAMES["ai"])
    _write_if_missing(df_ids, FILENAMES["ids"])
    _write_if_missing(df_fus, FILENAMES["fusion"])
    _write_if_missing(df_def, FILENAMES["defense"])

    logger.info("Data layer ready.")


# ──────────────────────────────────────────────
#  APIInterface – standardised read / write
# ──────────────────────────────────────────────

class APIInterface:
    """
    Single point of contact for all inter-module I/O.

    All modules exchange data as CSV files in data/processed/.
    This class enforces the agreed column schema so integration
    never breaks due to naming mismatches.
    """

    # Required columns per module output
    _SCHEMA = {
        "signal": ["timestamp", "signal_anomaly_score"],
        "ai":     ["timestamp", "ai_anomaly_score", "predicted_attack_type"],
        "ids":    ["timestamp", "ids_anomaly_score", "attack_category"],
        "fusion": ["timestamp", "fused_score", "severity_level", "attack_type"],
        "defense":["timestamp", "defense_action", "fused_score"],
    }

    def __init__(self):
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        os.makedirs(RESULTS_DIR,    exist_ok=True)

    # ── Generic helpers ─────────────────────

    def _path(self, key: str) -> str:
        return os.path.join(DATA_PROCESSED, FILENAMES[key])

    def _read(self, key: str) -> pd.DataFrame:
        path = self._path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected module output not found: {path}\n"
                "Run initialise_mock_data() first or ask the responsible student to share their CSV."
            )
        df = pd.read_csv(path)
        self._validate_schema(df, key)
        logger.debug(f"  ← Read {key:8s} | {path}  ({len(df)} rows, {len(df.columns)} cols)")
        return df

    def _validate_schema(self, df: pd.DataFrame, key: str) -> None:
        required = self._SCHEMA.get(key, [])
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Schema violation in '{key}' CSV: missing columns {missing}\n"
                f"Found columns: {list(df.columns)}"
            )

    # ── Public read methods ─────────────────

    def load_signal_features(self) -> pd.DataFrame:
        logger.info("  Loading signal features  (ECE – Pranav)…")
        df = self._read("signal")
        logger.info(f"  ✔ {len(df)} signal samples loaded.")
        return df

    def load_ai_scores(self) -> pd.DataFrame:
        logger.info("  Loading AI anomaly scores  (AI/ML – Shashwat)…")
        df = self._read("ai")
        logger.info(f"  ✔ {len(df)} AI predictions loaded.")
        return df

    def load_ids_scores(self) -> pd.DataFrame:
        logger.info("  Loading IDS network scores  (IT – Nithila)…")
        df = self._read("ids")
        logger.info(f"  ✔ {len(df)} IDS records loaded.")
        return df

    def load_fusion_scores(self) -> pd.DataFrame:
        logger.info("  Loading fused risk scores  (Data Science – Saanvi)…")
        df = self._read("fusion")
        logger.info(f"  ✔ {len(df)} fused records loaded.")
        return df

    def load_defense_actions(self) -> pd.DataFrame:
        logger.info("  Loading game-theoretic defense actions  (IT – Nithila)…")
        df = self._read("defense")
        logger.info(f"  ✔ {len(df)} defense decisions loaded.")
        return df

    # ── Public write methods ────────────────

    def save_final_output(self, df: pd.DataFrame) -> str:
        path = os.path.join(RESULTS_DIR, FILENAMES["output"])
        df.to_csv(path, index=False)
        logger.info(f"  ✔ Final outputs saved → {path}")
        return path

    def save_report_row(self, report: dict) -> str:
        """Appends a single-row summary to results/final_outputs.csv."""
        path = os.path.join(RESULTS_DIR, FILENAMES["output"])
        df   = pd.DataFrame([report])
        write_header = not os.path.exists(path)
        df.to_csv(path, mode="a", header=write_header, index=False)
        return path
