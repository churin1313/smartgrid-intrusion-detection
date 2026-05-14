"""
=============================================================================
MODULE 1: Smart Grid Signal Processing & Cyber-Physical Data Layer
=============================================================================
Responsibilities:
  1. Smart Grid Data Modeling  – PMU, SCADA, Smart Meter simulation
  2. Signal Processing          – Kalman filter, normalization, windowing
  3. Attack Injection           – FDI, sensor spoofing, measurement manipulation

Author  : Student 1
Project : Smart Grid Cyber-Physical Attack Detection
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
rng  = np.random.default_rng(SEED)

GRID_CONFIG = {
    "n_buses"     : 14,          # IEEE 14-bus system
    "n_pmu"       : 6,           # PMU-equipped buses
    "n_meters"    : 14,          # Smart meters (one per bus)
    "freq_nominal": 60.0,        # Hz
    "v_nominal"   : 1.0,         # p.u.
    "duration_s"  : 600,         # simulation duration (seconds)
    "fs"          : 30,          # PMU sampling rate (30 samples/s)
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SMART GRID DATA SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

class PMUSimulator:
    """
    Simulates Phasor Measurement Unit (PMU) signals for a set of buses.

    Outputs per sample:
        voltage_magnitude  [p.u.]
        phase_angle        [degrees]
        frequency          [Hz]
    """

    def __init__(self, config: dict = GRID_CONFIG):
        self.cfg  = config
        self.n    = config["n_pmu"]
        self.fs   = config["fs"]
        self.T    = config["duration_s"]
        self.t    = np.arange(0, self.T, 1 / self.fs)

    def generate(self) -> pd.DataFrame:
        frames = []
        for bus_id in range(self.n):
            # Voltage magnitude: nominal ± small random walk
            v_noise  = rng.normal(0, 0.002, len(self.t)).cumsum()
            v_noise -= v_noise.mean()
            voltage  = self.cfg["v_nominal"] + 0.01 * np.sin(
                2 * np.pi * 0.05 * self.t + bus_id) + v_noise * 0.005

            # Phase angle: slow drift + harmonic
            angle = (bus_id * 15) + 2 * np.sin(2 * np.pi * 0.02 * self.t) \
                    + rng.normal(0, 0.1, len(self.t))

            # Frequency: near 60 Hz with small deviations
            freq = self.cfg["freq_nominal"] \
                   + 0.05 * np.sin(2 * np.pi * 0.01 * self.t) \
                   + rng.normal(0, 0.01, len(self.t))

            df = pd.DataFrame({
                "timestamp"         : self.t,
                "bus_id"            : bus_id,
                "voltage_magnitude" : voltage,
                "phase_angle"       : angle,
                "frequency"         : freq,
            })
            frames.append(df)

        pmu_df = pd.concat(frames, ignore_index=True)
        pmu_df["source"] = "PMU"
        return pmu_df


class SCADASimulator:
    """
    Simulates SCADA sensor data: active power (P), reactive power (Q),
    current magnitude (I) for each bus.
    """

    def __init__(self, config: dict = GRID_CONFIG):
        self.cfg = config
        self.n   = config["n_buses"]
        self.fs  = 1                             # SCADA: 1 sample / second
        self.t   = np.arange(0, config["duration_s"], 1)

    def generate(self) -> pd.DataFrame:
        frames = []
        # Load profile: a daily sinusoidal pattern compressed into 10 min
        load_curve = 0.6 + 0.4 * np.sin(np.pi * self.t / self.cfg["duration_s"])

        for bus_id in range(self.n):
            base_load = rng.uniform(0.3, 1.2)
            P = base_load * load_curve + rng.normal(0, 0.02, len(self.t))
            Q = P * rng.uniform(0.2, 0.5) + rng.normal(0, 0.01, len(self.t))
            I = np.sqrt(P**2 + Q**2) / (self.cfg["v_nominal"] + 1e-9) \
                + rng.normal(0, 0.01, len(self.t))

            df = pd.DataFrame({
                "timestamp"         : self.t,
                "bus_id"            : bus_id,
                "active_power_P"    : P,
                "reactive_power_Q"  : Q,
                "current_I"         : I,
            })
            frames.append(df)

        scada_df = pd.concat(frames, ignore_index=True)
        scada_df["source"] = "SCADA"
        return scada_df


class SmartMeterSimulator:
    """
    Simulates smart meter energy consumption data (kWh / interval).
    Resolution: 15-minute intervals.
    """

    def __init__(self, config: dict = GRID_CONFIG):
        self.cfg      = config
        self.interval = 900                      # 15 min in seconds
        self.t        = np.arange(0, config["duration_s"], self.interval)
        self.n        = config["n_meters"]

    def generate(self) -> pd.DataFrame:
        frames = []
        for meter_id in range(self.n):
            base = rng.uniform(1.0, 5.0)
            consumption = base + 0.5 * rng.normal(0, 1, len(self.t)).cumsum() \
                          * 0.01 + rng.uniform(-0.2, 0.2, len(self.t))
            consumption = np.clip(consumption, 0.1, 15)

            df = pd.DataFrame({
                "timestamp"        : self.t,
                "meter_id"         : meter_id,
                "bus_id"           : meter_id % self.cfg["n_buses"],
                "consumption_kWh"  : consumption,
            })
            frames.append(df)

        meter_df = pd.concat(frames, ignore_index=True)
        meter_df["source"] = "SmartMeter"
        return meter_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIGNAL PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter1D:
    """
    Scalar Kalman filter for real-time noise reduction of a 1-D signal.

    State model:  x_k = x_{k-1} + w,   w ~ N(0, Q)
    Observation:  z_k = x_k + v,        v ~ N(0, R)
    """

    def __init__(self, process_variance: float = 1e-4,
                 measurement_variance: float = 1e-2,
                 initial_estimate: float = 0.0,
                 initial_error: float = 1.0):
        self.Q   = process_variance
        self.R   = measurement_variance
        self.x   = initial_estimate
        self.P   = initial_error

    def update(self, measurement: float) -> float:
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K      = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        return self.x

    def filter_series(self, series: np.ndarray) -> np.ndarray:
        self.x = series[0]
        return np.array([self.update(z) for z in series])


class MovingAverageFilter:
    """Simple centred moving-average filter."""

    def __init__(self, window: int = 5):
        self.w = window

    def filter_series(self, series: np.ndarray) -> np.ndarray:
        return pd.Series(series).rolling(self.w, center=True,
                                         min_periods=1).mean().values


class SignalProcessor:
    """
    Full preprocessing pipeline:
        1. Noise filtering  (Kalman or Moving Average per column)
        2. Normalisation    (z-score or min-max)
        3. Sliding-window segmentation
    """

    def __init__(self, filter_type: str = "kalman",
                 normalise: str = "zscore",
                 window_size: int = 30,
                 step_size: int = 15):
        assert filter_type in ("kalman", "moving_average")
        assert normalise   in ("zscore", "minmax", "none")
        self.filter_type = filter_type
        self.normalise   = normalise
        self.window_size = window_size
        self.step_size   = step_size
        self._stats: Dict[str, Tuple[float, float]] = {}

    # ── filtering ──────────────────────────────────────────────────────────
    def _filter_column(self, arr: np.ndarray) -> np.ndarray:
        if self.filter_type == "kalman":
            return KalmanFilter1D(initial_estimate=arr[0]).filter_series(arr)
        else:
            return MovingAverageFilter().filter_series(arr)

    def apply_filter(self, df: pd.DataFrame,
                     numeric_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in numeric_cols:
            df[col] = self._filter_column(df[col].values)
        return df

    # ── normalisation ──────────────────────────────────────────────────────
    def fit_normalise(self, df: pd.DataFrame,
                      numeric_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in numeric_cols:
            arr = df[col].values
            if self.normalise == "zscore":
                mu, sigma = arr.mean(), arr.std() + 1e-9
                self._stats[col] = (mu, sigma)
                df[col] = (arr - mu) / sigma
            elif self.normalise == "minmax":
                lo, hi = arr.min(), arr.max() + 1e-9
                self._stats[col] = (lo, hi)
                df[col] = (arr - lo) / (hi - lo)
        return df

    def transform_normalise(self, df: pd.DataFrame,
                            numeric_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in numeric_cols:
            arr = df[col].values
            a, b = self._stats.get(col, (0, 1))
            if self.normalise == "zscore":
                df[col] = (arr - a) / (b + 1e-9)
            elif self.normalise == "minmax":
                df[col] = (arr - a) / (b - a + 1e-9)
        return df

    # ── sliding window segmentation ────────────────────────────────────────
    def sliding_windows(self, arr: np.ndarray) -> np.ndarray:
        """
        Returns shape (n_windows, window_size, n_features).
        arr shape: (T, F)
        """
        T, F   = arr.shape
        starts = range(0, T - self.window_size + 1, self.step_size)
        return np.stack([arr[s: s + self.window_size] for s in starts])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ATTACK INJECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttackConfig:
    """Parameters controlling a single attack scenario."""
    attack_type   : str            = "FDI"        # FDI | spoofing | manipulation
    target_buses  : List[int]      = field(default_factory=lambda: [2, 5])
    target_cols   : List[str]      = field(default_factory=lambda:
                                           ["voltage_magnitude", "phase_angle"])
    start_frac    : float          = 0.3           # attack start (fraction of T)
    end_frac      : float          = 0.6           # attack end   (fraction of T)
    magnitude     : float          = 0.15          # perturbation magnitude
    stealth       : bool           = True          # keep within 3-sigma to evade


class AttackInjector:
    """
    Injects cyber-physical attacks into a clean PMU / SCADA DataFrame.

    Supported attacks
    -----------------
    FDI         – False Data Injection: bias added to target measurements.
    spoofing    – Sensor Spoofing: replace readings with a constant forged value.
    manipulation – Measurement Manipulation: multiplicative scaling / sign flip.
    coordinated – Combination of FDI on voltage + angle simultaneously.
    cascading   – Sequential bus-by-bus attack to simulate cascade.
    """

    def __init__(self, config: AttackConfig = None):
        self.cfg = config or AttackConfig()

    # ── helpers ────────────────────────────────────────────────────────────
    def _attack_mask(self, df: pd.DataFrame) -> pd.Series:
        T     = df["timestamp"].max()
        t_lo  = self.cfg.start_frac * T
        t_hi  = self.cfg.end_frac   * T
        bus_m = df["bus_id"].isin(self.cfg.target_buses)
        t_m   = (df["timestamp"] >= t_lo) & (df["timestamp"] <= t_hi)
        return bus_m & t_m

    # ── individual attack methods ──────────────────────────────────────────
    def inject_fdi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        False Data Injection: add a constant or ramp bias to selected features.
        """
        df   = df.copy()
        mask = self._attack_mask(df)
        n    = mask.sum()

        for col in self.cfg.target_cols:
            if col not in df.columns:
                continue
            if self.cfg.stealth:
                sigma  = df[col].std()
                bias   = rng.uniform(1.5 * sigma, 2.5 * sigma) * self.cfg.magnitude
            else:
                bias   = df[col].mean() * self.cfg.magnitude
            # ramp up over first 20% of attack window for stealthiness
            ramp   = np.ones(n)
            ramp_n = max(1, int(0.2 * n))
            ramp[:ramp_n] = np.linspace(0, 1, ramp_n)
            df.loc[mask, col] += bias * ramp

        df.loc[mask, "label"]       = 1
        df.loc[mask, "attack_type"] = "FDI"
        return df

    def inject_spoofing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sensor Spoofing: replace signal with a plausible but fixed forged value.
        """
        df   = df.copy()
        mask = self._attack_mask(df)

        for col in self.cfg.target_cols:
            if col not in df.columns:
                continue
            forged = df.loc[~mask, col].mean() \
                     + rng.normal(0, df[col].std() * 0.1)
            df.loc[mask, col] = forged

        df.loc[mask, "label"]       = 1
        df.loc[mask, "attack_type"] = "spoofing"
        return df

    def inject_manipulation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measurement Manipulation: scale measurements by an adversarial factor.
        """
        df     = df.copy()
        mask   = self._attack_mask(df)
        factor = 1 + self.cfg.magnitude * rng.choice([-1, 1])

        for col in self.cfg.target_cols:
            if col not in df.columns:
                continue
            df.loc[mask, col] *= factor

        df.loc[mask, "label"]       = 1
        df.loc[mask, "attack_type"] = "manipulation"
        return df

    def inject_coordinated(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coordinated Attack: simultaneous FDI on voltage + phase angle.
        """
        df = df.copy()
        self.cfg.target_cols = ["voltage_magnitude"]
        df = self.inject_fdi(df)

        self.cfg.target_cols = ["phase_angle"]
        df = self.inject_fdi(df)

        mask = self._attack_mask(df)
        df.loc[mask, "attack_type"] = "coordinated"
        return df

    def inject_cascading(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cascading Attack: propagates bus-by-bus with staggered time offsets.
        """
        df   = df.copy()
        T    = df["timestamp"].max()
        step = (self.cfg.end_frac - self.cfg.start_frac) \
               / max(len(self.cfg.target_buses), 1)

        for i, bus in enumerate(self.cfg.target_buses):
            t_lo   = (self.cfg.start_frac + i * step) * T
            t_hi   = (self.cfg.start_frac + (i + 1) * step) * T
            b_mask = df["bus_id"] == bus
            t_mask = (df["timestamp"] >= t_lo) & (df["timestamp"] <= t_hi)
            mask   = b_mask & t_mask

            for col in self.cfg.target_cols:
                if col not in df.columns:
                    continue
                bias = df[col].std() * self.cfg.magnitude * (i + 1)
                df.loc[mask, col] += bias

            df.loc[mask, "label"]       = 1
            df.loc[mask, "attack_type"] = "cascading"

        return df

    # ── main entry point ───────────────────────────────────────────────────
    def inject(self, df: pd.DataFrame,
               attack_type: Optional[str] = None) -> pd.DataFrame:
        """
        Inject the configured attack type into a clean DataFrame.

        Parameters
        ----------
        df          : clean DataFrame (from any simulator)
        attack_type : override self.cfg.attack_type if provided

        Returns
        -------
        DataFrame with injected anomalies + 'label' and 'attack_type' columns.
        """
        df = df.copy()
        # initialise label columns
        df["label"]       = 0
        df["attack_type"] = "none"

        atype = attack_type or self.cfg.attack_type
        dispatch = {
            "FDI"          : self.inject_fdi,
            "spoofing"     : self.inject_spoofing,
            "manipulation" : self.inject_manipulation,
            "coordinated"  : self.inject_coordinated,
            "cascading"    : self.inject_cascading,
        }
        fn = dispatch.get(atype)
        if fn is None:
            raise ValueError(f"Unknown attack type '{atype}'. "
                             f"Choose from: {list(dispatch)}")
        return fn(df)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FULL PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class SmartGridDataPipeline:
    """
    End-to-end pipeline:
        simulate → process → inject attacks → export datasets
    """

    PMU_FEATURES   = ["voltage_magnitude", "phase_angle", "frequency"]
    SCADA_FEATURES = ["active_power_P", "reactive_power_Q", "current_I"]

    def __init__(self, grid_config: dict = GRID_CONFIG,
                 filter_type: str = "kalman",
                 normalise:   str = "zscore",
                 window_size: int = 30,
                 step_size:   int = 15):
        self.cfg       = grid_config
        self.processor = SignalProcessor(filter_type, normalise,
                                         window_size, step_size)

    # ── step 1: simulate ───────────────────────────────────────────────────
    def simulate(self) -> Dict[str, pd.DataFrame]:
        print("[Pipeline] Simulating PMU data …")
        pmu_raw = PMUSimulator(self.cfg).generate()

        print("[Pipeline] Simulating SCADA data …")
        scada_raw = SCADASimulator(self.cfg).generate()

        print("[Pipeline] Simulating Smart Meter data …")
        meter_raw = SmartMeterSimulator(self.cfg).generate()

        return {"pmu": pmu_raw, "scada": scada_raw, "meter": meter_raw}

    # ── step 2: process ────────────────────────────────────────────────────
    def process(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        print("[Pipeline] Processing PMU signals …")
        pmu_p = self.processor.apply_filter(raw["pmu"], self.PMU_FEATURES)
        pmu_p = self.processor.fit_normalise(pmu_p, self.PMU_FEATURES)

        print("[Pipeline] Processing SCADA signals …")
        scada_p = self.processor.apply_filter(raw["scada"], self.SCADA_FEATURES)
        scada_p = self.processor.fit_normalise(scada_p, self.SCADA_FEATURES)

        return {"pmu": pmu_p, "scada": scada_p, "meter": raw["meter"]}

    # ── step 3: inject attacks ─────────────────────────────────────────────
    def inject_attacks(self,
                       processed: Dict[str, pd.DataFrame],
                       attack_configs: List[AttackConfig]
                       ) -> Dict[str, pd.DataFrame]:
        """
        Apply multiple attack scenarios to the PMU dataset.
        Returns one attacked DataFrame per scenario.
        """
        results = {}
        for acfg in attack_configs:
            print(f"[Pipeline] Injecting '{acfg.attack_type}' attack …")
            injector = AttackInjector(acfg)
            attacked = injector.inject(processed["pmu"], acfg.attack_type)
            results[acfg.attack_type] = attacked
        return results

    # ── step 4: sliding-window tensors ─────────────────────────────────────
    def make_windows(self, df: pd.DataFrame,
                     feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a flat DataFrame into (X_windows, y_labels).
        X shape: (N, window_size, n_features)
        y shape: (N,)
        """
        arr   = df[feature_cols].values
        X     = self.processor.sliding_windows(arr)
        # label = max label in window (1 if any attack present)
        y_raw = df["label"].values if "label" in df.columns \
                else np.zeros(len(df))
        T, W, S = len(y_raw), self.processor.window_size, self.processor.step_size
        y = np.array([y_raw[s: s + W].max()
                      for s in range(0, T - W + 1, S)])
        return X, y

    # ── full run ───────────────────────────────────────────────────────────
    def run(self, attack_configs: Optional[List[AttackConfig]] = None
            ) -> Dict:
        if attack_configs is None:
            attack_configs = [
                AttackConfig("FDI",         [2, 5], self.PMU_FEATURES),
                AttackConfig("spoofing",    [0, 3], self.PMU_FEATURES),
                AttackConfig("manipulation",[1],    ["voltage_magnitude"]),
                AttackConfig("coordinated", [4, 6], self.PMU_FEATURES),
                AttackConfig("cascading",   list(range(6)), self.PMU_FEATURES),
            ]

        raw       = self.simulate()
        processed = self.process(raw)
        attacked  = self.inject_attacks(processed, attack_configs)

        # build window datasets for each scenario
        datasets = {}
        feat_cols = self.PMU_FEATURES

        print("[Pipeline] Building clean window dataset …")
        X_clean, _ = self.make_windows(
            processed["pmu"].assign(label=0), feat_cols)
        datasets["clean"] = {"X": X_clean,
                              "y": np.zeros(len(X_clean), dtype=int),
                              "df": processed["pmu"]}

        for atype, adf in attacked.items():
            print(f"[Pipeline] Building window dataset for '{atype}' …")
            X_att, y_att = self.make_windows(adf, feat_cols)
            datasets[atype] = {"X": X_att, "y": y_att, "df": adf}

        datasets["raw_scada"] = {"df": processed["scada"]}
        datasets["raw_meter"] = {"df": raw["meter"]}

        print("[Pipeline] Done.")
        return datasets


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATASET EXPORT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def export_datasets(datasets: Dict, prefix: str = "smartgrid") -> None:
    """Save all DataFrames to CSV files."""
    for name, data in datasets.items():
        if "df" in data:
            path = f"{prefix}_{name}.csv"
            data["df"].to_csv(path, index=False)
            print(f"  Saved → {path}  ({len(data['df'])} rows)")

    # also save numpy arrays
    for name, data in datasets.items():
        if "X" in data:
            np.save(f"{prefix}_{name}_X.npy", data["X"])
            np.save(f"{prefix}_{name}_y.npy", data["y"])
            print(f"  Saved → {prefix}_{name}_X.npy  "
                  f"shape={data['X'].shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QUICK-START DEMO
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(datasets: Dict) -> None:
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    for name, data in datasets.items():
        tag = ""
        if "X" in data:
            X, y = data["X"], data["y"]
            n_att = y.sum()
            tag = (f"windows={X.shape[0]}, shape={X.shape}, "
                   f"attack_windows={int(n_att)} ({100*n_att/len(y):.1f}%)")
        elif "df" in data:
            tag = f"rows={len(data['df'])}"
        print(f"  {name:<20} {tag}")
    print("=" * 60)


if __name__ == "__main__":
    # ── configure custom attack scenarios (optional) ────────────────────────
    custom_attacks = [
        AttackConfig(
            attack_type  = "FDI",
            target_buses = [0, 2, 5],
            target_cols  = ["voltage_magnitude", "phase_angle"],
            start_frac   = 0.25,
            end_frac     = 0.55,
            magnitude    = 0.20,
            stealth      = True,
        ),
        AttackConfig(
            attack_type  = "cascading",
            target_buses = list(range(6)),
            target_cols  = ["voltage_magnitude", "frequency"],
            start_frac   = 0.60,
            end_frac     = 0.90,
            magnitude    = 0.15,
            stealth      = False,
        ),
    ]

    pipeline = SmartGridDataPipeline(
        grid_config = GRID_CONFIG,
        filter_type = "kalman",      # or "moving_average"
        normalise   = "zscore",      # or "minmax" / "none"
        window_size = 30,
        step_size   = 15,
    )

    datasets = pipeline.run(attack_configs=custom_attacks)
    print_summary(datasets)

    # Optionally export to disk:
    # export_datasets(datasets, prefix="smartgrid")

    # ── access tensors for Module 2 ─────────────────────────────────────────
    X_clean   = datasets["clean"]["X"]          # shape (N, 30, 3)
    y_clean   = datasets["clean"]["y"]

    X_fdi     = datasets["FDI"]["X"]
    y_fdi     = datasets["FDI"]["y"]

    X_cascade = datasets["cascading"]["X"]
    y_cascade = datasets["cascading"]["y"]

    print(f"\nReady for Module 2:")
    print(f"  X_clean   : {X_clean.shape}   y_clean   : {y_clean.shape}")
    print(f"  X_fdi     : {X_fdi.shape}   y_fdi     : {y_fdi.shape}")
    print(f"  X_cascade : {X_cascade.shape}   y_cascade : {y_cascade.shape}")
