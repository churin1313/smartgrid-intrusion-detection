import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

# ─────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────
print("📂 Loading raw data...")
pmu_df = pd.read_csv("data/raw/pmu_data.csv")
scada_df = pd.read_csv("data/raw/scada_data.csv")
smart_meter_df = pd.read_csv("data/raw/smart_meter_data.csv")
print(f"✅ PMU Data Loaded! Shape: {pmu_df.shape}")
print(f"✅ SCADA Data Loaded! Shape: {scada_df.shape}")
print(f"✅ Smart Meter Data Loaded! Shape: {smart_meter_df.shape}")

# ─────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ─────────────────────────────────────────
print("\n🔧 Handling missing values...")
pmu_df.fillna(method="ffill", inplace=True)
scada_df.fillna(method="ffill", inplace=True)
smart_meter_df.fillna(method="ffill", inplace=True)
print("✅ Missing values handled!")

# ─────────────────────────────────────────
# 3. KALMAN FILTER FUNCTION
# ─────────────────────────────────────────
def apply_kalman_filter(data):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[data[0]]])
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.P *= 1000.
    kf.R = 0.1
    kf.Q = 0.01
    filtered = []
    for z in data:
        kf.predict()
        kf.update(np.array([[z]]))
        filtered.append(float(kf.x[0]))
    return filtered

# ─────────────────────────────────────────
# 4. APPLY KALMAN FILTER TO PMU SIGNALS
# ─────────────────────────────────────────
print("\n🔧 Applying Kalman Filter to PMU signals...")
pmu_df["voltage_pu"] = apply_kalman_filter(pmu_df["voltage_pu"].values)
pmu_df["frequency_hz"] = apply_kalman_filter(pmu_df["frequency_hz"].values)
pmu_df["phase_angle_deg"] = apply_kalman_filter(pmu_df["phase_angle_deg"].values)
print("✅ Kalman Filter Applied!")

# ─────────────────────────────────────────
# 5. APPLY MOVING AVERAGE TO SCADA DATA
# ─────────────────────────────────────────
print("\n🔧 Applying Moving Average to SCADA signals...")
window = 5
scada_df["active_power_mw"] = scada_df["active_power_mw"].rolling(window, min_periods=1).mean()
scada_df["reactive_power_mvar"] = scada_df["reactive_power_mvar"].rolling(window, min_periods=1).mean()
scada_df["current_ka"] = scada_df["current_ka"].rolling(window, min_periods=1).mean()
print("✅ Moving Average Applied!")

# ─────────────────────────────────────────
# 6. NORMALIZATION (MIN-MAX SCALING)
# ─────────────────────────────────────────
print("\n🔧 Normalizing data...")
def minmax_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Normalize PMU
pmu_df["voltage_pu"] = minmax_normalize(pmu_df["voltage_pu"])
pmu_df["frequency_hz"] = minmax_normalize(pmu_df["frequency_hz"])
pmu_df["phase_angle_deg"] = minmax_normalize(pmu_df["phase_angle_deg"])

# Normalize SCADA
scada_df["active_power_mw"] = minmax_normalize(scada_df["active_power_mw"])
scada_df["reactive_power_mvar"] = minmax_normalize(scada_df["reactive_power_mvar"])
scada_df["current_ka"] = minmax_normalize(scada_df["current_ka"])

# Normalize Smart Meter
smart_meter_df["energy_consumption_kwh"] = minmax_normalize(smart_meter_df["energy_consumption_kwh"])
smart_meter_df["power_factor"] = minmax_normalize(smart_meter_df["power_factor"])
print("✅ Normalization Complete!")

# ─────────────────────────────────────────
# 7. TIME-SERIES SLIDING WINDOW SEGMENTATION
# ─────────────────────────────────────────
print("\n🔧 Applying Sliding Window Segmentation...")
def sliding_window(df, window_size=60, step=10):
    segments = []
    for i in range(0, len(df) - window_size, step):
        segment = df.iloc[i:i + window_size].copy()
        segment["window_id"] = i // step
        segments.append(segment)
    return pd.concat(segments, ignore_index=True)

pmu_windowed = sliding_window(pmu_df)
scada_windowed = sliding_window(scada_df)
print(f"✅ PMU Windowed Shape: {pmu_windowed.shape}")
print(f"✅ SCADA Windowed Shape: {scada_windowed.shape}")

# ─────────────────────────────────────────
# 8. SAVE PROCESSED DATA
# ─────────────────────────────────────────
print("\n💾 Saving processed data...")
pmu_df.to_csv("data/processed/pmu_processed.csv", index=False)
scada_df.to_csv("data/processed/scada_processed.csv", index=False)
smart_meter_df.to_csv("data/processed/smart_meter_processed.csv", index=False)
pmu_windowed.to_csv("data/processed/pmu_windowed.csv", index=False)
scada_windowed.to_csv("data/processed/scada_windowed.csv", index=False)

print("✅ All processed files saved to data/processed/")
print("   → pmu_processed.csv")
print("   → scada_processed.csv")
print("   → smart_meter_processed.csv")
print("   → pmu_windowed.csv")
print("   → scada_windowed.csv")
print("\n🎉 signal_preprocessing.py COMPLETE!")