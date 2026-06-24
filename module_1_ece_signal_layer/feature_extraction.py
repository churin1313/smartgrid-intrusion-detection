import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# 1. LOAD ATTACKED DATA
# ─────────────────────────────────────────
print("📂 Loading attacked data...")
pmu_df = pd.read_csv("data/processed/pmu_attacked.csv")
scada_df = pd.read_csv("data/processed/scada_attacked.csv")
print(f"✅ PMU Data Loaded! Shape: {pmu_df.shape}")
print(f"✅ SCADA Data Loaded! Shape: {scada_df.shape}")

# ─────────────────────────────────────────
# 2. PMU FEATURE EXTRACTION
# ─────────────────────────────────────────
print("\n🔧 Extracting PMU Features...")
WINDOW_SIZE = 60

pmu_features = []
for bus_id in pmu_df["bus_id"].unique():
    bus_data = pmu_df[pmu_df["bus_id"] == bus_id].reset_index(drop=True)
    for i in range(0, len(bus_data) - WINDOW_SIZE, 10):
        window = bus_data.iloc[i:i + WINDOW_SIZE]
        pmu_features.append({
            # Identity
            "bus_id": bus_id,
            "window_id": i // 10,
            # Voltage features
            "voltage_mean": window["voltage_pu"].mean(),
            "voltage_std": window["voltage_pu"].std(),
            "voltage_min": window["voltage_pu"].min(),
            "voltage_max": window["voltage_pu"].max(),
            # Frequency features
            "frequency_mean": window["frequency_hz"].mean(),
            "frequency_std": window["frequency_hz"].std(),
            # Phase angle features
            "phase_angle_mean": window["phase_angle_deg"].mean(),
            "phase_angle_std": window["phase_angle_deg"].std(),
            # Rate of change
            "voltage_delta": window["voltage_pu"].diff().abs().mean(),
            "frequency_delta": window["frequency_hz"].diff().abs().mean(),
            # Residual score
            "residual_score": (window["voltage_pu"] - window["voltage_pu"].mean()).abs().mean(),
            # Labels
            "attack_label": int(window["attack_label"].max()),
            "attack_type": window["attack_type"].mode()[0]
        })

pmu_features_df = pd.DataFrame(pmu_features)
print(f"✅ PMU Features Extracted! Shape: {pmu_features_df.shape}")

# ─────────────────────────────────────────
# 3. SCADA FEATURE EXTRACTION
# ─────────────────────────────────────────
print("\n🔧 Extracting SCADA Features...")
scada_features = []
for bus_id in scada_df["bus_id"].unique():
    bus_data = scada_df[scada_df["bus_id"] == bus_id].reset_index(drop=True)
    for i in range(0, len(bus_data) - WINDOW_SIZE, 10):
        window = bus_data.iloc[i:i + WINDOW_SIZE]
        scada_features.append({
            # Identity
            "bus_id": bus_id,
            "window_id": i // 10,
            # Active power features
            "active_power_mean": window["active_power_mw"].mean(),
            "active_power_std": window["active_power_mw"].std(),
            "active_power_max": window["active_power_mw"].max(),
            # Reactive power features
            "reactive_power_mean": window["reactive_power_mvar"].mean(),
            "reactive_power_std": window["reactive_power_mvar"].std(),
            # Current features
            "current_mean": window["current_ka"].mean(),
            "current_std": window["current_ka"].std(),
            # Labels
            "attack_label": int(window["attack_label"].max()),
            "attack_type": window["attack_type"].mode()[0]
        })

scada_features_df = pd.DataFrame(scada_features)
print(f"✅ SCADA Features Extracted! Shape: {scada_features_df.shape}")

# ─────────────────────────────────────────
# 4. MERGE PMU + SCADA FEATURES
# ─────────────────────────────────────────
print("\n🔧 Merging PMU and SCADA Features...")
merged_df = pd.merge(
    pmu_features_df,
    scada_features_df,
    on=["bus_id", "window_id"],
    suffixes=("_pmu", "_scada")
)

# Keep one attack label
merged_df["attack_label"] = merged_df[["attack_label_pmu", "attack_label_scada"]].max(axis=1)
merged_df["attack_type"] = merged_df["attack_type_pmu"]
merged_df.drop(["attack_label_pmu", "attack_label_scada", "attack_type_scada"], axis=1, inplace=True)

print(f"✅ Merged Features Shape: {merged_df.shape}")

# ─────────────────────────────────────────
# 5. FINAL SUMMARY
# ─────────────────────────────────────────
print("\n📊 Final Dataset Summary:")
print(f"   Total Samples  : {len(merged_df)}")
print(f"   Total Features : {len(merged_df.columns)}")
print(f"   Normal Samples : {len(merged_df[merged_df['attack_label'] == 0])}")
print(f"   Attack Samples : {len(merged_df[merged_df['attack_label'] == 1])}")
print(f"\n   Attack Type Distribution:")
print(merged_df["attack_type"].value_counts())

# ─────────────────────────────────────────
# 6. SAVE FINAL OUTPUT
# ─────────────────────────────────────────
print("\n💾 Saving final output...")
merged_df.to_csv("data/processed/processed_signal_features.csv", index=False)
print("✅ Final file saved!")
print("   → processed_signal_features.csv")
print("\n🎉 feature_extraction.py COMPLETE!")
print("\n✅ YOUR TEAM DELIVERABLE IS READY!")
print("   📁 data/processed/processed_signal_features.csv")