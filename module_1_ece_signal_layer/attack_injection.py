import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# 1. LOAD PROCESSED DATA
# ─────────────────────────────────────────
print("📂 Loading processed data...")
pmu_df = pd.read_csv("data/processed/pmu_processed.csv")
scada_df = pd.read_csv("data/processed/scada_processed.csv")
print(f"✅ PMU Data Loaded! Shape: {pmu_df.shape}")
print(f"✅ SCADA Data Loaded! Shape: {scada_df.shape}")

# ─────────────────────────────────────────
# 2. ADD ATTACK LABEL COLUMN (0 = Normal)
# ─────────────────────────────────────────
pmu_df["attack_label"] = 0
pmu_df["attack_type"] = "normal"
scada_df["attack_label"] = 0
scada_df["attack_type"] = "normal"

# ─────────────────────────────────────────
# 3. FALSE DATA INJECTION (FDI)
# ─────────────────────────────────────────
print("\n⚠️ Injecting False Data Injection (FDI) attacks...")
fdi_indices = np.random.choice(len(pmu_df), size=int(len(pmu_df) * 0.10), replace=False)
pmu_df.loc[fdi_indices, "voltage_pu"] += np.random.uniform(0.2, 0.5, size=len(fdi_indices))
pmu_df.loc[fdi_indices, "attack_label"] = 1
pmu_df.loc[fdi_indices, "attack_type"] = "FDI"
print(f"✅ FDI Attack Injected on {len(fdi_indices)} samples!")

# ─────────────────────────────────────────
# 4. SENSOR SPOOFING
# ─────────────────────────────────────────
print("\n⚠️ Injecting Sensor Spoofing attacks...")
spoof_indices = np.random.choice(len(pmu_df), size=int(len(pmu_df) * 0.08), replace=False)
pmu_df.loc[spoof_indices, "frequency_hz"] = 0.99
pmu_df.loc[spoof_indices, "phase_angle_deg"] = 0.01
pmu_df.loc[spoof_indices, "attack_label"] = 1
pmu_df.loc[spoof_indices, "attack_type"] = "spoofing"
print(f"✅ Sensor Spoofing Injected on {len(spoof_indices)} samples!")

# ─────────────────────────────────────────
# 5. MEASUREMENT MANIPULATION
# ─────────────────────────────────────────
print("\n⚠️ Injecting Measurement Manipulation attacks...")
manip_indices = np.random.choice(len(scada_df), size=int(len(scada_df) * 0.08), replace=False)
scada_df.loc[manip_indices, "active_power_mw"] *= np.random.uniform(1.5, 2.0, size=len(manip_indices))
scada_df.loc[manip_indices, "reactive_power_mvar"] *= np.random.uniform(1.5, 2.0, size=len(manip_indices))
scada_df.loc[manip_indices, "attack_label"] = 1
scada_df.loc[manip_indices, "attack_type"] = "manipulation"
print(f"✅ Measurement Manipulation Injected on {len(manip_indices)} samples!")

# ─────────────────────────────────────────
# 6. ATTACK SUMMARY
# ─────────────────────────────────────────
print("\n📊 Attack Summary:")
print(f"   Total PMU Samples     : {len(pmu_df)}")
print(f"   Normal PMU Samples    : {len(pmu_df[pmu_df['attack_label'] == 0])}")
print(f"   Attacked PMU Samples  : {len(pmu_df[pmu_df['attack_label'] == 1])}")
print(f"   Total SCADA Samples   : {len(scada_df)}")
print(f"   Normal SCADA Samples  : {len(scada_df[scada_df['attack_label'] == 0])}")
print(f"   Attacked SCADA Samples: {len(scada_df[scada_df['attack_label'] == 1])}")

# ─────────────────────────────────────────
# 7. SAVE ATTACK INJECTED DATA
# ─────────────────────────────────────────
print("\n💾 Saving attack injected data...")
pmu_df.to_csv("data/processed/pmu_attacked.csv", index=False)
scada_df.to_csv("data/processed/scada_attacked.csv", index=False)

print("✅ Attack injected files saved!")
print("   → pmu_attacked.csv")
print("   → scada_attacked.csv")
print("\n🎉 attack_injection.py COMPLETE!")