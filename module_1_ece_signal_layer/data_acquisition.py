import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# 1. LOAD IEEE 14-BUS SYSTEM
# ─────────────────────────────────────────
net = pn.case14()
print("✅ IEEE 14-Bus System Loaded Successfully!")
print(f"   Buses: {len(net.bus)} | Lines: {len(net.line)}")

# ─────────────────────────────────────────
# 2. SETTINGS
# ─────────────────────────────────────────
NUM_TIMESTEPS = 1000       # 1000 time readings
NUM_BUSES = len(net.bus)   # 14 buses
FREQUENCY = 50.0           # 50Hz (India standard)
np.random.seed(42)

# ─────────────────────────────────────────
# 3. GENERATE PMU SIGNALS
# ─────────────────────────────────────────
timestamps = pd.date_range(start="2024-01-01", periods=NUM_TIMESTEPS, freq="100ms")

pmu_data = []
for t in range(NUM_TIMESTEPS):
    for bus in range(NUM_BUSES):
        pmu_data.append({
            "timestamp": timestamps[t],
            "bus_id": bus,
            "voltage_pu": 1.0 + np.random.normal(0, 0.01),        # Voltage in per unit
            "phase_angle_deg": np.random.normal(0, 5),             # Phase angle in degrees
            "frequency_hz": FREQUENCY + np.random.normal(0, 0.02) # Frequency near 50Hz
        })

pmu_df = pd.DataFrame(pmu_data)
print(f"✅ PMU Data Generated! Shape: {pmu_df.shape}")

# ─────────────────────────────────────────
# 4. GENERATE SCADA SENSOR DATA
# ─────────────────────────────────────────
scada_data = []
for t in range(NUM_TIMESTEPS):
    for bus in range(NUM_BUSES):
        scada_data.append({
            "timestamp": timestamps[t],
            "bus_id": bus,
            "active_power_mw": np.random.uniform(10, 300),    # Active power in MW
            "reactive_power_mvar": np.random.uniform(5, 100), # Reactive power in MVAR
            "current_ka": np.random.uniform(0.1, 1.5)         # Current in kA
        })

scada_df = pd.DataFrame(scada_data)
print(f"✅ SCADA Data Generated! Shape: {scada_df.shape}")

# ─────────────────────────────────────────
# 5. GENERATE SMART METER DATA
# ─────────────────────────────────────────
smart_meter_data = []
for t in range(NUM_TIMESTEPS):
    for bus in range(NUM_BUSES):
        smart_meter_data.append({
            "timestamp": timestamps[t],
            "bus_id": bus,
            "energy_consumption_kwh": np.random.uniform(50, 500), # Energy in kWh
            "power_factor": np.random.uniform(0.85, 1.0)          # Power factor
        })

smart_meter_df = pd.DataFrame(smart_meter_data)
print(f"✅ Smart Meter Data Generated! Shape: {smart_meter_df.shape}")

# ─────────────────────────────────────────
# 6. SAVE TO CSV FILES
# ─────────────────────────────────────────
pmu_df.to_csv("data/raw/pmu_data.csv", index=False)
scada_df.to_csv("data/raw/scada_data.csv", index=False)
smart_meter_df.to_csv("data/raw/smart_meter_data.csv", index=False)

print("\n✅ All files saved to data/raw/")
print("   → pmu_data.csv")
print("   → scada_data.csv")
print("   → smart_meter_data.csv")
print("\n🎉 data_acquisition.py COMPLETE!")