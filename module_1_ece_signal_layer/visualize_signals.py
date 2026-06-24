import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("📂 Loading data for visualization...")
pmu_clean = pd.read_csv("data/processed/pmu_processed.csv")
pmu_attacked = pd.read_csv("data/processed/pmu_attacked.csv")
scada_attacked = pd.read_csv("data/processed/scada_attacked.csv")
features_df = pd.read_csv("data/processed/processed_signal_features.csv")
print("✅ Data Loaded!")

# Filter Bus 0 for clear visualization
bus0_clean = pmu_clean[pmu_clean["bus_id"] == 0].reset_index(drop=True)
bus0_attacked = pmu_attacked[pmu_attacked["bus_id"] == 0].reset_index(drop=True)

# ─────────────────────────────────────────
# 2. PLOT 1 - PMU SIGNALS CLEAN VS ATTACKED
# ─────────────────────────────────────────
print("\n📊 Generating Plot 1 - PMU Clean vs Attacked...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle("PMU Signals - Clean vs Attacked (Bus 0)", fontsize=16, fontweight="bold")

# Voltage
axes[0].plot(bus0_clean["voltage_pu"][:200], color="blue", label="Clean", linewidth=1)
axes[0].plot(bus0_attacked["voltage_pu"][:200], color="red", label="Attacked", linewidth=1, alpha=0.7)
axes[0].set_title("Voltage (p.u.)")
axes[0].set_ylabel("Voltage")
axes[0].legend()
axes[0].grid(True)

# Frequency
axes[1].plot(bus0_clean["frequency_hz"][:200], color="green", label="Clean", linewidth=1)
axes[1].plot(bus0_attacked["frequency_hz"][:200], color="orange", label="Attacked", linewidth=1, alpha=0.7)
axes[1].set_title("Frequency (Hz)")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].grid(True)

# Phase Angle
axes[2].plot(bus0_clean["phase_angle_deg"][:200], color="purple", label="Clean", linewidth=1)
axes[2].plot(bus0_attacked["phase_angle_deg"][:200], color="red", label="Attacked", linewidth=1, alpha=0.7)
axes[2].set_title("Phase Angle (degrees)")
axes[2].set_ylabel("Phase Angle")
axes[2].set_xlabel("Time Steps")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("results/plot1_pmu_clean_vs_attacked.png", dpi=150)
plt.show()
print("✅ Plot 1 Saved!")

# ─────────────────────────────────────────
# 3. PLOT 2 - ATTACK TYPE DISTRIBUTION
# ─────────────────────────────────────────
print("\n📊 Generating Plot 2 - Attack Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Attack Distribution in Dataset", fontsize=16, fontweight="bold")

# Pie chart
attack_counts = features_df["attack_type"].value_counts()
colors = ["#2ecc71", "#e74c3c", "#e67e22", "#3498db"]
axes[0].pie(attack_counts.values, labels=attack_counts.index,
            autopct="%1.1f%%", colors=colors, startangle=90)
axes[0].set_title("Attack Type Distribution")

# Bar chart
axes[1].bar(attack_counts.index, attack_counts.values, color=colors)
axes[1].set_title("Attack Sample Count")
axes[1].set_xlabel("Attack Type")
axes[1].set_ylabel("Number of Samples")
axes[1].grid(True, axis="y")
for i, v in enumerate(attack_counts.values):
    axes[1].text(i, v + 5, str(v), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("results/plot2_attack_distribution.png", dpi=150)
plt.show()
print("✅ Plot 2 Saved!")

# ─────────────────────────────────────────
# 4. PLOT 3 - FEATURE HEATMAP
# ─────────────────────────────────────────
print("\n📊 Generating Plot 3 - Feature Correlation Heatmap...")
numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = features_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(im)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=90, fontsize=8)
ax.set_yticklabels(numeric_cols, fontsize=8)
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("results/plot3_feature_heatmap.png", dpi=150)
plt.show()
print("✅ Plot 3 Saved!")

# ─────────────────────────────────────────
# 5. PLOT 4 - SCADA POWER READINGS
# ─────────────────────────────────────────
print("\n📊 Generating Plot 4 - SCADA Power Readings...")
scada_bus0 = scada_attacked[scada_attacked["bus_id"] == 0].reset_index(drop=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("SCADA Power Readings - Bus 0", fontsize=16, fontweight="bold")

axes[0].plot(scada_bus0["active_power_mw"][:200], color="blue", linewidth=1)
axes[0].set_title("Active Power (MW)")
axes[0].set_ylabel("Power (MW)")
axes[0].grid(True)

# Highlight attacked points
attacked_points = scada_bus0[scada_bus0["attack_label"] == 1][:200]
axes[0].scatter(attacked_points.index, attacked_points["active_power_mw"],
                color="red", s=10, label="Attack Points", zorder=5)
axes[0].legend()

axes[1].plot(scada_bus0["reactive_power_mvar"][:200], color="green", linewidth=1)
axes[1].set_title("Reactive Power (MVAR)")
axes[1].set_ylabel("Power (MVAR)")
axes[1].set_xlabel("Time Steps")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("results/plot4_scada_power.png", dpi=150)
plt.show()
print("✅ Plot 4 Saved!")

print("\n🎉 ALL PLOTS GENERATED & SAVED!")
print("   📁 results/plot1_pmu_clean_vs_attacked.png")
print("   📁 results/plot2_attack_distribution.png")
print("   📁 results/plot3_feature_heatmap.png")
print("   📁 results/plot4_scada_power.png")