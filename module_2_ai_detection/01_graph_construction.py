"""
Module 2 - Step 1: Graph-Based Smart Grid Modeling (IEEE-14 Bus)
Replaces random topology with authentic IEEE-14 bus system.

IEEE 14-Bus Standard:
  - 14 buses, 20 transmission lines
  - 1 Slack bus (bus 0 / IEEE bus 1)
  - 4 PV generator buses (1, 2, 5, 7 / IEEE buses 2, 3, 6, 8)
  - 9 PQ load buses (all others)
  - Realistic line impedances from IEEE standard data
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# 1. IEEE-14 BUS SYSTEM CONSTANTS
# ─────────────────────────────────────────────

# Standard IEEE-14 bus line data: (from_bus, to_bus, R_pu, X_pu, B_pu)
# Source: IEEE 14-bus test case (MATPOWER / standard reference)
IEEE14_LINES = [
    (0,  1,  0.01938, 0.05917, 0.0528),
    (0,  4,  0.05403, 0.22304, 0.0492),
    (1,  2,  0.04699, 0.19797, 0.0438),
    (1,  3,  0.05811, 0.17632, 0.0340),
    (1,  4,  0.05695, 0.17388, 0.0346),
    (2,  3,  0.06701, 0.17103, 0.0128),
    (3,  4,  0.01335, 0.04211, 0.0000),
    (3,  6,  0.00000, 0.20912, 0.0000),   # transformer
    (3,  8,  0.00000, 0.55618, 0.0000),   # transformer
    (4,  5,  0.00000, 0.25202, 0.0000),   # transformer
    (5, 10,  0.09498, 0.19890, 0.0000),
    (5, 11,  0.12291, 0.25581, 0.0000),
    (5, 12,  0.06615, 0.13027, 0.0000),
    (6,  7,  0.00000, 0.17615, 0.0000),   # transformer
    (6,  8,  0.00000, 0.11001, 0.0000),   # transformer
    (8,  9,  0.03181, 0.08450, 0.0000),
    (8, 13,  0.12711, 0.27038, 0.0000),
    (9, 10,  0.08205, 0.19207, 0.0000),
    (11, 12, 0.22092, 0.19988, 0.0000),
    (12, 13, 0.17093, 0.34802, 0.0000),
]

# Bus type: 0=PQ (load), 1=PV (generator), 2=Slack
IEEE14_BUS_TYPES = [2,1,1,0,0,1,0,1,0,0,0,0,0,0]

# Voltage set-points per bus (p.u.) — from IEEE reference solution
IEEE14_V_SETPOINT = {
    0: 1.060, 1: 1.045, 2: 1.010, 3: 1.018,
    4: 1.020, 5: 1.070, 6: 1.062, 7: 1.090,
    8: 1.056, 9: 1.051, 10: 1.057, 11: 1.055,
    12: 1.050, 13: 1.036
}

# Nominal active power injection (MW, p.u.) per bus — generation (+) / load (-)
IEEE14_P_NOMINAL = [2.324, 0.400, 0.000,-0.478,-0.076, 0.000,-0.295, 0.000,
                    -0.295,-0.090,-0.035,-0.061,-0.135,-0.149]

# Nominal reactive power (MVAR)
IEEE14_Q_NOMINAL = [-0.169, 0.424, 0.234,-0.039,-0.016, 0.122,-0.166, 0.176,
                    -0.058,-0.058,-0.018,-0.016,-0.058,-0.050]

NUM_BUSES = 14
NUM_LINES = len(IEEE14_LINES)


# ─────────────────────────────────────────────
# 2. BUILD STATIC GRAPH STRUCTURE
# ─────────────────────────────────────────────

def build_ieee14_graph():
    """
    Returns:
      edge_index : [2, 2*NUM_LINES] bidirectional
      edge_attr  : [2*NUM_LINES, 3]  — (R, X, B) per edge
    """
    edges_fwd, edges_bwd, attrs = [], [], []

    for (u, v, R, X, B) in IEEE14_LINES:
        edges_fwd.append([u, v])
        edges_bwd.append([v, u])
        attrs.append([R, X, B])

    # Bidirectional (undirected graph)
    all_edges = edges_fwd + edges_bwd
    all_attrs = attrs + attrs  # same physical parameters both directions

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(all_attrs, dtype=torch.float32)

    return edge_index, edge_attr


# ─────────────────────────────────────────────
# 3. GENERATE REALISTIC TIME-SERIES NODE FEATURES
# ─────────────────────────────────────────────

def generate_node_features(time_steps=200, seed=42):
    """
    Simulate realistic IEEE-14 bus dynamics over T timesteps.

    Each node has 6 features per timestep:
      [V_mag, V_angle, P_inject, Q_inject, freq_dev, node_type]

    Realism improvements over random baseline:
      - Bus-specific voltage set-points with smooth temporal variation
      - Load profile: sinusoidal daily cycle + Gaussian noise
      - Correlated frequency deviations across buses
      - Voltage magnitudes follow their IEEE set-points ± small noise
      - V_angle reflects approximate load flow relationships
    """
    np.random.seed(seed)
    features = np.zeros((time_steps, NUM_BUSES, 6))

    # Time index (normalized 0→1 represents one "day")
    t_vec = np.linspace(0, 2 * np.pi, time_steps)

    # Shared load factor: daily sinusoidal profile + noise
    load_profile = 0.8 + 0.2 * np.sin(t_vec - np.pi / 2)  # peaks at midday

    # Shared frequency deviation (system-wide, correlated)
    freq_base = np.random.normal(0, 0.005, time_steps)
    freq_base = np.convolve(freq_base, np.ones(5)/5, mode='same')  # smooth

    for b in range(NUM_BUSES):
        v0   = IEEE14_V_SETPOINT[b]
        p0   = IEEE14_P_NOMINAL[b]
        q0   = IEEE14_Q_NOMINAL[b]
        btype = IEEE14_BUS_TYPES[b]

        for t in range(time_steps):
            lf = load_profile[t]

            # Voltage magnitude: set-point + small Gaussian noise
            noise_v = np.random.normal(0, 0.008)
            features[t, b, 0] = np.clip(v0 + noise_v, 0.90, 1.10)

            # Voltage angle: approximate, scaled by load; slack bus = 0
            if btype == 2:  # Slack
                features[t, b, 1] = 0.0
            else:
                angle_base = -0.1 * (b / NUM_BUSES)  # rough angle gradient
                features[t, b, 1] = angle_base + np.random.normal(0, 0.02)

            # Active power: load buses scale with load_profile, gen buses = constant
            if p0 < 0:  # load bus
                features[t, b, 2] = p0 * lf + np.random.normal(0, 0.01)
            else:       # generator bus
                features[t, b, 2] = p0 + np.random.normal(0, 0.015)

            # Reactive power: proportional variation
            if q0 < 0:
                features[t, b, 3] = q0 * lf + np.random.normal(0, 0.005)
            else:
                features[t, b, 3] = q0 + np.random.normal(0, 0.008)

            # Frequency deviation: shared system signal + per-bus noise
            features[t, b, 4] = freq_base[t] + np.random.normal(0, 0.002)

            # Node type (static)
            features[t, b, 5] = float(btype)

    return features


# ─────────────────────────────────────────────
# 4. INJECT ATTACKS (labeled anomalies)
# ─────────────────────────────────────────────

def inject_attacks(features, attack_ratio=0.15, seed=42):
    """
    Inject three attack types with metadata tracking.

    Returns:
      features_attacked : modified feature array
      labels            : [T, N] binary node labels
      attack_metadata   : list of dicts with attack details
    """
    np.random.seed(seed + 1)
    T, N, F = features.shape
    labels   = np.zeros((T, N), dtype=int)
    modified = features.copy()
    metadata = []

    attack_times = np.random.choice(T, int(T * attack_ratio), replace=False)

    for t in sorted(attack_times):
        atype = np.random.choice(['stealth_fdi', 'coordinated', 'cascading'],
                                 p=[0.4, 0.35, 0.25])

        if atype == 'stealth_fdi':
            # Small bias on 1 bus — stays within normal-looking range
            target = np.random.randint(0, N)
            bias_v  = np.random.uniform(0.04, 0.07)
            bias_p  = np.random.uniform(0.08, 0.18)
            modified[t, target, 0] += bias_v
            modified[t, target, 2] += bias_p
            labels[t, target] = 1
            metadata.append({'t': int(t), 'type': atype,
                              'nodes': [int(target)],
                              'bias_v': round(bias_v, 4),
                              'bias_p': round(bias_p, 4)})

        elif atype == 'coordinated':
            # 3–4 buses attacked simultaneously; correlated voltage/angle spikes
            k       = np.random.randint(3, 5)
            targets = np.random.choice(N, size=k, replace=False).tolist()
            for tgt in targets:
                modified[t, tgt, 0] += np.random.uniform(0.08, 0.18)   # V_mag
                modified[t, tgt, 1] += np.random.uniform(0.10, 0.25)   # V_angle
                modified[t, tgt, 3] += np.random.uniform(0.15, 0.35)   # Q
                labels[t, tgt] = 1
            metadata.append({'t': int(t), 'type': atype, 'nodes': [int(x) for x in targets]})

        elif atype == 'cascading':
            # Anomaly propagates from origin node across timesteps t, t+1, t+2
            origin  = np.random.randint(0, N)
            horizon = min(3, T - t)
            cascade_nodes = []
            for offset in range(horizon):
                # Each step affects a different (possibly connected) node
                node  = np.random.randint(0, N)
                scale = 0.25 * (offset + 1)
                modified[t + offset, node, 0] += scale        # V_mag drift
                modified[t + offset, node, 4] += scale * 0.4  # freq drift
                labels[t + offset, node] = 1
                cascade_nodes.append({'t': int(t + offset), 'node': int(node)})
            metadata.append({'t': int(t), 'type': atype,
                              'origin': int(origin), 'cascade': cascade_nodes})

    print(f"  Injected {len(metadata)} attack events")
    print(f"  Attack nodes: {labels.sum()} / {labels.size} "
          f"({labels.sum() / labels.size:.1%})")

    return modified, labels, metadata


# ─────────────────────────────────────────────
# 5. BUILD PyG GRAPH OBJECTS
# ─────────────────────────────────────────────

def build_pyg_graphs(features, labels, edge_index, edge_attr):
    """One PyG Data object per timestep."""
    return [
        Data(
            x          = torch.tensor(features[t], dtype=torch.float32),
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor(labels[t], dtype=torch.long)
        )
        for t in range(features.shape[0])
    ]


# ─────────────────────────────────────────────
# 6. VISUALIZE IEEE-14 TOPOLOGY
# ─────────────────────────────────────────────

# Hand-tuned positions that resemble standard IEEE-14 bus diagrams
IEEE14_POSITIONS = {
    0:  (0.0,  2.0),
    1:  (1.0,  2.0),
    2:  (2.5,  2.5),
    3:  (2.0,  1.5),
    4:  (0.8,  1.0),
    5:  (3.5,  2.0),
    6:  (3.5,  1.0),
    7:  (4.5,  2.5),
    8:  (4.5,  1.5),
    9:  (3.0,  0.5),
    10: (2.5,  0.0),
    11: (2.0,  0.5),
    12: (1.5,  0.0),
    13: (1.0,  0.5),
}

def visualize_grid(edge_index, labels_t, num_buses=14, save_path="grid_topology_ieee14.png"):
    G = nx.Graph()
    G.add_nodes_from(range(num_buses))
    G.add_edges_from(edge_index.t().tolist())

    pos = IEEE14_POSITIONS
    bus_type_color = {2: '#00d4ff', 1: '#00ff99', 0: '#ff4d6d'}
    colors = [
        '#ff4d6d' if labels_t[i] == 1
        else ('#00ff99' if IEEE14_BUS_TYPES[i] == 1
              else ('#FFD700' if IEEE14_BUS_TYPES[i] == 2
                    else '#00d4ff'))
        for i in range(num_buses)
    ]

    plt.figure(figsize=(12, 8), facecolor='#04080f')
    ax = plt.gca()
    ax.set_facecolor('#04080f')

    nx.draw_networkx_edges(G, pos, edge_color='#1a3a5c', width=1.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=9,
                            font_weight='bold', ax=ax)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color='#FFD700', label='Slack bus'),
        Patch(color='#00ff99', label='PV (Generator)'),
        Patch(color='#00d4ff', label='PQ (Load)'),
        Patch(color='#ff4d6d', label='Attack node'),
    ]
    ax.legend(handles=legend, loc='lower right',
              facecolor='#0a1628', labelcolor='white', fontsize=9)

    plt.title("IEEE-14 Bus Topology — Smart Grid Security Framework",
              color='white', fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#04080f')
    plt.close()
    print(f"  Graph saved to {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json, os
    os.makedirs("data", exist_ok=True)

    TIME_STEPS = 200

    print("=== Step 1: IEEE-14 bus graph structure ===")
    edge_index, edge_attr = build_ieee14_graph()
    print(f"  Buses: {NUM_BUSES} | Lines: {NUM_LINES} (bidirectional edges: {edge_index.shape[1]})")

    print("\n=== Step 2: Generating realistic time-series features ===")
    features = generate_node_features(time_steps=TIME_STEPS)
    print(f"  Shape: {features.shape}  [T x N x F]")

    print("\n=== Step 3: Injecting attacks ===")
    features_attacked, labels, metadata = inject_attacks(features, attack_ratio=0.15)

    print("\n=== Step 4: Building PyG graphs ===")
    graphs = build_pyg_graphs(features_attacked, labels, edge_index, edge_attr)
    print(f"  Graphs: {len(graphs)} | Sample: {graphs[0]}")

    print("\n=== Step 5: Visualizing IEEE-14 topology ===")
    # Find a timestep with an attack node to highlight in viz
    attack_t = next((t for t in range(TIME_STEPS) if labels[t].any()), 0)
    visualize_grid(edge_index, labels[attack_t], save_path="data/grid_topology_ieee14.png")

    # ── Save outputs ──
    torch.save(graphs, "data/pyg_graphs.pt")
    with open("data/attack_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Summary ===")
    print(f"  pyg_graphs.pt       → data/pyg_graphs.pt")
    print(f"  attack_metadata.json → data/attack_metadata.json")
    print(f"  topology image      → data/grid_topology_ieee14.png")
    print(f"  Unique attack events: {len(metadata)}")

    # Attack type breakdown
    from collections import Counter
    breakdown = Counter(m['type'] for m in metadata)
    for atype, cnt in breakdown.items():
        print(f"    {atype}: {cnt}")