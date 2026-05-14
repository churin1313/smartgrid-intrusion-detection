"""
=============================================================================
MODULE 2: Spatio-Temporal Deep Learning Detection Engine
=============================================================================
Responsibilities:
  1. Graph-Based Smart Grid Modeling  – bus topology → PyG graph
  2. Deep Learning Model Development  – GNN + LSTM + Transformer
  3. Attack Detection Scope           – stealth, coordinated, cascading

Author  : Student 2 – Shashwat (CSE AI & ML)
Project : Smart Grid Cyber-Physical Attack Detection
=============================================================================
"""

import numpy as np
import pandas as pd
import warnings, os, time
warnings.filterwarnings("ignore")

# ── PyTorch core ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ── PyTorch Geometric (GNN) ─────────────────────────────────────────────────
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn   import GCNConv, GATConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARN] torch_geometric not found – GNN layers replaced by MLP fallback.")

# ── Scikit-learn (baseline) ─────────────────────────────────────────────────
from sklearn.ensemble        import IsolationForest, RandomForestClassifier
from sklearn.metrics         import (classification_report, roc_auc_score,
                                     confusion_matrix, f1_score)
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"[Module 2] Device: {DEVICE}")
print(f"[Module 2] PyTorch Geometric available: {HAS_PYG}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  IEEE 14-BUS GRAPH TOPOLOGY
# ─────────────────────────────────────────────────────────────────────────────

# Standard IEEE 14-bus transmission line connectivity (undirected edges)
IEEE14_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,0),          # ring backbone
    (0,5),(5,10),(10,11),(11,12),(12,13),   # branches
    (1,4),(2,5),(3,8),(4,9),(6,7),
    (7,8),(8,9),(9,13),(5,6),(6,11),
]

def build_edge_index(edges, n_nodes: int) -> torch.Tensor:
    """
    Build a COO edge_index tensor (2 × E) for an undirected graph.
    Each undirected edge is stored as two directed edges.
    """
    src, dst = [], []
    for u, v in edges:
        if u < n_nodes and v < n_nodes:
            src += [u, v]
            dst += [v, u]
    return torch.tensor([src, dst], dtype=torch.long)


class GridGraphBuilder:
    """
    Converts processed bus-level features into PyG graph objects.

    Each graph snapshot represents one time step (or window) with:
        node features  : [voltage, angle, frequency, P, Q, I]  (6-dim)
        edge_index     : IEEE 14-bus topology
        y              : attack label (0 / 1)
    """

    def __init__(self, n_buses: int = 14,
                 edges=IEEE14_EDGES,
                 node_feat_dim: int = 3):
        self.n_buses       = n_buses
        self.edge_index    = build_edge_index(edges, n_buses)
        self.node_feat_dim = node_feat_dim

    def dataframe_to_graphs(self, pmu_df: pd.DataFrame,
                            feature_cols: list,
                            label_col: str = "label") -> list:
        """
        One PyG Data object per unique timestamp.
        pmu_df must have columns: timestamp, bus_id, <features>, label
        """
        graphs = []
        for ts, grp in pmu_df.groupby("timestamp"):
            grp = grp.sort_values("bus_id").reset_index(drop=True)

            # pad / truncate to n_buses
            n = len(grp)
            x = np.zeros((self.n_buses, len(feature_cols)), dtype=np.float32)
            x[:min(n, self.n_buses)] = grp[feature_cols].values[:self.n_buses]

            y = int(grp[label_col].max()) if label_col in grp.columns else 0

            data = Data(
                x          = torch.tensor(x, dtype=torch.float),
                edge_index = self.edge_index.clone(),
                y          = torch.tensor([y], dtype=torch.long),
            )
            graphs.append(data)
        return graphs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Wraps numpy (X, y) windows for LSTM / Transformer training.
    X shape : (N, T, F)
    y shape : (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X: np.ndarray, y: np.ndarray,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 batch_size: int = 64,
                 ) -> tuple:
    ds   = WindowDataset(X, y)
    N    = len(ds)
    n_te = int(N * test_ratio)
    n_va = int(N * val_ratio)
    n_tr = N - n_va - n_te
    tr, va, te = random_split(ds, [n_tr, n_va, n_te],
                              generator=torch.Generator().manual_seed(SEED))
    loader = lambda s, shuf: DataLoader(s, batch_size=batch_size, shuffle=shuf)
    return loader(tr, True), loader(va, False), loader(te, False)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEEP LEARNING MODELS
# ─────────────────────────────────────────────────────────────────────────────

# ── 3-A  GNN Encoder (spatial) ───────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    2-layer Graph Attention Network encoder.
    Falls back to a plain MLP if PyG is unavailable.

    Input : (batch_size × n_nodes, node_feat_dim)
    Output: (batch_size, gnn_hidden)
    """

    def __init__(self, node_feat_dim: int = 3,
                 gnn_hidden: int = 64,
                 n_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.use_pyg = HAS_PYG

        if self.use_pyg:
            self.conv1 = GATConv(node_feat_dim, gnn_hidden,
                                 heads=n_heads, dropout=dropout)
            self.conv2 = GATConv(gnn_hidden * n_heads, gnn_hidden,
                                 heads=1,     dropout=dropout)
        else:
            # MLP fallback
            self.mlp = nn.Sequential(
                nn.Linear(node_feat_dim, gnn_hidden),
                nn.ReLU(),
                nn.Linear(gnn_hidden, gnn_hidden),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None, batch=None):
        if self.use_pyg and edge_index is not None:
            x = F.elu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = self.conv2(x, edge_index)
            # global pool → (batch_size, gnn_hidden)
            x = global_mean_pool(x, batch)
        else:
            # flatten bus dim → mean pool
            x = self.mlp(x)
            if x.dim() == 3:           # (B, n_nodes, H)
                x = x.mean(dim=1)
        return x


# ── 3-B  LSTM Temporal Encoder ───────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM that encodes a time window.
    Input : (batch, T, feature_dim)
    Output: (batch, lstm_hidden * 2)
    """

    def __init__(self, input_dim: int = 3,
                 hidden_dim: int = 128,
                 n_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers    = n_layers,
                            batch_first   = True,
                            bidirectional = True,
                            dropout       = dropout if n_layers > 1 else 0)
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        # x: (B, T, F)
        out, (h, _) = self.lstm(x)
        # concat last hidden from both directions
        h_fwd = h[-2]   # (B, H)
        h_bwd = h[-1]
        return torch.cat([h_fwd, h_bwd], dim=-1)   # (B, 2H)


# ── 3-C  Transformer Temporal Encoder ────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Positional-encoded Transformer encoder for temporal attack detection.
    Input : (batch, T, feature_dim)
    Output: (batch, d_model)
    """

    def __init__(self, feature_dim: int = 3,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_emb    = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x):
        # x: (B, T, F)
        B, T, _ = x.shape
        pos      = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x        = self.input_proj(x) + self.pos_emb(pos)         # (B, T, D)
        out      = self.encoder(x)                                 # (B, T, D)
        return out.mean(dim=1)                                     # (B, D)


# ── 3-D  Spatio-Temporal Fusion (GNN + LSTM/Transformer) ─────────────────────

class SpatioTemporalDetector(nn.Module):
    """
    Full model fusing spatial (GNN) and temporal (LSTM or Transformer) streams.

    Architecture
    ────────────
      Temporal stream : LSTM or Transformer on raw windows (B, T, F)
      Spatial  stream : GNN on graph snapshots (optional, requires PyG)
      Fusion          : concat → MLP → anomaly score / binary label

    Parameters
    ----------
    feature_dim   : number of node/signal features
    n_classes     : 2 (normal vs attack) or more for multi-class
    temporal_type : "lstm" | "transformer"
    use_gnn       : True only when graph batches are available
    """

    def __init__(self,
                 feature_dim   : int  = 3,
                 n_classes     : int  = 2,
                 temporal_type : str  = "lstm",
                 use_gnn       : bool = False,
                 lstm_hidden   : int  = 128,
                 transformer_d : int  = 64,
                 gnn_hidden    : int  = 64,
                 dropout       : float = 0.3):
        super().__init__()
        self.use_gnn       = use_gnn and HAS_PYG
        self.temporal_type = temporal_type

        # ── temporal encoder ──────────────────────────────────────────────
        if temporal_type == "lstm":
            self.temporal_enc = LSTMEncoder(feature_dim, lstm_hidden,
                                            dropout=dropout)
            temp_dim = lstm_hidden * 2
        else:
            self.temporal_enc = TransformerEncoder(feature_dim, transformer_d,
                                                   dropout=dropout)
            temp_dim = transformer_d

        # ── spatial encoder ───────────────────────────────────────────────
        if self.use_gnn:
            self.gnn_enc = GNNEncoder(feature_dim, gnn_hidden, dropout=dropout)
            fusion_dim   = temp_dim + gnn_hidden
        else:
            fusion_dim = temp_dim

        # ── fusion classifier ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

        # ── anomaly score head (regression: 0→normal, 1→attack) ───────────
        self.anomaly_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def encode(self, x_seq, graph_batch=None):
        # temporal
        z_t = self.temporal_enc(x_seq)                        # (B, temp_dim)

        # spatial (optional)
        if self.use_gnn and graph_batch is not None:
            z_s = self.gnn_enc(graph_batch.x,
                               graph_batch.edge_index,
                               graph_batch.batch)             # (B, gnn_hidden)
            z   = torch.cat([z_t, z_s], dim=-1)
        else:
            z = z_t

        return z

    def forward(self, x_seq, graph_batch=None):
        z      = self.encode(x_seq, graph_batch)
        logits = self.classifier(z)                          # (B, n_classes)
        score  = self.anomaly_head(z).squeeze(-1)            # (B,)
        return logits, score


# ── 3-E  Autoencoder Anomaly Detector (unsupervised baseline) ────────────────

class LSTMAutoencoder(nn.Module):
    """
    Reconstruction-based anomaly detector.
    High reconstruction error → anomaly.
    Input / Output : (batch, T, F)
    """

    def __init__(self, feature_dim: int = 3,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(feature_dim, hidden_dim,
                               num_layers=n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim,
                               num_layers=n_layers, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)
        self.out_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        # encode
        _, (h, c) = self.encoder(x)
        # repeat last hidden as decoder seed
        B, T, _   = x.shape
        dec_in    = h[-1].unsqueeze(1).repeat(1, T, 1)   # (B, T, H)
        dec_out, _ = self.decoder(dec_in, (h, c))
        recon      = self.out_proj(dec_out)               # (B, T, F)
        return recon

    def anomaly_score(self, x):
        recon = self.forward(x)
        score = ((x - recon) ** 2).mean(dim=[1, 2])      # (B,)
        return score


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAINING & EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Generic trainer for SpatioTemporalDetector and LSTMAutoencoder.

    Parameters
    ----------
    model       : nn.Module
    lr          : learning rate
    weight_decay: L2 regularisation
    patience    : early stopping patience (epochs)
    mode        : "supervised" | "autoencoder"
    """

    def __init__(self, model: nn.Module,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 patience: int = 10,
                 mode: str = "supervised",
                 class_weights: torch.Tensor = None):
        self.model  = model.to(DEVICE)
        self.optim  = torch.optim.AdamW(model.parameters(),
                                        lr=lr, weight_decay=weight_decay)
        self.sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                          self.optim, patience=patience // 2, factor=0.5)
        self.patience = patience
        self.mode     = mode

        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights.to(DEVICE))
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        self.mse_loss = nn.MSELoss()
        self.history  = {"train_loss": [], "val_loss": [],
                         "val_f1": [], "val_auc": []}

    # ── single epoch ───────────────────────────────────────────────────────
    def _train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        for batch in loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.optim.zero_grad()

            if self.mode == "autoencoder":
                recon = self.model(x)
                loss  = self.mse_loss(recon, x)
            else:
                logits, score = self.model(x)
                loss = self.ce_loss(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader) -> dict:
        self.model.eval()
        total   = 0.0
        all_y   = []
        all_pred= []
        all_prob= []

        for batch in loader:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)

            if self.mode == "autoencoder":
                recon = self.model(x)
                loss  = self.mse_loss(recon, x)
                score = ((x - recon) ** 2).mean(dim=[1, 2])
                pred  = (score > score.mean()).long()
                prob  = score.cpu().numpy()
            else:
                logits, score = self.model(x)
                loss   = self.ce_loss(logits, y)
                pred   = logits.argmax(dim=-1)
                prob   = score.cpu().numpy()

            total    += loss.item()
            all_y    += y.cpu().tolist()
            all_pred += pred.cpu().tolist()
            all_prob += prob.tolist()

        f1  = f1_score(all_y, all_pred, zero_division=0)
        try:
            auc = roc_auc_score(all_y, all_prob)
        except Exception:
            auc = float("nan")

        return {"loss": total / len(loader), "f1": f1, "auc": auc,
                "y_true": all_y, "y_pred": all_pred}

    # ── full training loop ─────────────────────────────────────────────────
    def fit(self, train_loader, val_loader, epochs: int = 50) -> dict:
        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        print(f"\n[Trainer] mode={self.mode}  epochs={epochs}  "
              f"device={DEVICE}")
        print("-" * 60)

        for ep in range(1, epochs + 1):
            t0         = time.time()
            train_loss = self._train_epoch(train_loader)
            val_res    = self._eval_epoch(val_loader)
            val_loss   = val_res["loss"]

            self.sched.step(val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_res["f1"])
            self.history["val_auc"].append(val_res["auc"])

            elapsed = time.time() - t0
            print(f"  Epoch {ep:3d}/{epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"F1={val_res['f1']:.3f}  AUC={val_res['auc']:.3f}  "
                  f"[{elapsed:.1f}s]")

            # early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone()
                                 for k, v in self.model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {ep}.")
                    break

        # restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    # ── evaluation on test set ─────────────────────────────────────────────
    def evaluate(self, test_loader) -> dict:
        res = self._eval_epoch(test_loader)
        print("\n" + "=" * 60)
        print("  TEST SET EVALUATION")
        print("=" * 60)
        print(classification_report(res["y_true"], res["y_pred"],
                                    target_names=["Normal", "Attack"],
                                    zero_division=0))
        cm = confusion_matrix(res["y_true"], res["y_pred"])
        print(f"  Confusion Matrix:\n{cm}")
        print(f"  ROC-AUC : {res['auc']:.4f}")
        print("=" * 60)
        return res


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ANOMALY SCORE FUSION LAYER  (output to Module 3 / dashboard)
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyScoreFusion:
    """
    Aggregates anomaly scores from multiple models into a single
    fused score per window, for output to a decision / alert layer.

    Fusion strategies: mean | max | weighted
    """

    def __init__(self, strategy: str = "mean",
                 weights: list = None):
        assert strategy in ("mean", "max", "weighted")
        self.strategy = strategy
        self.weights  = weights   # only used for "weighted"

    def fuse(self, score_dict: dict) -> np.ndarray:
        """
        score_dict : {model_name: np.ndarray shape (N,)}
        Returns    : fused_score  shape (N,)
        """
        names  = list(score_dict.keys())
        scores = np.stack([score_dict[n] for n in names], axis=0)  # (M, N)

        if self.strategy == "max":
            return scores.max(axis=0)
        elif self.strategy == "weighted":
            w = np.array(self.weights or [1] * len(names), dtype=float)
            w /= w.sum()
            return (scores * w[:, None]).sum(axis=0)
        else:  # mean
            return scores.mean(axis=0)

    @staticmethod
    def threshold_alerts(fused_score: np.ndarray,
                         threshold: float = 0.5) -> np.ndarray:
        return (fused_score >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BASELINE MODELS (sklearn)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineDetectors:
    """
    Scikit-learn baseline for comparison:
        • IsolationForest  (unsupervised)
        • RandomForest     (supervised)
    """

    def __init__(self):
        self.iso_forest = IsolationForest(n_estimators=200,
                                          contamination=0.1,
                                          random_state=SEED)
        self.rf         = RandomForestClassifier(n_estimators=200,
                                                 class_weight="balanced",
                                                 random_state=SEED)

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """(N, T, F) → (N, T*F)"""
        return X.reshape(X.shape[0], -1)

    def fit_unsupervised(self, X_train: np.ndarray):
        print("[Baseline] Fitting IsolationForest …")
        self.iso_forest.fit(self._flatten(X_train))

    def predict_unsupervised(self, X: np.ndarray) -> np.ndarray:
        raw   = self.iso_forest.score_samples(self._flatten(X))
        score = 1 - (raw - raw.min()) / (raw.ptp() + 1e-9)   # normalise
        return score

    def fit_supervised(self, X_train: np.ndarray, y_train: np.ndarray):
        print("[Baseline] Fitting RandomForest …")
        self.rf.fit(self._flatten(X_train), y_train)

    def predict_supervised(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(self._flatten(X))[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        Xf = self._flatten(X_test)
        print("\n[Baseline] IsolationForest:")
        if_pred = (self.iso_forest.predict(Xf) == -1).astype(int)
        print(classification_report(y_test, if_pred,
                                    target_names=["Normal", "Attack"],
                                    zero_division=0))

        print("[Baseline] RandomForest:")
        rf_pred = self.rf.predict(Xf)
        print(classification_report(y_test, rf_pred,
                                    target_names=["Normal", "Attack"],
                                    zero_division=0))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FULL MODULE-2 ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class DetectionEngine:
    """
    Ties together all Module-2 components.

    Usage
    -----
    engine = DetectionEngine()
    results = engine.run(datasets)   # datasets from Module 1 pipeline
    """

    def __init__(self,
                 temporal_type : str  = "lstm",   # "lstm" | "transformer"
                 use_gnn       : bool = False,     # True requires PyG + graphs
                 epochs        : int  = 30,
                 batch_size    : int  = 64,
                 lr            : float= 1e-3):
        self.temporal_type = temporal_type
        self.use_gnn       = use_gnn
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.lr            = lr

    # ── build combined dataset ─────────────────────────────────────────────
    @staticmethod
    def combine_datasets(datasets: dict,
                         attack_keys: list) -> tuple:
        """
        Merge clean + multiple attack window arrays into one (X, y).
        """
        X_parts, y_parts = [], []
        if "clean" in datasets:
            X_parts.append(datasets["clean"]["X"])
            y_parts.append(datasets["clean"]["y"])
        for k in attack_keys:
            if k in datasets:
                X_parts.append(datasets[k]["X"])
                y_parts.append(datasets[k]["y"])
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        # shuffle
        idx = np.random.permutation(len(y))
        return X[idx], y[idx]

    def run(self, datasets: dict) -> dict:
        attack_keys = [k for k in datasets
                       if k not in ("clean", "raw_scada", "raw_meter")]

        print("\n" + "=" * 60)
        print("  MODULE 2 – DETECTION ENGINE")
        print("=" * 60)
        print(f"  Attack scenarios : {attack_keys}")

        X, y = self.combine_datasets(datasets, attack_keys)
        F    = X.shape[-1]   # feature dim

        print(f"  Dataset shape    : X={X.shape}  y={y.shape}")
        print(f"  Attack rate      : {y.mean()*100:.1f}%")

        # class weights for imbalanced data
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum() + 1
        cw    = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float)

        # ── data loaders ───────────────────────────────────────────────────
        tr_ld, va_ld, te_ld = make_loaders(X, y,
                                           batch_size=self.batch_size)

        # ── supervised spatio-temporal model ───────────────────────────────
        print(f"\n[Engine] Building SpatioTemporalDetector "
              f"(temporal={self.temporal_type}) …")
        st_model = SpatioTemporalDetector(
            feature_dim   = F,
            temporal_type = self.temporal_type,
            use_gnn       = self.use_gnn,
        )
        st_trainer = Trainer(st_model, lr=self.lr, patience=8,
                             mode="supervised", class_weights=cw)
        st_trainer.fit(tr_ld, va_ld, epochs=self.epochs)
        st_res = st_trainer.evaluate(te_ld)

        # ── autoencoder (unsupervised) ─────────────────────────────────────
        print("\n[Engine] Building LSTMAutoencoder …")
        ae_model   = LSTMAutoencoder(feature_dim=F)
        ae_trainer = Trainer(ae_model, lr=self.lr, patience=8,
                             mode="autoencoder")
        # train only on clean data
        X_clean = datasets["clean"]["X"]
        y_clean = datasets["clean"]["y"]
        tr_c, va_c, _ = make_loaders(X_clean, y_clean,
                                     batch_size=self.batch_size)
        ae_trainer.fit(tr_c, va_c, epochs=self.epochs)

        # ── anomaly scores ─────────────────────────────────────────────────
        print("\n[Engine] Computing anomaly scores …")
        st_scores  = self._extract_scores(st_model, te_ld, mode="supervised")
        ae_scores  = self._extract_ae_scores(ae_model, te_ld)

        fuser = AnomalyScoreFusion(strategy="weighted", weights=[0.7, 0.3])
        fused = fuser.fuse({"st": st_scores, "ae": ae_scores})
        alerts= AnomalyScoreFusion.threshold_alerts(fused, threshold=0.5)

        # ── baselines ──────────────────────────────────────────────────────
        baselines = BaselineDetectors()
        X_tr_np   = np.concatenate([b[0].numpy() for b in tr_ld], axis=0)
        y_tr_np   = np.concatenate([b[1].numpy() for b in tr_ld], axis=0)
        X_te_np   = np.concatenate([b[0].numpy() for b in te_ld], axis=0)
        y_te_np   = np.concatenate([b[1].numpy() for b in te_ld], axis=0)

        baselines.fit_unsupervised(X_tr_np)
        baselines.fit_supervised(X_tr_np, y_tr_np)
        baselines.evaluate(X_te_np, y_te_np)

        return {
            "model"          : st_model,
            "autoencoder"    : ae_model,
            "history"        : st_trainer.history,
            "test_results"   : st_res,
            "anomaly_scores" : fused,
            "alerts"         : alerts,
            "baselines"      : baselines,
        }

    # ── helpers ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _extract_scores(self, model, loader, mode="supervised") -> np.ndarray:
        model.eval()
        scores = []
        for x, _ in loader:
            x = x.to(DEVICE)
            _, s = model(x)
            scores.append(s.cpu().numpy())
        return np.concatenate(scores)

    @torch.no_grad()
    def _extract_ae_scores(self, model, loader) -> np.ndarray:
        model.eval()
        scores = []
        for x, _ in loader:
            x = x.to(DEVICE)
            s = model.anomaly_score(x)
            # normalise to [0,1]
            s = s.cpu().numpy()
            scores.append(s)
        all_s = np.concatenate(scores)
        return (all_s - all_s.min()) / (all_s.ptp() + 1e-9)

    def save_model(self, results: dict, path: str = "detection_model.pt"):
        torch.save({
            "state_dict"   : results["model"].state_dict(),
            "ae_state_dict": results["autoencoder"].state_dict(),
            "history"      : results["history"],
        }, path)
        print(f"[Engine] Model saved → {path}")

    def load_model(self, model: nn.Module, path: str = "detection_model.pt"):
        ckpt = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"[Engine] Model loaded ← {path}")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# 8.  QUICK-START DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Option A: use Module 1 directly ───────────────────────────────────
    try:
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "module1", "module1_smart_grid_data_layer.py")
        m1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m1)

        pipeline = m1.SmartGridDataPipeline()
        datasets = pipeline.run()
        print("[Module 2] Loaded datasets from Module 1.")

    except Exception as e:
        # ── Option B: generate synthetic stand-in data ─────────────────────
        print(f"[Module 2] Module 1 not found ({e}). Generating stub data …")
        rng = np.random.default_rng(SEED)
        N, T, F = 2000, 30, 3

        X_clean = rng.normal(0, 1, (N, T, F)).astype(np.float32)
        y_clean = np.zeros(N, dtype=int)

        X_att   = rng.normal(0.5, 1.2, (N, T, F)).astype(np.float32)
        y_att   = np.ones(N, dtype=int)

        datasets = {
            "clean"    : {"X": X_clean,              "y": y_clean},
            "FDI"      : {"X": X_att,                "y": y_att},
            "cascading": {"X": X_att * 1.1,          "y": y_att},
        }

    # ── Run detection engine ───────────────────────────────────────────────
    engine = DetectionEngine(
        temporal_type = "lstm",    # switch to "transformer" if preferred
        use_gnn       = False,     # set True when PyG graphs are available
        epochs        = 20,
        batch_size    = 64,
        lr            = 1e-3,
    )

    results = engine.run(datasets)

    print(f"\n[Module 2] Fused anomaly scores shape : "
          f"{results['anomaly_scores'].shape}")
    print(f"[Module 2] Alert count                : "
          f"{results['alerts'].sum()} / {len(results['alerts'])}")

    # Optionally persist the model:
    # engine.save_model(results, "smartgrid_detector.pt")
