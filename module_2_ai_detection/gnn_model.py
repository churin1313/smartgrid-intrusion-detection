"""
Module 2 - Step 2: Graph Neural Network (GNN)
Spatial encoder — learns node embeddings from grid topology.
Architecture: GCNConv / GATConv → BatchNorm → ReLU → node classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import DataLoader


# ─────────────────────────────────────────────
# 1. GNN ENCODER (Spatial Embedding)
# ─────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder.
    Uses Graph Attention Network (GAT) layers for message passing.
    GAT allows each node to attend to its neighbors with learned weights —
    critical for detecting anomalies in specific buses.

    Input : node features x [num_nodes, in_channels]
    Output: node embeddings  [num_nodes, hidden_dim]
    """

    def __init__(self, in_channels, hidden_dim=64, out_dim=128, heads=4, dropout=0.3):
        super(GNNEncoder, self).__init__()

        # Layer 1: GAT with multi-head attention
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            concat=True   # concat all heads → hidden_dim * heads
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        # Layer 2: GAT, reduce to out_dim
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=out_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )
        self.bn2 = nn.BatchNorm1d(out_dim)

        # Layer 3: GCN for structural smoothing
        self.conv3 = GCNConv(out_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        return x  # Node embeddings: [num_nodes, out_dim]


# ─────────────────────────────────────────────
# 2. NODE-LEVEL CLASSIFIER (Attack Detection)
# ─────────────────────────────────────────────

class GNNDetector(nn.Module):
    """
    Full GNN-based node-level attack detector.
    Outputs per-node anomaly scores (logits for binary classification).
    """

    def __init__(self, in_channels, hidden_dim=64, embed_dim=128,
                 num_classes=2, heads=4, dropout=0.3):
        super(GNNDetector, self).__init__()

        self.encoder = GNNEncoder(in_channels, hidden_dim, embed_dim, heads, dropout)

        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, edge_index, edge_attr=None):
        embeddings = self.encoder(x, edge_index, edge_attr)
        logits = self.classifier(embeddings)
        return logits, embeddings  # return both for fusion layer

    def anomaly_score(self, x, edge_index, edge_attr=None):
        """
        Returns per-node anomaly probability [0, 1].
        Used as output to the fusion layer.
        """
        logits, embeddings = self.forward(x, edge_index, edge_attr)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1], embeddings  # class 1 = attack probability


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

def train_gnn(model, graphs, epochs=50, lr=1e-3, batch_size=16,
              train_ratio=0.8, device='cpu'):
    """
    Train the GNN detector on graph snapshots.

    Args:
        model  : GNNDetector instance
        graphs : list of PyG Data objects (from graph_construction.py)
        epochs : training epochs
        lr     : learning rate
        device : 'cuda' or 'cpu'
    """
    model = model.to(device)

    # Train/val split
    split = int(len(graphs) * train_ratio)
    train_graphs = graphs[:split]
    val_graphs   = graphs[split:]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)

    # Class-weighted loss to handle imbalance (attacks are rare)
    class_weights = torch.tensor([1.0, 5.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\nTraining GNN on {len(train_graphs)} graphs | Validating on {len(val_graphs)}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # ── TRAIN ──
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # ── VALIDATE ──
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index, batch.edge_attr)
                val_loss += criterion(logits, batch.y).item()
                preds = logits.argmax(dim=-1)
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)

        acc = correct / total
        scheduler.step()

        history['train_loss'].append(total_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f} | "
                  f"Val Acc: {acc:.4f}")

    print("=" * 60)
    return model, history


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load graphs from Step 1
    graphs = torch.load("data/pyg_graphs.pt", weights_only=False)
    sample = graphs[0]

    in_channels = sample.x.shape[1]   # 6 features per node
    print(f"Node features: {in_channels} | Nodes: {sample.x.shape[0]}")

    # Initialize model
    model = GNNDetector(
        in_channels=in_channels,
        hidden_dim=64,
        embed_dim=128,
        num_classes=2,
        heads=4,
        dropout=0.3
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trained_model, history = train_gnn(
        model, graphs, epochs=50, lr=1e-3, batch_size=16, device=device
    )

    # Save
    torch.save(trained_model.state_dict(), "gnn_detector.pth")
    print("\nGNN model saved to gnn_detector.pth")
