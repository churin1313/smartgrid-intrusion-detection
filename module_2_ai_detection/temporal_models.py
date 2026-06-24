import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from gnn_model import GNNDetector


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class GridSequenceDataset(Dataset):
    def __init__(self, embeddings_list, labels_list, window_size=10):
        self.sequences = []
        self.targets   = []

        T = len(embeddings_list)
        num_nodes = embeddings_list[0].shape[0]

        for t in range(window_size, T):
            for node in range(num_nodes):
                seq = torch.stack([
                    embeddings_list[t - window_size + i][node]
                    for i in range(window_size)
                ])
                label = labels_list[t][node]

                self.sequences.append(seq)
                self.targets.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ─────────────────────────────────────────────
# LSTM MODEL
# ─────────────────────────────────────────────

class LSTMDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.fc(last_hidden)

    def anomaly_score(self, x):
        return F.softmax(self.forward(x), dim=-1)[:, 1]


# ─────────────────────────────────────────────
# TRANSFORMER MODEL
# ─────────────────────────────────────────────

class TransformerDetector(nn.Module):
    def __init__(self, input_dim, d_model=128):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=3
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        return self.fc(out[:, -1])

    def anomaly_score(self, x):
        return F.softmax(self.forward(x), dim=-1)[:, 1]


# ─────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────

def train_model(model, dataset, device, epochs=40):
    model = model.to(device)

    split = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, range(split))
    val_set   = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {model.__class__.__name__}")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch} | Loss: {total_loss:.4f}")

    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load graph data
    graphs = torch.load("data/pyg_graphs.pt", weights_only=False)

    # Load trained GNN
    gnn = GNNDetector(in_channels=6, hidden_dim=64, embed_dim=128)
    gnn.load_state_dict(torch.load("models/gnn_detector.pth", map_location=device))
    gnn.eval().to(device)

    print("\nExtracting GNN embeddings...")

    embeddings_list, labels_list = [], []

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            _, emb = gnn(g.x, g.edge_index, g.edge_attr)
            embeddings_list.append(emb.cpu())
            labels_list.append(g.y.cpu())

    dataset = GridSequenceDataset(embeddings_list, labels_list)

    print(f"Dataset size: {len(dataset)}")

    embed_dim = embeddings_list[0].shape[1]

    # Train LSTM
    lstm = LSTMDetector(embed_dim)
    lstm = train_model(lstm, dataset, device)
    torch.save(lstm.state_dict(), "models/lstm_detector.pth")

    # Train Transformer
    tf = TransformerDetector(embed_dim)
    tf = train_model(tf, dataset, device)
    torch.save(tf.state_dict(), "models/transformer_detector.pth")

    print("\nAll models trained and saved successfully 🚀")