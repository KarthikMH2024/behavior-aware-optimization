# ============================================================
# Imports
# ============================================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import netron

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv("data.csv")

X_raw = df[[f"s{i+1}" for i in range(25)]].values.astype(np.float32)
y = df["best_route"].values

X = X_raw.reshape(-1, 1, 5, 5)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
route_names = list(le.classes_)

print("Route encoding:", dict(zip(route_names, range(num_classes))))

# ============================================================
# Train / Val / Test split
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=16, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    batch_size=16
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
    batch_size=16
)

# ============================================================
# Lightweight Transformer (Global Context)
# ============================================================
class LightweightAttention(nn.Module):
    def __init__(self, d_model=16, heads=2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 25, 1)
        x = self.embed(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x.permute(0, 2, 1).contiguous()
        return x.view(B, -1, 5, 5)

# ============================================================
# Transformer → CNN → Behavioral Pooling
# ============================================================
class RouteModel(nn.Module):
    def __init__(self, num_classes, out_channels, kernel_size, pooling):
        super().__init__()

        self.attn = LightweightAttention(d_model=16)

        self.conv = nn.Conv2d(16, out_channels, kernel_size)
        self.relu = nn.ReLU()
        self.pooling = pooling

        conv_out = 5 - kernel_size + 1

        if pooling in ["avg", "min"]:
            fc_in = out_channels
        else:
            fc_in = out_channels * conv_out * conv_out

        self.fc1 = nn.Linear(fc_in, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.attn(x)
        x = self.relu(self.conv(x))

        if self.pooling == "avg":
            x = torch.mean(x, dim=(2, 3))
        elif self.pooling == "min":
            x, _ = torch.min(x.view(x.size(0), x.size(1), -1), dim=2)

        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ============================================================
# Training & Behavioral Fitness
# ============================================================
def train_and_evaluate(cfg, epochs=20):
    model = RouteModel(
        num_classes,
        cfg["out_channels"],
        cfg["kernel_size"],
        cfg["pooling"]
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Validation behavioral metrics
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            loss = nn.CrossEntropyLoss(reduction="none")(preds, yb)
            val_losses.extend(loss.numpy())

    val_losses = np.array(val_losses)
    fitness = val_losses.max() + val_losses.var()

    # Test
    test_preds, test_losses = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            test_preds.extend(preds.argmax(1).numpy())
            test_losses.extend(
                nn.CrossEntropyLoss(reduction="none")(preds, yb).numpy()
            )

    return fitness, model, np.array(test_preds), np.array(test_losses)

# ============================================================
# Baseline (trained avg-pool CNN)
# ============================================================
def compute_baseline():
    cfg = {"out_channels": 8, "kernel_size": 3, "pooling": "avg"}
    _, _, _, losses = train_and_evaluate(cfg, epochs=15)
    return losses

baseline_losses = compute_baseline()

# ============================================================
# Genetic Algorithm Search
# ============================================================
population = [
    {"out_channels": 8, "kernel_size": 2, "pooling": "avg"},
    {"out_channels": 8, "kernel_size": 2, "pooling": "min"},
    {"out_channels": 16, "kernel_size": 3, "pooling": "avg"},
    {"out_channels": 16, "kernel_size": 3, "pooling": "min"},
]

best = None
best_score = np.inf

for gen in range(3):
    print(f"\n=== Generation {gen+1} ===")
    results = []

    for cfg in population:
        score, model, preds, losses = train_and_evaluate(cfg)
        print(cfg, "→ fitness:", round(score, 4))
        results.append((score, cfg, model, preds, losses))

    results.sort(key=lambda x: x[0])
    best_score, best_cfg, best_model, best_preds, best_losses = results[0]
    best = results[0]

    parent = best_cfg
    population = [
        parent,
        {
            "out_channels": max(4, parent["out_channels"] + np.random.choice([-4, 4])),
            "kernel_size": np.random.choice([2, 3]),
            "pooling": np.random.choice(["avg", "min"])
        }
    ]

# ============================================================
# Final Evaluation
# ============================================================
print("\n=== Final Test Performance ===")
print("Worst-case loss:", best_losses.max())
print("Variance:", best_losses.var())
print("95th percentile:", np.percentile(best_losses, 95))
print("Regret vs baseline:", np.mean(best_losses - baseline_losses))

acc = accuracy_score(y_test, best_preds)
print("\nAccuracy:", acc)
print(classification_report(y_test, best_preds, target_names=route_names))

# ============================================================
# Save for Netron
# ============================================================
torch.save(best_model, "route_transformer_cnn_behavioral.pt")
print("\nSaved model for Netron.")
netron.start("route_transformer_cnn_behavioral.pt")
input("Press ENTER to close Netron...")

