"""
California Housing — AutoResearch training script.
Usage: uv run train.py

The agent modifies this file to improve val_loss (MSE, lower is better).
"""

import time

import torch
import torch.nn as nn

from ar.platform import detect_platform
from ar.platform_utils import (
    synchronize,
    get_autocast_context,
    get_peak_memory_mb,
    seed_everything,
    should_compile,
)
from prepare import load_data, evaluate, TIME_BUDGET, NUM_FEATURES

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

HIDDEN_DIMS = [256, 128]    # hidden layer dimensions
DROPOUT = 0.05              # dropout rate
LEARNING_RATE = 1e-3        # Adam learning rate
WEIGHT_DECAY = 1e-5         # L2 regularization
BATCH_SIZE = 256            # training batch size
ACTIVATION = "gelu"         # activation: relu, gelu, silu, tanh

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout, activation):
        super().__init__()
        act_fns = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        act_class = act_fns.get(activation, nn.ReLU)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), act_class(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

platform = detect_platform()
device = torch.device(platform.device)
autocast_ctx = get_autocast_context(platform)
seed_everything(42, device)

print(f"Platform: {platform.device_name} ({platform.device})")
print(f"Memory: {platform.usable_memory_mb:,} MB usable")
print(f"Time budget: {TIME_BUDGET}s")

train_X, train_y, val_X, val_y, stats = load_data(device=device)
input_dim = NUM_FEATURES
output_dim = 1
print(f"Task: regression | Features: {input_dim} | Train: {stats['n_train']} | Val: {stats['n_val']}")

model = MLP(input_dim, output_dim, HIDDEN_DIMS, DROPOUT, ACTIVATION).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

if should_compile(platform):
    model = torch.compile(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start = time.time()
total_training_time = 0.0
step = 0
epoch = 0
n_train = len(train_X)

while True:
    model.train()
    epoch += 1
    perm = torch.randperm(n_train, device=device)

    for i in range(0, n_train, BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        xb, yb = train_X[idx], train_y[idx]

        with autocast_ctx:
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1

    synchronize(device)
    total_training_time = time.time() - t_start

    if total_training_time >= TIME_BUDGET:
        break

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
val_metric = evaluate(model, val_X, val_y, autocast_ctx)
peak_mem = get_peak_memory_mb(device)
t_end = time.time()

print("---")
print(f"val_loss:         {val_metric:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_memory_mb:   {peak_mem:.1f}")
print(f"num_steps:        {step}")
print(f"num_epochs:       {epoch}")
print(f"num_params:       {num_params}")
print(f"device:           {platform.device}")
print(f"device_name:      {platform.device_name}")
