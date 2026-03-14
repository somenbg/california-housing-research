"""Data preparation and evaluation for California Housing AutoResearch.

Loads the California Housing dataset, normalizes features and targets,
splits into train/val, and provides fixed evaluation.
DO NOT MODIFY — this file is read-only for the agent.
"""

import os
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60  # training time budget in seconds
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
VAL_FRACTION = 0.2
SEED = 42

# Feature descriptions for reference:
#   MedInc     - median income in block group
#   HouseAge   - median house age in block group
#   AveRooms   - average number of rooms per household
#   AveBedrms  - average number of bedrooms per household
#   Population - block group population
#   AveOccup   - average number of household members
#   Latitude   - block group latitude
#   Longitude  - block group longitude
#   MedHouseVal - median house value ($100k) [TARGET]

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET_NAME = "MedHouseVal"
NUM_FEATURES = 8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(device="cpu", val_fraction=VAL_FRACTION):
    """Load California Housing data. Returns tensors on the specified device.

    Returns:
        train_X, train_y, val_X, val_y, feature_stats
        feature_stats is a dict with normalization info for reference.
    """
    import pandas as pd

    device_t = torch.device(device) if isinstance(device, str) else device

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df[TARGET_NAME].values.astype(np.float32)

    # Normalize features (zero mean, unit variance)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    X = (X - x_mean) / x_std

    # Normalize target
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y = (y - y_mean) / y_std

    # Deterministic train/val split
    n = len(X)
    n_val = max(1, int(n * val_fraction))
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_X = torch.tensor(X[train_idx], device=device_t)
    train_y = torch.tensor(y[train_idx], device=device_t)
    val_X = torch.tensor(X[val_idx], device=device_t)
    val_y = torch.tensor(y[val_idx], device=device_t)

    feature_stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }

    return train_X, train_y, val_X, val_y, feature_stats


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_X, val_y, autocast_ctx=None):
    """Evaluate model on validation set.

    Returns val_loss (MSE on normalized targets, lower is better).
    """
    import contextlib
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    model.eval()

    with ctx:
        pred = model(val_X)

    pred = pred.float().squeeze(-1)
    val_y_f = val_y.float()
    return ((pred - val_y_f) ** 2).mean().item()
