"""Data preparation and evaluation for tabular AutoResearch.

Provides data loading, evaluation, and constants.
DO NOT MODIFY — this file is read-only for the agent.

If data/dataset.csv exists, loads it. Otherwise generates synthetic data
so that `uv run train.py` works immediately out of the box.
"""

import os
import math
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60  # training time budget in seconds
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
VAL_FRACTION = 0.2
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples=5000, n_features=20, noise=0.1, seed=SEED):
    """Generate a synthetic regression dataset with nonlinear interactions."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)

    true_w = rng.randn(n_features).astype(np.float32)
    y = (
        X @ true_w
        + 0.5 * (X[:, 0] * X[:, 1])
        - 0.3 * np.sin(X[:, 2] * 3)
        + 0.2 * (X[:, 3] ** 2)
        + noise * rng.randn(n_samples).astype(np.float32)
    )
    return X, y


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path=None, target_col=None, device="cpu", val_fraction=VAL_FRACTION):
    """Load tabular data. Returns tensors on the specified device.

    If data_path points to an existing CSV, loads it. Otherwise generates
    synthetic data so the template works immediately.

    Returns:
        train_X, train_y, val_X, val_y, task_type, num_classes
        task_type is "regression" or "classification" (auto-detected).
    """
    data_path = data_path or DATA_PATH
    device_t = torch.device(device) if isinstance(device, str) else device

    if data_path and os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        if target_col is None:
            target_col = df.columns[-1]
        y_raw = df[target_col].values
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values.astype(np.float32)
    else:
        print("No dataset found — using synthetic regression data (5000 samples, 20 features)")
        X, y_raw = generate_synthetic_data()

    # Auto-detect task type
    unique = np.unique(y_raw)
    if len(unique) <= 20 and np.all(unique == unique.astype(int)):
        task_type = "classification"
        y = y_raw.astype(np.int64)
        # Remap classes to 0..n-1
        classes = sorted(np.unique(y))
        class_map = {c: i for i, c in enumerate(classes)}
        y = np.array([class_map[c] for c in y], dtype=np.int64)
        num_classes = len(classes)
    else:
        task_type = "regression"
        y = y_raw.astype(np.float32)
        num_classes = 0

    # Normalize features (zero mean, unit variance)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    # Normalize regression targets
    if task_type == "regression":
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y = (y - y_mean) / y_std

    # Train/val split (deterministic)
    n = len(X)
    n_val = max(1, int(n * val_fraction))
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_X = torch.tensor(X[train_idx], device=device_t)
    train_y = torch.tensor(y[train_idx], device=device_t)
    val_X = torch.tensor(X[val_idx], device=device_t)
    val_y = torch.tensor(y[val_idx], device=device_t)

    return train_X, train_y, val_X, val_y, task_type, num_classes


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_X, val_y, task_type, autocast_ctx=None):
    """Evaluate model on validation set.

    Returns:
        For regression: val_loss (MSE, lower is better)
        For classification: val_accuracy (higher is better)
    """
    import contextlib
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext()
    model.eval()

    with ctx:
        pred = model(val_X)

    if task_type == "classification":
        pred_classes = pred.argmax(dim=-1)
        return (pred_classes == val_y).float().mean().item()
    else:
        pred = pred.float().squeeze(-1)
        val_y_f = val_y.float()
        return ((pred - val_y_f) ** 2).mean().item()
