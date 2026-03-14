# California Housing — Autonomous ML Research

Autonomous research on the [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) using the AutoResearch framework. An AI agent systematically explores model architectures, hyperparameters, and training strategies to minimize prediction error.

## Dataset

- **20,640 samples** from the 1990 US Census
- **8 features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target**: MedHouseVal (median house value in $100k units)
- **Task**: Regression (minimize MSE on normalized targets)

## Quick Start

```bash
# Install dependencies
uv sync

# Detect your platform
uv run ar detect

# Run the baseline
uv run train.py

# Start autonomous research (point an AI agent to program.md)
# Or run manually:
uv run ar run --max-experiments 10

# Check progress
uv run ar status
uv run ar analyze
```

## How It Works

1. `prepare.py` (read-only) loads data, normalizes features/targets, and provides fixed evaluation
2. `train.py` (agent-editable) defines the model, optimizer, and training loop
3. `program.md` instructs the AI agent on the experiment protocol
4. `task.yaml` declares the research configuration
5. `results.jsonl` logs every experiment with hypothesis, metric, and status

The agent modifies only `train.py`, runs it, evaluates the result, and keeps improvements or discards regressions. Each experiment runs in a 60-second time budget.

## Project Structure

```
train.py          # Editable training script (agent modifies this)
prepare.py        # Read-only data loading and evaluation
program.md        # Agent instructions with research hints
task.yaml         # Research task configuration
data/dataset.csv  # California Housing dataset
results.jsonl     # Experiment log (created on first run)
ar/               # AutoResearch framework (platform detection, tracking, CLI)
```

## Baseline

| Metric | Value |
|---|---|
| val_loss (MSE) | 0.178453 |
| Parameters | 9,473 |
| Architecture | MLP [128, 64] |
| Platform | Apple M2 (MPS) |

## Research Directions

- **Architecture**: Wider/deeper networks, residual connections, batch normalization
- **Loss functions**: Huber/SmoothL1 (target is clipped at 5.0)
- **Feature engineering**: Lat/lon interactions, polynomial features
- **Optimization**: Learning rate scheduling, different optimizers
- **Regularization**: Dropout tuning, weight decay

## Platform Support

Runs on CPU, Apple MPS, and NVIDIA CUDA. Platform is auto-detected.
