# AutoResearch Program: california-housing

Predict California median house values from census features (8 inputs → 1 output). Dataset: 20,640 samples from the 1990 US Census. Target is MedHouseVal in $100k units. Rich research surface: nonlinear relationships, geographic features, skewed distributions.


## Platform
- Device: Apple M2 (mps)
- Memory: 8,192 MB total, ~4,915 MB usable
- Dtype: torch.float16
- torch.compile: disabled (limited support)

## Objective
Minimize **val_loss** (lower is better).

## Editable Files
- `train.py`

The agent modifies ONLY these files. Everything is fair game: model architecture,
optimizer, hyperparameters, training loop, batch size, model size, feature
engineering, regularization, etc.

## Read-Only Files
- `prepare.py`

These contain data loading, evaluation, and constants. Do NOT modify them.

## Metric
- Primary: **val_loss** (lower is better)
- Secondary: peak_memory_mb, training_seconds, num_params

## Time Budget
60 seconds per experiment.

## Setup

To set up a new experiment session:

1. **Create a branch**: `git checkout -b autoresearch/<tag>` from the current branch.
2. **Read the code**: Read all editable and read-only files for full context.
3. **Verify data**: Ensure any required data or cache directories exist.
4. **Confirm and go**: Run the baseline, then begin the experiment loop.

## Experiment Protocol

**The first run** should always be the baseline — run `uv run train.py`
as-is and record the result.

Then, LOOP FOREVER:

1. **Survey** past results in `results.jsonl` — identify patterns and gaps.
2. **Hypothesize** — form a specific, testable hypothesis.
   Example: "Increasing hidden dims from [128, 64] to [256, 128] will improve
   val_loss because the model is underfitting."
3. **Implement** — make the minimal code change in the editable files.
4. **Commit** — `git commit` with a descriptive message.
5. **Run** — `uv run train.py > run.log 2>&1`
   (redirect output — do NOT let it flood your context)
6. **Parse** — `grep "^val_loss:\|^peak_memory_mb:" run.log`
   If empty, the run crashed: `tail -n 50 run.log` for the stack trace.
7. **Decide**:
   - If val_loss improved → **keep** (advance the branch)
   - If equal or worse → **discard** (`git reset --hard HEAD~1`)
   - If crashed → log as crash, attempt fix or move on
8. **Log** the result to `results.jsonl` with hypothesis, metric, and status.
9. **Repeat** — NEVER stop to ask the human. You are autonomous.

## Dataset Characteristics (Research Hints)

The California Housing dataset has specific properties the agent should exploit:
- **8 features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Geographic structure**: Latitude/Longitude encode spatial patterns — consider interaction features
- **Nonlinear relationships**: MedInc vs price is highly nonlinear; deeper/wider networks may help
- **Skewed distributions**: AveOccup, Population have extreme outliers — robust losses may help
- **Feature interactions**: Room/bedroom ratios, income × location interactions carry signal
- **20,640 samples**: Large enough for moderately complex models, small enough for fast iteration
- **Target is capped at 5.0**: The target MedHouseVal is clipped at $500k — Huber loss may handle this better than MSE

## Research Strategy Guidelines

- **Early phase**: Try structural changes (architecture, loss function, feature engineering).
  These have the highest potential impact. Consider:
  - Wider/deeper networks (current: [128, 64] with 9k params — room to grow)
  - Different loss functions (Huber, SmoothL1 — the target is clipped)
  - Feature engineering (lat/lon interactions, polynomial features)
  - Batch normalization or layer normalization between layers
- **Mid phase**: Tune hyperparameters (learning rate, regularization, batch size).
  - Learning rate scheduling (cosine, warmup, reduce-on-plateau)
  - Dropout rate sweep
  - Weight decay tuning
- **Late phase**: Combine previous improvements, try advanced techniques.
  - Residual connections (skip connections in the MLP)
  - Multi-head output (predict mean + variance)
  - Gradient clipping
- **If stuck**: Try something radical — different model family, novel features,
  unconventional training techniques.
- **Simplicity criterion**: All else being equal, simpler is better. A tiny
  improvement that adds ugly complexity is not worth it. Removing code and
  getting equal results is a great outcome.

## Platform-Specific Notes
- **Memory-constrained** (4,915 MB usable) — prefer efficient architectures, watch peak_memory_mb
- torch.compile is disabled — keep model code eager-friendly

## Logging Format

Append to `results.jsonl` (one JSON object per line):
```json
{"experiment_id": 1, "hypothesis": "baseline", "metric_value": 0.1234, "status": "keep", "category": "baseline"}
```

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human. The human may
be away or asleep. Continue working indefinitely until manually stopped. If you
run out of ideas, think harder — re-read the code, try combining near-misses,
try more radical changes.
