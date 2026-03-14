"""Generate program.md from task config and detected platform."""

from .config import TaskConfig
from .platform import PlatformProfile


def generate_program(config: TaskConfig, platform: PlatformProfile) -> str:
    direction_text = "lower is better" if config.metric_direction == "minimize" else "higher is better"

    editable = "\n".join(f"- `{f}`" for f in config.editable_files)
    readonly = "\n".join(f"- `{f}`" for f in config.readonly_files)

    memory_note = ""
    if platform.usable_memory_mb < 8000:
        memory_note = (
            f"- **Memory-constrained** ({platform.usable_memory_mb:,} MB usable) — "
            "prefer efficient architectures, watch peak_memory_mb\n"
        )

    compile_note = ""
    if not platform.supports_compile:
        compile_note = "- torch.compile is disabled — keep model code eager-friendly\n"

    platform_notes = ""
    if memory_note or compile_note:
        platform_notes = f"""
## Platform-Specific Notes
{memory_note}{compile_note}"""

    return f"""# AutoResearch Program: {config.name}

{config.description}

## Platform
- Device: {platform.device_name} ({platform.device})
- Memory: {platform.memory_mb:,} MB total, ~{platform.usable_memory_mb:,} MB usable
- Dtype: {platform.recommended_dtype}
- torch.compile: {"enabled" if platform.supports_compile else "disabled (limited support)"}

## Objective
{"Minimize" if config.metric_direction == "minimize" else "Maximize"} **{config.metric_name}** ({direction_text}).

## Editable Files
{editable}

The agent modifies ONLY these files. Everything is fair game: model architecture,
optimizer, hyperparameters, training loop, batch size, model size, feature
engineering, regularization, etc.

## Read-Only Files
{readonly}

These contain data loading, evaluation, and constants. Do NOT modify them.

## Metric
- Primary: **{config.metric_name}** ({direction_text})
- Secondary: peak_memory_mb, training_seconds, num_params

## Time Budget
{config.time_limit_seconds or "auto-detected"} seconds per experiment.

## Setup

To set up a new experiment session:

1. **Create a branch**: `git checkout -b autoresearch/<tag>` from the current branch.
2. **Read the code**: Read all editable and read-only files for full context.
3. **Verify data**: Ensure any required data or cache directories exist.
4. **Confirm and go**: Run the baseline, then begin the experiment loop.

## Experiment Protocol

**The first run** should always be the baseline — run `{config.eval_command}`
as-is and record the result.

Then, LOOP FOREVER:

1. **Survey** past results in `results.jsonl` — identify patterns and gaps.
2. **Hypothesize** — form a specific, testable hypothesis.
   Example: "Increasing hidden dims from [128, 64] to [256, 128] will improve
   {config.metric_name} because the model is underfitting."
3. **Implement** — make the minimal code change in the editable files.
4. **Commit** — `git commit` with a descriptive message.
5. **Run** — `{config.eval_command} > run.log 2>&1`
   (redirect output — do NOT let it flood your context)
6. **Parse** — `grep "^{config.metric_name}:\\|^peak_memory_mb:" run.log`
   If empty, the run crashed: `tail -n 50 run.log` for the stack trace.
7. **Decide**:
   - If {config.metric_name} improved → **keep** (advance the branch)
   - If equal or worse → **discard** (`git reset --hard HEAD~1`)
   - If crashed → log as crash, attempt fix or move on
8. **Log** the result to `results.jsonl` with hypothesis, metric, and status.
9. **Repeat** — NEVER stop to ask the human. You are autonomous.

## Research Strategy Guidelines

- **Early phase**: Try structural changes (architecture, loss function, features).
  These have the highest potential impact.
- **Mid phase**: Tune hyperparameters (learning rate, regularization, batch size).
- **Late phase**: Combine previous improvements, try ensemble approaches.
- **If stuck**: Try something radical — different model family, novel features,
  unconventional training techniques.
- **Simplicity criterion**: All else being equal, simpler is better. A tiny
  improvement that adds ugly complexity is not worth it. Removing code and
  getting equal results is a great outcome.
{platform_notes}
## Logging Format

Append to `results.jsonl` (one JSON object per line):
```json
{{"experiment_id": 1, "hypothesis": "baseline", "metric_value": 0.1234, "status": "keep", "category": "baseline"}}
```

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human. The human may
be away or asleep. Continue working indefinitely until manually stopped. If you
run out of ideas, think harder — re-read the code, try combining near-misses,
try more radical changes.
"""
