"""Experiment tracking — results.jsonl persistence and querying."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExperimentRecord:
    experiment_id: int
    timestamp: str
    commit: str
    hypothesis: str
    category: str
    change_summary: str
    files_changed: list
    metric_name: str
    metric_value: float
    metric_direction: str
    secondary_metrics: dict
    status: str  # "keep", "discard", "crash"
    reasoning: str = ""
    parent_experiment: int = 0
    platform_device: str = ""
    platform_name: str = ""


class ExperimentTracker:
    """Read/write experiment records to results.jsonl."""

    def __init__(self, results_path: str = "results.jsonl"):
        self.results_path = results_path
        self._records: list[ExperimentRecord] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.results_path):
            return
        with open(self.results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._records.append(ExperimentRecord(**data))

    @property
    def experiments(self) -> list[ExperimentRecord]:
        return list(self._records)

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def next_id(self) -> int:
        if not self._records:
            return 1
        return max(r.experiment_id for r in self._records) + 1

    @property
    def best_result(self) -> Optional[ExperimentRecord]:
        kept = [r for r in self._records if r.status == "keep"]
        if not kept:
            return None
        if kept[0].metric_direction == "minimize":
            return min(kept, key=lambda r: r.metric_value)
        return max(kept, key=lambda r: r.metric_value)

    @property
    def baseline(self) -> Optional[ExperimentRecord]:
        kept = [r for r in self._records if r.status == "keep"]
        return kept[0] if kept else None

    def record(self, rec: ExperimentRecord):
        self._records.append(rec)
        with open(self.results_path, "a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    def summary(self) -> str:
        total = len(self._records)
        if total == 0:
            return "No experiments recorded yet."

        kept = sum(1 for r in self._records if r.status == "keep")
        discarded = sum(1 for r in self._records if r.status == "discard")
        crashed = sum(1 for r in self._records if r.status == "crash")

        lines = [
            f"Experiments:  {total} total",
            f"  Kept:       {kept}",
            f"  Discarded:  {discarded}",
            f"  Crashed:    {crashed}",
        ]

        bl = self.baseline
        best = self.best_result
        if bl and best:
            lines.append(f"Baseline:     {bl.metric_value:.6f} ({bl.metric_name})")
            lines.append(f"Best:         {best.metric_value:.6f}")
            delta = best.metric_value - bl.metric_value
            if bl.metric_direction == "maximize":
                pct = (delta / bl.metric_value * 100) if bl.metric_value else 0
                lines.append(f"Improvement:  {delta:+.6f} ({pct:+.1f}%)")
            else:
                pct = (-delta / bl.metric_value * 100) if bl.metric_value else 0
                lines.append(f"Improvement:  {delta:+.6f} ({pct:+.1f}%)")
            if best.commit:
                lines.append(f"Best commit:  {best.commit} — {best.change_summary}")

        return "\n".join(lines)
