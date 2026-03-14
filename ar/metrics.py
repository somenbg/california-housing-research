"""Metric extraction and comparison."""

import re
from dataclasses import dataclass, field


@dataclass
class MetricResult:
    name: str
    value: float
    direction: str  # "minimize" or "maximize"
    secondary: dict = field(default_factory=dict)

    def is_better_than(self, other: "MetricResult") -> bool:
        if self.direction == "minimize":
            return self.value < other.value
        return self.value > other.value

    def delta_from(self, baseline: "MetricResult") -> float:
        return self.value - baseline.value

    def __str__(self):
        return f"{self.name}={self.value:.6f}"


# Standard keys the framework knows how to parse
_SECONDARY_KEYS = [
    "training_seconds", "total_seconds", "peak_memory_mb", "peak_vram_mb",
    "mfu_percent", "total_tokens_M", "num_steps", "num_params", "num_params_M",
    "depth", "device", "device_name",
]


def parse_metrics(
    output: str,
    metric_name: str,
    direction: str,
    extra_patterns: dict | None = None,
) -> MetricResult | None:
    """Parse training script output for the primary metric and secondary info.

    Expected output format (after the '---' separator):
        val_loss:          0.1234
        training_seconds:  58.3
        peak_memory_mb:    245.1
    """
    extra_patterns = extra_patterns or {}

    # Try custom regex pattern first
    primary_value = None
    if metric_name in extra_patterns:
        m = re.search(extra_patterns[metric_name], output, re.MULTILINE)
        if m:
            primary_value = float(m.group(1))

    # Fall back to standard "key: value" format
    if primary_value is None:
        pat = rf"^{re.escape(metric_name)}:\s+([0-9.eE+-]+)"
        m = re.search(pat, output, re.MULTILINE)
        if m:
            primary_value = float(m.group(1))

    if primary_value is None:
        return None

    secondary = {}
    for key in _SECONDARY_KEYS:
        pat = rf"^{re.escape(key)}:\s+(.+)"
        m = re.search(pat, output, re.MULTILINE)
        if m:
            val = m.group(1).strip()
            try:
                secondary[key] = float(val)
            except ValueError:
                secondary[key] = val

    return MetricResult(
        name=metric_name,
        value=primary_value,
        direction=direction,
        secondary=secondary,
    )
