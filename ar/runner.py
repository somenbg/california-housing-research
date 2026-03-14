"""Experiment runner — execute training scripts, capture output, parse metrics."""

import os
import subprocess
import time
from typing import Optional

from .config import TaskConfig
from .metrics import MetricResult, parse_metrics


class ExperimentRunner:
    """Run a training script, capture output, and parse metrics."""

    def __init__(self, config: TaskConfig):
        self.config = config

    def run(self, timeout_seconds: int = 0) -> "RunResult":
        """Execute the training script and return parsed results.

        Args:
            timeout_seconds: Kill the process after this many seconds.
                0 means use config.time_limit_seconds * 2 + 120 as a generous upper bound.
        """
        cmd = self.config.eval_command
        if not timeout_seconds:
            budget = self.config.time_limit_seconds or 300
            timeout_seconds = budget * 2 + 120

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=os.getcwd(),
            )
            elapsed = time.time() - t0
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            return RunResult(
                success=False,
                metric=None,
                stdout="",
                stderr=f"TIMEOUT after {elapsed:.0f}s",
                elapsed=elapsed,
                returncode=-1,
            )

        # Write log
        log_path = "run.log"
        with open(log_path, "w") as f:
            f.write(stdout)
            if stderr:
                f.write("\n--- STDERR ---\n")
                f.write(stderr)

        if returncode != 0:
            # Grab last 50 lines for crash diagnosis
            lines = (stdout + "\n" + stderr).strip().split("\n")
            tail = "\n".join(lines[-50:])
            return RunResult(
                success=False,
                metric=None,
                stdout=stdout,
                stderr=stderr,
                elapsed=elapsed,
                returncode=returncode,
                error_tail=tail,
            )

        metric = parse_metrics(
            stdout,
            self.config.metric_name,
            self.config.metric_direction,
            self.config.extract_patterns or None,
        )

        return RunResult(
            success=metric is not None,
            metric=metric,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
            returncode=returncode,
        )


class RunResult:
    def __init__(
        self,
        success: bool,
        metric: Optional[MetricResult],
        stdout: str,
        stderr: str,
        elapsed: float,
        returncode: int,
        error_tail: str = "",
    ):
        self.success = success
        self.metric = metric
        self.stdout = stdout
        self.stderr = stderr
        self.elapsed = elapsed
        self.returncode = returncode
        self.error_tail = error_tail

    @property
    def peak_memory_mb(self) -> float:
        if self.metric and "peak_memory_mb" in self.metric.secondary:
            return float(self.metric.secondary["peak_memory_mb"])
        if self.metric and "peak_vram_mb" in self.metric.secondary:
            return float(self.metric.secondary["peak_vram_mb"])
        return 0.0
