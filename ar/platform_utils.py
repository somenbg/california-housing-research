"""Portable utilities for device-agnostic training.

Centralizes all platform-specific dispatch so templates never
need to write `if device == "cuda"` checks.
"""

import sys
import contextlib

import torch

from .platform import PlatformProfile


def synchronize(device: torch.device):
    """Device-agnostic synchronization barrier."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def get_autocast_context(platform: PlatformProfile):
    """Return the appropriate mixed-precision autocast context manager."""
    if platform.device == "cuda":
        return torch.amp.autocast("cuda", dtype=platform.recommended_dtype)
    elif platform.device == "mps":
        return torch.amp.autocast("mps", dtype=torch.float16)
    elif platform.supports_bfloat16:
        return torch.amp.autocast("cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def get_peak_memory_mb(device: torch.device) -> float:
    """Peak memory usage in MB, works on all platforms."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except Exception:
            return 0.0
    else:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return rusage / (1024 * 1024)  # macOS: bytes
        return rusage / 1024  # Linux: KB


def seed_everything(seed: int, device: torch.device):
    """Seed all relevant RNGs for reproducibility."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def should_compile(platform: PlatformProfile) -> bool:
    """Whether torch.compile is expected to work reliably."""
    return platform.supports_compile


def should_pin_memory(platform: PlatformProfile) -> bool:
    """Whether to use pinned memory for data loading."""
    return platform.device == "cuda"


def get_embedding_dtype(platform: PlatformProfile) -> torch.dtype:
    """Embedding dtype matched to autocast dtype for consistency."""
    if platform.device == "cuda" and platform.supports_bfloat16:
        return torch.bfloat16
    elif platform.device == "mps":
        return torch.float16
    elif platform.supports_bfloat16:
        return torch.bfloat16
    return torch.float32
