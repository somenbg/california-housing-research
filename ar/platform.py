"""Platform detection and hardware profiling.

Auto-detects CPU, MPS (Apple Silicon), or CUDA and provides a unified
PlatformProfile with recommended training defaults scaled to available hardware.
"""

import os
import sys
import subprocess
import platform as platform_mod
from dataclasses import dataclass

import torch


@dataclass
class PlatformProfile:
    device: str              # "cuda", "mps", "cpu"
    device_name: str         # "NVIDIA H100", "Apple M2", "AMD Ryzen 9"
    compute_type: str        # "gpu_cuda", "gpu_mps", "cpu"
    memory_mb: int           # total device memory (VRAM or system RAM)
    usable_memory_mb: int    # memory available for training
    num_cores: int           # CPU cores
    supports_bfloat16: bool
    supports_float16: bool
    supports_compile: bool
    peak_flops: float        # estimated peak FLOPS (for MFU calculation)

    @property
    def recommended_dtype(self) -> torch.dtype:
        if self.device == "cuda" and self.supports_bfloat16:
            return torch.bfloat16
        elif self.device == "mps":
            return torch.float16
        elif self.supports_bfloat16:
            return torch.bfloat16
        return torch.float32

    def compute_defaults(self, model_class: str = "transformer") -> dict:
        """Compute sensible training defaults scaled to available hardware."""
        mem = self.usable_memory_mb

        if model_class == "transformer":
            if mem >= 40_000:
                return dict(depth=8, seq_len=2048, batch_size=128, time_budget=300)
            elif mem >= 16_000:
                return dict(depth=6, seq_len=1024, batch_size=64, time_budget=300)
            elif mem >= 8_000:
                return dict(depth=4, seq_len=512, batch_size=32, time_budget=300)
            elif mem >= 4_000:
                return dict(depth=2, seq_len=256, batch_size=8, time_budget=120)
            else:
                return dict(depth=1, seq_len=128, batch_size=4, time_budget=60)
        elif model_class == "cnn":
            if mem >= 8_000:
                return dict(batch_size=128, img_size=224, time_budget=300)
            elif mem >= 4_000:
                return dict(batch_size=32, img_size=128, time_budget=120)
            else:
                return dict(batch_size=8, img_size=64, time_budget=60)
        elif model_class == "tabular":
            return dict(batch_size=256, time_budget=60)
        else:
            return dict(time_budget=120)

    def summary(self) -> str:
        compile_status = "enabled" if self.supports_compile else "disabled (limited support)"
        lines = [
            "AutoResearch Platform Detection",
            "───────────────────────────────",
            f"Device:          {self.device} ({self.device_name})",
            f"Memory:          {self.memory_mb:,} MB total → {self.usable_memory_mb:,} MB usable",
            f"Dtype:           {self.recommended_dtype}",
            f"torch.compile:   {compile_status}",
            f"CPU cores:       {self.num_cores}",
            f"Peak FLOPS:      {self.peak_flops:.1e}",
            "",
            "Recommended defaults:",
        ]
        for model_class in ["transformer", "cnn", "tabular"]:
            defaults = self.compute_defaults(model_class)
            parts = [f"{k}={v}" for k, v in defaults.items()]
            lines.append(f"  {model_class:14s} {', '.join(parts)}")
        lines.append("")
        lines.append("Ready. Run 'ar init --task <template>' to get started.")
        lines.append("Available templates: tabular, tiny-lm")
        return "\n".join(lines)


def _sysctl(key: str) -> str:
    """Read a macOS sysctl value."""
    result = subprocess.run(
        ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
    )
    return result.stdout.strip()


# Peak FLOPS lookup tables (bf16/fp16 TFLOPS)
_CUDA_FLOPS = {
    "H100": 989.5e12, "H200": 989.5e12,
    "A100": 312e12, "A10G": 125e12,
    "RTX 4090": 330e12, "RTX 4080": 200e12, "RTX 4070": 150e12,
    "RTX 3090": 142e12, "RTX 3080": 119e12, "RTX 3070": 81e12,
    "RTX 3060": 51e12, "RTX 2080": 80e12,
}

_MPS_FLOPS = {
    "M1 Ultra": 20.8e12, "M1 Max": 10.4e12, "M1 Pro": 5.2e12, "M1": 2.6e12,
    "M2 Ultra": 27.2e12, "M2 Max": 13.6e12, "M2 Pro": 6.8e12, "M2": 3.6e12,
    "M3 Ultra": 28.4e12, "M3 Max": 14.2e12, "M3 Pro": 7.0e12, "M3": 4.1e12,
    "M4 Max": 16.0e12, "M4 Pro": 8.0e12, "M4": 4.6e12,
}


def _lookup_flops(name: str, table: dict, default: float) -> float:
    # Match longest key first to avoid "M1" matching "M1 Pro"
    for key in sorted(table, key=len, reverse=True):
        if key.lower() in name.lower():
            return table[key]
    return default


def _detect_cuda() -> PlatformProfile:
    props = torch.cuda.get_device_properties(0)
    memory_mb = props.total_mem // (1024 * 1024)
    cap = torch.cuda.get_device_capability()
    return PlatformProfile(
        device="cuda",
        device_name=props.name,
        compute_type="gpu_cuda",
        memory_mb=memory_mb,
        usable_memory_mb=int(memory_mb * 0.9),
        num_cores=os.cpu_count() or 1,
        supports_bfloat16=cap >= (8, 0),
        supports_float16=True,
        supports_compile=True,
        peak_flops=_lookup_flops(props.name, _CUDA_FLOPS, 100e12),
    )


def _detect_mps() -> PlatformProfile:
    try:
        total_bytes = int(_sysctl("hw.memsize"))
        memory_mb = total_bytes // (1024 * 1024)
    except Exception:
        memory_mb = 8192

    try:
        cpu_name = _sysctl("machdep.cpu.brand_string")
    except Exception:
        cpu_name = "Apple Silicon"

    return PlatformProfile(
        device="mps",
        device_name=cpu_name,
        compute_type="gpu_mps",
        memory_mb=memory_mb,
        usable_memory_mb=int(memory_mb * 0.6),
        num_cores=os.cpu_count() or 1,
        supports_bfloat16=True,
        supports_float16=True,
        supports_compile=False,
        peak_flops=_lookup_flops(cpu_name, _MPS_FLOPS, 3.6e12),
    )


def _detect_cpu() -> PlatformProfile:
    if sys.platform == "darwin":
        try:
            memory_mb = int(_sysctl("hw.memsize")) // (1024 * 1024)
        except Exception:
            memory_mb = 4096
        try:
            cpu_name = _sysctl("machdep.cpu.brand_string")
        except Exception:
            cpu_name = platform_mod.processor() or "Unknown CPU"
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        memory_mb = int(line.split()[1]) // 1024
                        break
                else:
                    memory_mb = 4096
        except Exception:
            memory_mb = 4096
        cpu_name = platform_mod.processor() or "Unknown CPU"

    return PlatformProfile(
        device="cpu",
        device_name=cpu_name,
        compute_type="cpu",
        memory_mb=memory_mb,
        usable_memory_mb=int(memory_mb * 0.5),
        num_cores=os.cpu_count() or 1,
        supports_bfloat16=False,
        supports_float16=False,
        supports_compile=True,
        peak_flops=50e9,
    )


def detect_platform(device_override: str | None = None) -> PlatformProfile:
    """Auto-detect hardware and return a PlatformProfile.

    Override with AR_DEVICE env var or device_override parameter.
    Values: "auto", "cpu", "mps", "cuda", "cuda:0", etc.
    """
    override = device_override or os.environ.get("AR_DEVICE")

    if override and override != "auto":
        if override == "cpu":
            return _detect_cpu()
        elif override == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return _detect_mps()
            raise RuntimeError("MPS requested but not available on this system")
        elif override.startswith("cuda"):
            if torch.cuda.is_available():
                return _detect_cuda()
            raise RuntimeError("CUDA requested but not available on this system")

    if torch.cuda.is_available():
        return _detect_cuda()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return _detect_mps()
    else:
        return _detect_cpu()
