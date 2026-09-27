"""
Edge Hardware Profiler & Metric Evaluation Toolkit.

Provides generic functions to evaluate any PyTorch model's hardware footprint
for edge deployment analysis:
  - Trainable parameter count
  - On-disk model size (KB / MB)
  - Forward-pass MACs and FLOPs via torchinfo
  - Single-image inference latency (ms)

All functions accept any nn.Module, making them model-agnostic. This allows
profiling before all team models are finalized.
"""

import os
import time
import tempfile
from typing import Tuple, Union

import torch
import torch.nn as nn

try:
    from torchinfo import summary
except ImportError:
    summary = None


def get_parameter_count(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (trainable_params, total_params) for the given model.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_size(model: nn.Module) -> Tuple[float, float]:
    """
    Saves the model's state_dict to a temporary file and measures its disk footprint.
    Returns (size_kb, size_mb).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_bytes = os.path.getsize(tmp.name)
    os.remove(tmp.name)

    size_kb = size_bytes / 1024.0
    size_mb = size_kb / 1024.0
    return size_kb, size_mb


def get_flops_and_macs(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 64, 64),
) -> Tuple[Union[int, str], Union[int, str]]:
    """
    Computes MACs and FLOPs for a single forward pass.
    FLOPs ≈ 2 × MACs (standard convention for multiply-accumulate operations).
    Requires `torchinfo` to be installed.
    """
    if summary is None:
        return "N/A (torchinfo not installed)", "N/A"

    model_stats = summary(model, input_size=input_size, verbose=0)
    macs = model_stats.total_mult_adds
    flops = macs * 2
    return macs, flops


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 64, 64),
    device: torch.device = None,
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> float:
    """
    Measures average single-image inference latency in milliseconds.
    Includes GPU synchronization if running on CUDA.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    dummy = torch.randn(*input_size, device=device)

    with torch.no_grad():
        # Warmup passes (let GPU caches and JIT settle)
        for _ in range(warmup_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(timed_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    avg_ms = ((end - start) / timed_runs) * 1000.0
    return avg_ms
