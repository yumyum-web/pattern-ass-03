"""
Cross-Architecture Benchmarking Suite.

Profiles all four project models (Model A, Model B, MobileNetV2, ShuffleNetV2)
across five hardware-relevant metrics and outputs comparison tables in both
Markdown and LaTeX formats. Also generates a trade-off scatter plot.

Usage (run from repo root):
    python experiments/benchmark_all.py

Colab usage:
    !pip install torchinfo
    !python experiments/benchmark_all.py
"""

import os
import sys

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.model_a import create_model_a
from src.models.model_b import ModelB
from src.models.sota_models import get_mobilenet_v2, get_shufflenet_v2
from src.utils.profiler import (
    get_parameter_count,
    get_model_size,
    get_flops_and_macs,
    measure_inference_latency,
)


def benchmark_model(model_name, model, input_size=(1, 3, 64, 64)):
    """Profiles a single model across all hardware metrics."""
    print(f"  Benchmarking {model_name}...")

    trainable, total = get_parameter_count(model)
    size_kb, size_mb = get_model_size(model)
    macs, flops = get_flops_and_macs(model, input_size)
    latency_ms = measure_inference_latency(model, input_size)

    return {
        "Model": model_name,
        "Trainable Params": trainable,
        "Total Params": total,
        "Size (KB)": size_kb,
        "Size (MB)": size_mb,
        "MACs": macs,
        "FLOPs": flops,
        "Latency (ms)": latency_ms,
    }


def print_markdown_table(results):
    """Prints a clean Markdown comparison table."""
    print("\n### Cross-Architecture Hardware Comparison\n")
    print("| Model | Trainable Params | Size (MB) | MACs | Latency (ms) |")
    print("| :--- | ---: | ---: | ---: | ---: |")
    for r in results:
        macs_str = f"{r['MACs']:,}" if isinstance(r["MACs"], int) else r["MACs"]
        print(
            f"| {r['Model']} "
            f"| {r['Trainable Params']:,} "
            f"| {r['Size (MB)']:.2f} "
            f"| {macs_str} "
            f"| {r['Latency (ms)']:.2f} |"
        )


def print_latex_table(results):
    """Prints a LaTeX tabular block for the report."""
    print("\n% --- Paste this into report/Outliers_A03_EN3150.tex ---")
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\caption{Cross-Architecture Hardware Profiling Comparison}")
    print("\\label{tab:profiling}")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Model & Trainable Params & Size (MB) & MACs & Latency (ms) \\\\")
    print("\\midrule")
    for r in results:
        macs_str = f"{r['MACs']:,}" if isinstance(r["MACs"], int) else r["MACs"]
        print(
            f"{r['Model']} & {r['Trainable Params']:,} & {r['Size (MB)']:.2f} "
            f"& {macs_str} & {r['Latency (ms)']:.2f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def generate_tradeoff_scatter(results, save_path="figures/trade_off_scatter.pdf"):
    """
    Generates an Accuracy-vs-Parameters-vs-Latency scatter plot.
    Note: Test accuracy is not available in this profiling-only script.
    Uses parameter count on x-axis and latency on y-axis, with bubble size
    proportional to model disk size.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names = [r["Model"] for r in results]
    params = [r["Trainable Params"] for r in results]
    latencies = [r["Latency (ms)"] for r in results]
    sizes_mb = [r["Size (MB)"] for r in results]

    # Bubble size proportional to disk footprint (scaled for visibility)
    max_size = max(sizes_mb) if max(sizes_mb) > 0 else 1
    bubble_sizes = [max(80, (s / max_size) * 800) for s in sizes_mb]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for i, (name, p, lat, bs) in enumerate(zip(names, params, latencies, bubble_sizes)):
        ax.scatter(p, lat, s=bs, c=colors[i % len(colors)], alpha=0.7,
                   edgecolors="black", linewidth=0.8, zorder=3)
        ax.annotate(name, (p, lat), textcoords="offset points",
                    xytext=(10, 10), fontsize=8, fontweight="bold")

    ax.set_xlabel("Trainable Parameters", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inference Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Edge Deployment Trade-off: Parameters vs Latency\n(Bubble size ∝ Disk Footprint)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n[Benchmark] Trade-off scatter plot saved to '{save_path}'")


if __name__ == "__main__":
    print("=" * 65)
    print("  Edge Architecture Profiler — Cross-Model Comparison")
    print("=" * 65)

    # Instantiate all 4 project models
    models_to_benchmark = {
        "Model A (Standard CNN)": create_model_a(num_classes=9),
        "Model B (DS-CNN <100k)": ModelB(num_classes=9),
        "MobileNetV2 (SOTA)": get_mobilenet_v2(num_classes=9, pretrained=False),
        "ShuffleNetV2 (SOTA)": get_shufflenet_v2(num_classes=9, pretrained=False),
    }

    print(f"\nProfiling {len(models_to_benchmark)} architectures on input (1, 3, 64, 64)...\n")

    results = []
    for name, model in models_to_benchmark.items():
        results.append(benchmark_model(name, model))

    # Print tables
    print_markdown_table(results)
    print_latex_table(results)

    # Generate scatter plot
    generate_tradeoff_scatter(results)

    # Summary printout
    print("\n" + "=" * 65)
    print("  Quick Summary")
    print("=" * 65)
    for r in results:
        print(f"  {r['Model']:<28s} | {r['Trainable Params']:>10,} params | {r['Size (MB)']:>7.2f} MB | {r['Latency (ms)']:>7.2f} ms")
    print("=" * 65)
    print("\nBenchmarking complete!")
