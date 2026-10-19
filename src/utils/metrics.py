"""
Evaluation metrics for RealWaste 9-class image classification.

Provides:
  - Accuracy, Macro-Precision, Macro-Recall computation
  - 9×9 Normalised Confusion Matrix generation and plotting
  - Per-class breakdown table

Designed to be called after model.eval() on a DataLoader.
"""

from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # headless — no GUI window, safe on servers / HPC / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# Canonical RealWaste class names (matches src/data.py CLASSES ordering)
CLASSES: Tuple[str, ...] = (
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Misc. Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs model inference on all batches in `loader` and collects predictions.

    Args:
        model:  Trained nn.Module (called with model.eval()).
        loader: DataLoader yielding (images, labels) batches.
        device: torch.device to run inference on.

    Returns:
        (all_preds, all_targets): Two 1-D numpy arrays of length N (dataset size).
    """
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy().tolist())

    return np.array(all_preds), np.array(all_targets)


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Computes top-level classification metrics.

    Args:
        preds:   1-D array of predicted class indices.
        targets: 1-D array of ground-truth class indices.

    Returns:
        (accuracy, macro_precision, macro_recall): All as floats in [0, 1].
    """
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average="macro", zero_division=0)
    rec = recall_score(targets, preds, average="macro", zero_division=0)
    return acc, prec, rec


def print_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    model_name: str = "Model",
) -> None:
    """
    Prints a formatted metrics summary and per-class breakdown.

    Args:
        preds:      Predicted class indices.
        targets:    Ground-truth class indices.
        model_name: Label shown in the header.
    """
    acc, prec, rec = compute_metrics(preds, targets)

    print(f"\n{'=' * 55}")
    print(f"  {model_name} — Test Set Evaluation")
    print(f"{'=' * 55}")
    print(f"  Accuracy        : {acc * 100:.2f}%")
    print(f"  Macro-Precision : {prec * 100:.2f}%")
    print(f"  Macro-Recall    : {rec * 100:.2f}%")
    print(f"{'=' * 55}")
    print("\nPer-Class Report:")
    print(classification_report(targets, preds, target_names=CLASSES, zero_division=0))


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    ax: plt.Axes,
    class_names: Tuple[str, ...] = CLASSES,
    colormap: str = "Blues",
) -> None:
    """
    Draws a normalised 9×9 confusion matrix heatmap onto the given Axes object.

    Normalisation: each row is divided by its sum so values represent
    per-class recall (fraction of true samples of that class predicted correctly).

    Args:
        preds:       Predicted class indices (1-D array).
        targets:     Ground-truth class indices (1-D array).
        model_name:  Title shown above the heatmap.
        ax:          Matplotlib Axes to draw on.
        class_names: Ordered sequence of class label strings.
        colormap:    Matplotlib colormap name (default "Blues").
    """
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    # Row-normalise to [0, 1]  (handle zero-count classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div-by-zero for empty classes
    cm_norm = cm.astype(float) / row_sums

    n = len(class_names)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=colormap, vmin=0.0, vmax=1.0)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalised recall", rotation=270, labelpad=14, fontsize=8)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("True label", fontsize=9)
    ax.set_title(model_name, fontsize=10, fontweight="bold", pad=10)

    # Annotate cells with percentage values
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(
                j, i,
                f"{val:.0%}",
                ha="center", va="center",
                fontsize=6,
                color=color,
                fontweight="bold" if i == j else "normal",
            )


def save_confusion_matrices(
    results: List[Tuple[str, np.ndarray, np.ndarray]],
    save_path: str = "figures/confusion_matrices.pdf",
    class_names: Tuple[str, ...] = CLASSES,
) -> None:
    """
    Generates and saves a figure containing one confusion matrix per model.

    Args:
        results:    List of (model_name, preds, targets) tuples — one per model.
        save_path:  Output file path (PDF for vector quality, PNG also accepted).
        class_names: Class label strings.
    """
    n_models = len(results)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(7 * n_models, 7),
        squeeze=False,
    )

    for idx, (model_name, preds, targets) in enumerate(results):
        plot_confusion_matrix(
            preds, targets,
            model_name=model_name,
            ax=axes[0][idx],
            class_names=class_names,
        )

    fig.suptitle(
        "Normalised Confusion Matrices — SOTA Transfer Learning\nRealWaste 64×64 Test Set",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Metrics] Confusion matrices saved to '{save_path}'")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Metrics Module Self-Test ===")
    rng = np.random.default_rng(42)
    n = 713  # test-set size
    fake_targets = rng.integers(0, 9, size=n)
    # Simulate 70% accurate predictions
    fake_preds = fake_targets.copy()
    noise_mask = rng.random(n) > 0.70
    fake_preds[noise_mask] = rng.integers(0, 9, size=noise_mask.sum())

    print_metrics(fake_preds, fake_targets, model_name="Synthetic Sanity Check")

    save_confusion_matrices(
        [("Synthetic Model", fake_preds, fake_targets)],
        save_path="figures/test_confusion.pdf",
    )
    print("\n✅ Metrics module verified successfully.")
