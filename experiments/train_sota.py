"""
experiments/train_sota.py
─────────────────────────
Fine-tunes MobileNetV2 and ShuffleNetV2 (1.0×) on the RealWaste 64×64 dataset
using the identical 70/15/15 deterministic split (seed=42) defined in src/data.py.

Training protocol
-----------------
- Optimiser   : Adam (lr=0.001, weight_decay=1e-4)
- Scheduler   : CosineAnnealingLR (T_max=20)
- Loss        : CrossEntropyLoss
- Epochs      : 20 per model
- Device      : CUDA if available, else CPU
- Batch size  : 32

Outputs
-------
  weights/mobilenet_v2.pth           — saved best-val-acc checkpoint
  weights/shufflenet_v2.pth          — saved best-val-acc checkpoint
  figures/sota_loss_curves.pdf       — training & validation loss/accuracy curves
  figures/confusion_matrices.pdf     — 9×9 normalised confusion matrices (test set)

Run
---
  uv run python experiments/train_sota.py
"""

import os
import sys
import time
from typing import Dict, List, Tuple

# Allow running from repo root: `uv run python experiments/train_sota.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import get_realwaste_dataloaders, CLASSES
from src.models.sota_models import (
    get_mobilenet_v2,
    get_shufflenet_v2,
    count_parameters,
    get_model_size_mb,
    print_model_summary,
)
from src.utils.metrics import (
    collect_predictions,
    compute_metrics,
    print_metrics,
    save_confusion_matrices,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
EPOCHS       = 20
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
DATA_DIR     = "data"
WEIGHTS_DIR  = "weights"
FIGURES_DIR  = "figures"
SEED         = 42

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Runs one training epoch.

    Returns:
        (avg_loss, accuracy): Floats for average cross-entropy loss and top-1 accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates model on `loader` without gradient computation.

    Returns:
        (avg_loss, accuracy): Floats.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return running_loss / total, correct / total


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
) -> Dict[str, List[float]]:
    """
    Full training loop for `model` over `EPOCHS` epochs.
    Saves the checkpoint with the best validation accuracy.

    Args:
        model:        nn.Module to train (already on CPU; moved inside).
        model_name:   Display name for logging.
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader.
        device:       Compute device.
        save_path:    Path to save the best model state_dict (.pth).

    Returns:
        history dict with keys: train_loss, val_loss, train_acc, val_acc.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_acc = 0.0

    print(f"\n{'─' * 60}")
    print(f"  Training: {model_name}")
    print(f"  Device  : {device} | Epochs: {EPOCHS} | LR: {LR}")
    print(f"{'─' * 60}")
    print(f"  {'Ep':>3}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>9}  {'ValAcc':>8}  {'Time':>6}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*6}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"  {epoch:>3}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%"
            f"  {val_loss:>9.4f}  {val_acc*100:>7.2f}%  {elapsed:>5.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n  ✅ Best val accuracy: {best_val_acc * 100:.2f}% — checkpoint saved to '{save_path}'")
    return history


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: str = "figures/sota_loss_curves.pdf",
) -> None:
    """
    Plots training & validation loss and accuracy curves for each model side-by-side.

    Args:
        histories: Dict mapping model name → history dict from train_model().
        save_path: Output PDF path.
    """
    model_names = list(histories.keys())
    n = len(model_names)
    epochs = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = [axes]  # make iterable

    for row, name in enumerate(model_names):
        h = histories[name]

        # Loss
        ax_loss = axes[row][0]
        ax_loss.plot(epochs, h["train_loss"], label="Train", linewidth=1.8, color="#2196F3")
        ax_loss.plot(epochs, h["val_loss"],   label="Val",   linewidth=1.8, color="#F44336", linestyle="--")
        ax_loss.set_title(f"{name} — Loss Curve", fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Cross-Entropy Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        # Accuracy
        ax_acc = axes[row][1]
        ax_acc.plot(epochs, [a * 100 for a in h["train_acc"]], label="Train", linewidth=1.8, color="#4CAF50")
        ax_acc.plot(epochs, [a * 100 for a in h["val_acc"]],   label="Val",   linewidth=1.8, color="#FF9800", linestyle="--")
        ax_acc.set_title(f"{name} — Accuracy Curve", fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(0, 100)
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)

    fig.suptitle("SOTA Transfer Learning — Training Curves\nRealWaste 64×64", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Plot] Training curves saved to '{save_path}'")


def print_final_summary(results: List[Tuple]) -> None:
    """
    Prints a Markdown comparison table of all evaluated models.

    Args:
        results: List of (name, preds, targets, trainable_params, size_mb, avg_epoch_time_s) tuples.
    """
    print("\n" + "=" * 80)
    print("  SOTA MODEL COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Model':<20} {'Params':>10} {'Size(MB)':>10} {'Epoch(s)':>9} {'TestAcc':>9} {'MacroPrec':>10} {'MacroRec':>10}"
    print(header)
    print("-" * 80)

    for (name, preds, targets, params, size_mb, avg_time) in results:
        acc, prec, rec = compute_metrics(preds, targets)
        print(
            f"  {name:<18} {params:>10,} {size_mb:>10.2f} {avg_time:>9.1f}"
            f"  {acc*100:>7.2f}%  {prec*100:>9.2f}%  {rec*100:>9.2f}%"
        )
    print("=" * 80)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[Data] Loading RealWaste 64×64 dataloaders...")
    train_loader, val_loader, test_loader, _ = get_realwaste_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=64,
        seed=SEED,
        num_workers=NUM_WORKERS,
        download=True,
    )
    print(
        f"       Train: {len(train_loader.dataset):,} | "
        f"Val: {len(val_loader.dataset):,} | "
        f"Test: {len(test_loader.dataset):,}"
    )

    # ── Model definitions ─────────────────────────────────────────────────
    model_configs = [
        {
            "name":       "MobileNetV2",
            "model":      get_mobilenet_v2(num_classes=9, pretrained=True),
            "save_path":  os.path.join(WEIGHTS_DIR, "mobilenet_v2.pth"),
        },
        {
            "name":       "ShuffleNetV2-1.0×",
            "model":      get_shufflenet_v2(num_classes=9, pretrained=True, width_mult="1_0"),
            "save_path":  os.path.join(WEIGHTS_DIR, "shufflenet_v2.pth"),
        },
    ]

    # Print architecture summaries before training
    for cfg in model_configs:
        print_model_summary(cfg["model"], cfg["name"])

    # ── Training ──────────────────────────────────────────────────────────
    histories: Dict[str, Dict[str, List[float]]] = {}

    for cfg in model_configs:
        t_start = time.time()
        history = train_model(
            model=cfg["model"],
            model_name=cfg["name"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_path=cfg["save_path"],
        )
        t_total = time.time() - t_start
        cfg["avg_epoch_time"] = t_total / EPOCHS
        histories[cfg["name"]] = history

    # Plot training curves
    plot_training_curves(histories, save_path=os.path.join(FIGURES_DIR, "sota_loss_curves.pdf"))

    # ── Evaluation on test set ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)

    confusion_data = []
    summary_rows   = []

    for cfg in model_configs:
        # Reload best checkpoint
        cfg["model"].load_state_dict(torch.load(cfg["save_path"], map_location=device))
        cfg["model"] = cfg["model"].to(device)

        preds, targets = collect_predictions(cfg["model"], test_loader, device)
        print_metrics(preds, targets, model_name=cfg["name"])

        confusion_data.append((cfg["name"], preds, targets))

        trainable, _ = count_parameters(cfg["model"])
        size_mb = get_model_size_mb(cfg["model"])
        summary_rows.append((
            cfg["name"], preds, targets,
            trainable, size_mb, cfg["avg_epoch_time"],
        ))

    # Save confusion matrices
    save_confusion_matrices(
        confusion_data,
        save_path=os.path.join(FIGURES_DIR, "confusion_matrices.pdf"),
    )

    # Final comparison table
    print_final_summary(summary_rows)

    print("\n✅ train_sota.py complete. All outputs saved to figures/ and weights/.")


if __name__ == "__main__":
    main()
