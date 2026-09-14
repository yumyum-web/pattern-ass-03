"""
Optimizer ablation study for Model B (Assignment 03, Section 3: Optimizer
Selection & Tuning, 15 marks).

Trains the same Model B architecture from three independent random
initializations under three optimizer configurations:

    1. Adam        (lr=0.001)                  -- our selected optimizer
    2. SGD          (lr=0.01, momentum=0.0)     -- baseline, no momentum
    3. SGD+Momentum (lr=0.01, momentum=0.9)     -- momentum comparison

for >=20 epochs each, and reports:
    - per-epoch train/val loss & accuracy history (JSON, for the report)
    - overlaid validation loss + validation accuracy curves (PDF figure)
    - a convergence/final-performance comparison table (console + Markdown)

Usage:
    uv run python experiments/train_optimizers.py [--epochs 20] [--batch-size 64]

This script is self-contained (it does not depend on src/utils/trainer.py,
which lives on a different feature branch) so it can be developed and run
independently of the rest of the team's pipeline.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless: no GUI backend required
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data import get_realwaste_dataloaders
from src.models.model_b import ModelB, count_trainable_parameters

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("experiments/results")

OPTIMIZER_CONFIGS: Dict[str, Dict] = {
    "Adam": {"lr": 0.001, "momentum": None},
    "SGD": {"lr": 0.01, "momentum": 0.0},
    "SGD+Momentum": {"lr": 0.01, "momentum": 0.9},
}


def get_device() -> torch.device:
    """Selects the best available compute device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(name: str, model: nn.Module) -> torch.optim.Optimizer:
    cfg = OPTIMIZER_CONFIGS[name]
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> tuple:
    """Runs one train (optimizer given) or eval (optimizer=None) epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_one_config(
    optimizer_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    seed: int = 42,
) -> Dict:
    """Trains Model B from scratch under a single optimizer configuration."""
    torch.manual_seed(seed)  # identical initial weights across configs -> fair comparison

    model = ModelB(num_classes=9).to(device)
    optimizer = build_optimizer(optimizer_name, model)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time_sec": []}

    print(f"\n=== Training Model B with optimizer: {optimizer_name} "
          f"(lr={OPTIMIZER_CONFIGS[optimizer_name]['lr']}, "
          f"momentum={OPTIMIZER_CONFIGS[optimizer_name]['momentum']}) ===")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_sec"].append(elapsed)

        print(f"  Epoch {epoch:2d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

    return history


def epochs_to_reach_accuracy(val_acc_history: List[float], target: float) -> int:
    """First 1-indexed epoch at which val_acc >= target, or -1 if never reached."""
    for i, acc in enumerate(val_acc_history, start=1):
        if acc >= target:
            return i
    return -1


def plot_comparison(histories: Dict[str, Dict], out_path: Path) -> None:
    """Overlays validation loss and validation accuracy curves for all optimizers."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, history in histories.items():
        epochs = range(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], marker="o", markersize=3, label=name)
        axes[1].plot(epochs, history["val_acc"], marker="o", markersize=3, label=name)

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Model B Optimizer Ablation: Adam vs SGD vs SGD+Momentum")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nSaved comparison figure to {out_path}")


def print_comparison_table(histories: Dict[str, Dict], target_acc: float = 0.5) -> None:
    """Prints a console + Markdown table of convergence speed and final performance."""
    rows = []
    for name, history in histories.items():
        final_val_acc = history["val_acc"][-1]
        best_val_acc = max(history["val_acc"])
        avg_epoch_time = sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"])
        conv_epoch = epochs_to_reach_accuracy(history["val_acc"], target_acc)
        rows.append((name, final_val_acc, best_val_acc, conv_epoch, avg_epoch_time))

    header = f"{'Optimizer':<14} | {'Final Val Acc':>14} | {'Best Val Acc':>13} | {'Epoch to ' + str(int(target_acc*100)) + '%':>12} | {'Avg s/epoch':>11}"
    print("\n" + header)
    print("-" * len(header))
    for name, final_acc, best_acc, conv_epoch, avg_time in rows:
        conv_str = str(conv_epoch) if conv_epoch != -1 else "not reached"
        print(f"{name:<14} | {final_acc:>14.4f} | {best_acc:>13.4f} | {conv_str:>12} | {avg_time:>11.2f}")

    md_lines = [
        "| Optimizer | Final Val Acc | Best Val Acc | Epoch to " + f"{int(target_acc*100)}%" + " | Avg s/epoch |",
        "|---|---|---|---|---|",
    ]
    for name, final_acc, best_acc, conv_epoch, avg_time in rows:
        conv_str = str(conv_epoch) if conv_epoch != -1 else "not reached"
        md_lines.append(f"| {name} | {final_acc:.4f} | {best_acc:.4f} | {conv_str} | {avg_time:.2f} |")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / "optimizer_comparison_table.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nSaved Markdown comparison table to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model B optimizer ablation study")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _test_loader, _class_to_idx = get_realwaste_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    probe_model = ModelB(num_classes=9)
    print(f"Model B trainable parameters: {count_trainable_parameters(probe_model):,} (budget: 100,000)")

    histories: Dict[str, Dict] = {}
    for optimizer_name in OPTIMIZER_CONFIGS:
        histories[optimizer_name] = train_one_config(
            optimizer_name, train_loader, val_loader, device, epochs=args.epochs, seed=args.seed,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "optimizer_histories.json", "w") as f:
        json.dump(histories, f, indent=2)
    print(f"\nSaved raw per-epoch histories to {RESULTS_DIR / 'optimizer_histories.json'}")

    plot_comparison(histories, FIGURES_DIR / "optimizer_comparison.pdf")
    print_comparison_table(histories)


if __name__ == "__main__":
    main()
