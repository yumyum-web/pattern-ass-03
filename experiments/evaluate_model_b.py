"""
Experiment: Train Model B (Depthwise Separable CNN) to convergence and evaluate
it on the held-out test set.

Required by Assignment Section 4: "Evaluate both models on the test set and
record the accuracy, confusion matrix, precision, and recall." The optimizer
ablation in `train_optimizers.py` deliberately reports validation metrics only
(it exists to compare optimizers, not to produce final numbers), so this script
supplies Model B's final test-set results.

Uses the optimizer selected by that ablation: Adam, lr=0.001, no weight decay,
no LR schedule -- matching Model A's recipe so the Section 4 comparison varies
only the architecture.

Outputs (mirroring Model A's artefacts so the two are directly comparable):
    weights/model_b_best.pt                   best-validation checkpoint
    figures/model_b_loss.pdf / .png           train/val loss & accuracy curves
    figures/model_b_confusion_matrix.pdf/.png 9x9 test confusion matrix
    results/model_b_metrics.json              metrics, same schema as Model A
    experiments/results/model_b_metrics.json  (duplicate; both paths are in use)

Note on timing: `epoch_times` is wall-clock and therefore sensitive to whatever
else the machine is doing. The JSON records mean, median and min; prefer MEDIAN
for the Section 4 comparison table, and ideally time every model in one sitting
on one machine so the rows are comparable.

Note on units: accuracies in the emitted JSON are PERCENTAGES (0-100), matching
`results/model_a_metrics.json`, so the comparison table can read both files
without a unit conversion. (`optimizer_histories.json` from the ablation stores
fractions; it is a separate artefact and is left unchanged.)

Usage:
    uv run python -m experiments.evaluate_model_b [--epochs 20] [--num-workers 0]
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: no GUI backend required
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data import CLASSES, get_realwaste_dataloaders
from src.models.model_b import ModelB, count_trainable_parameters


def set_seed(seed: int = 42) -> None:
    """Seeds every RNG that affects training, for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Selects the best available compute device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, float]:
    """One train (optimizer given) or eval (optimizer=None) pass. Accuracy in [0, 100]."""
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

    return total_loss / total, correct / total * 100.0


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred) over the whole loader."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1)
        y_pred.extend(preds.cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())

    return np.array(y_true), np.array(y_pred)


def plot_curves(history: Dict[str, List[float]], base_path: Path) -> None:
    """Exports train/val loss and accuracy curves as both PDF and PNG."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=1.8, markersize=4)
    ax1.plot(epochs, history["val_loss"], "s--", color="#d62728", label="Val Loss", linewidth=1.8, markersize=4)
    ax1.set_title("Loss Convergence: Model B (Depthwise Separable CNN)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)

    ax2.plot(epochs, history["train_acc"], "o-", color="#2ca02c", label="Train Accuracy", linewidth=1.8, markersize=4)
    ax2.plot(epochs, history["val_acc"], "s--", color="#ff7f0e", label="Val Accuracy", linewidth=1.8, markersize=4)
    ax2.set_title("Classification Accuracy: Model B (Depthwise Separable CNN)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(base_path.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: List[str], base_path: Path) -> None:
    """Exports an annotated 9x9 confusion matrix heatmap as both PDF and PNG."""
    base_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar=True, ax=ax, linewidths=0.5, linecolor="#dddddd",
    )
    ax.set_title("Model B Test Confusion Matrix (9 Classes)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, labelpad=8)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(base_path.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test-evaluate Model B on RealWaste 64x64")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="weights/model_b_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    num_classes = len(CLASSES)

    print("=" * 72, flush=True)
    print("Model B (Depthwise Separable CNN) -- Training & Test Evaluation", flush=True)
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}", flush=True)
    print("=" * 72, flush=True)

    train_loader, val_loader, test_loader, _ = get_realwaste_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        download=True,
    )
    print(f"  Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | Test: {len(test_loader.dataset):,}", flush=True)

    model = ModelB(num_classes=num_classes).to(device)
    total_params = count_trainable_parameters(model)
    print(f"  Trainable parameters: {total_params:,} (budget: 100,000)", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_times": []}
    best_val_acc, best_epoch = 0.0, 0
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Time (s)':^8}", flush=True)
    print("-" * 72, flush=True)

    start_all = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(elapsed)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, args.checkpoint)

        print(f"{epoch:^7d} | {train_loss:^11.4f} | {train_acc:^10.2f} | {val_loss:^10.4f} | {val_acc:^9.2f} | {elapsed:^8.2f}{' *' if is_best else ''}", flush=True)

    total_time = time.time() - start_all
    print("-" * 72, flush=True)
    print(f"[Training Complete] Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}", flush=True)
    _times = history["epoch_times"]
    print(f"Epoch time: median {np.median(_times):.2f}s | mean {np.mean(_times):.2f}s | min {np.min(_times):.2f}s", flush=True)
    if np.max(_times) > 3 * np.median(_times):
        print("  WARNING: epoch times vary >3x -- the machine was contended during training.", flush=True)
        print("  Quote median_epoch_time (not mean) in the comparison table.", flush=True)

    # Evaluate the BEST checkpoint on the test set (matching Model A's protocol)
    print(f"\n[Evaluation] Loading best checkpoint from epoch {best_epoch} and scoring the test set...", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = predict(model, test_loader, device)
    test_acc = (y_true == y_pred).mean() * 100.0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("-" * 72, flush=True)
    print(f"Test Accuracy:      {test_acc:.2f}%  ({int((y_true == y_pred).sum())} / {len(y_true)} correct)", flush=True)
    print(f"Macro Precision:    {p_macro:.4f}", flush=True)
    print(f"Macro Recall:       {r_macro:.4f}", flush=True)
    print(f"Macro F1-Score:     {f1_macro:.4f}", flush=True)
    print(f"Weighted F1-Score:  {f1_weighted:.4f}", flush=True)
    print("-" * 72, flush=True)

    plot_curves(history, Path("figures/model_b_loss"))
    plot_confusion(cm, list(CLASSES), Path("figures/model_b_confusion_matrix"))
    print("  Saved figures/model_b_loss.{pdf,png} and figures/model_b_confusion_matrix.{pdf,png}", flush=True)

    results: Dict[str, Any] = {
        "model_name": "Model B (Depthwise Separable CNN)",
        "num_classes": num_classes,
        "classes": list(CLASSES),
        "parameter_counts": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": total_params,
            "non_trainable_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "Adam",
            "scheduler": "None",
            "seed": args.seed,
        },
        "training_history": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
            "epoch_times": history["epoch_times"],
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "mean_epoch_time": float(np.mean(history["epoch_times"])),
            "median_epoch_time": float(np.median(history["epoch_times"])),
            "min_epoch_time": float(np.min(history["epoch_times"])),
            "total_training_time": total_time,
        },
        "test_metrics": {
            "accuracy": float(test_acc),
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix": cm.tolist(),
        },
    }

    for out_dir in ("results", "experiments/results"):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(out_dir, "model_b_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"  Saved {path}", flush=True)

    print("\n✅ Model B training and test evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
