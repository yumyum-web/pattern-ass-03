"""
Modular Training & Evaluation Pipeline for Edge CNN Architectures.
Provides functions for training with epoch timing, validation checkpoints,
loss/accuracy tracking, metric calculation (precision, recall, confusion matrix),
and visualization export.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns


def get_default_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Executes one training epoch over the dataloader.
    Returns:
        avg_loss (float): Average training loss.
        accuracy (float): Training classification accuracy in [0, 100].
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Executes one validation/test evaluation pass.
    Returns:
        avg_loss (float): Average loss.
        accuracy (float): Classification accuracy in [0, 100].
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 20,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
    scheduler: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Trains the network for the specified number of epochs, recording per-epoch
    loss, accuracy, and elapsed training duration. Saves the model with the
    highest validation accuracy to `checkpoint_path`.
    """
    if device is None:
        device = get_default_device()

    model = model.to(device)

    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": [],
        "best_epoch": 0,
        "best_val_acc": 0.0,
    }

    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[Training] Starting {num_epochs}-epoch training on device: {device}")
        print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc (%)':^13} | {'Val Loss':^10} | {'Val Acc (%)':^11} | {'Time (s)':^8}")
        print("-" * 72)

    total_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_duration)

        is_best = val_acc > history["best_val_acc"]
        if is_best:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch
            if checkpoint_path:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }, checkpoint_path)

        if verbose:
            flag = " *" if is_best else ""
            print(f"{epoch:^7d} | {train_loss:^11.4f} | {train_acc:^13.2f} | {val_loss:^10.4f} | {val_acc:^11.2f} | {epoch_duration:^8.2f}{flag}")

    total_training_time = time.time() - total_start_time
    history["total_training_time"] = total_training_time
    history["mean_epoch_time"] = float(np.mean(history["epoch_times"]))

    if verbose:
        print("-" * 72)
        print(f"[Training Complete] Best Val Acc: {history['best_val_acc']:.2f}% at Epoch {history['best_epoch']}")
        print(f"Total Time: {total_training_time:.2f}s | Mean Epoch Time: {history['mean_epoch_time']:.2f}s/epoch\n")

    return history


@torch.no_grad()
def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates the model on the unseen test dataset.
    Computes overall accuracy, macro & weighted precision, recall, F1,
    and returns predictions and ground truth labels for confusion matrix generation.
    """
    if device is None:
        device = get_default_device()

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []

    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "test_accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
    }


def plot_training_curves(
    history: Dict[str, Any],
    save_path: str,
    title_suffix: str = "Model A (Standard CNN)",
) -> None:
    """
    Generates high-resolution training and validation loss and accuracy curves.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=1.8, markersize=4)
    ax1.plot(epochs, history["val_loss"], "s--", color="#d62728", label="Val Loss", linewidth=1.8, markersize=4)
    ax1.set_title(f"Loss Convergence: {title_suffix}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.9)

    # Accuracy subplot
    ax2.plot(epochs, history["train_acc"], "o-", color="#2ca02c", label="Train Accuracy", linewidth=1.8, markersize=4)
    ax2.plot(epochs, history["val_acc"], "s--", color="#ff7f0e", label="Val Accuracy", linewidth=1.8, markersize=4)
    ax2.set_title(f"Classification Accuracy: {title_suffix}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    classes: List[str],
    save_path: str,
    title: str = "Model A Test Confusion Matrix (9 Classes)",
) -> None:
    """
    Plots a professional, annotated 9x9 confusion matrix heatmap.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="#dddddd",
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, labelpad=8)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
