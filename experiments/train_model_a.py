"""
Experiment: Train Model A (Standard 2D CNN Baseline) on RealWaste 64x64.
Trains for 20 epochs using the Adam optimizer with cosine learning rate schedule,
evaluates on the unseen test set, exports loss/accuracy curves and confusion matrix,
and serializes evaluation metrics to JSON.
"""

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_realwaste_dataloaders, CLASSES
from src.models import create_model_a
from src.utils.trainer import (
    train_model,
    evaluate_test_set,
    plot_training_curves,
    plot_confusion_matrix_heatmap,
    get_default_device,
)


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic execution across CPU/GPU and random libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiment(
    data_dir: str = "data",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    figures_dir: str = "figures",
    results_dir: str = "results",
    checkpoints_dir: str = "checkpoints",
) -> None:
    """Executes the full Model A training and evaluation workflow."""
    set_seed(seed)
    device = get_default_device()
    print("=" * 72)
    print(f"Model A (Standard CNN Baseline) Training Pipeline")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 72)

    # 1. Data Loading
    print("\n[Step 1/5] Loading RealWaste 64x64 dataset splits...")
    train_loader, val_loader, test_loader, class_to_idx = get_realwaste_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=64,
        seed=seed,
        num_workers=2,
        download=False,
    )
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples:   {len(val_loader.dataset):,}")
    print(f"  Test samples:  {len(test_loader.dataset):,}")

    # 2. Model Initialization
    print("\n[Step 2/5] Initializing Model A architecture...")
    model = create_model_a(num_classes=len(CLASSES), in_channels=3)
    param_counts = model.count_parameters()
    print(f"  Total Parameters:     {param_counts['total_parameters']:,}")
    print(f"  Trainable Parameters: {param_counts['trainable_parameters']:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Model Training
    print("\n[Step 3/5] Commencing 20-epoch training loop...")
    checkpoint_path = os.path.join(checkpoints_dir, "model_a_best.pt")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
        device=device,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        verbose=True,
    )

    # 4. Evaluation on Test Set
    print("\n[Step 4/5] Evaluating best checkpoint on unseen Test Set...")
    test_metrics = evaluate_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    print("-" * 72)
    print(f"Test Accuracy:         {test_metrics['test_accuracy']:.2f}%")
    print(f"Macro Precision:       {test_metrics['precision_macro']:.4f}")
    print(f"Macro Recall:          {test_metrics['recall_macro']:.4f}")
    print(f"Macro F1-Score:        {test_metrics['f1_macro']:.4f}")
    print(f"Weighted F1-Score:     {test_metrics['f1_weighted']:.4f}")
    print("-" * 72)

    # 5. Exporting Visualizations and Metrics
    print("\n[Step 5/5] Exporting figures and result metrics...")
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Loss & Accuracy Curves
    loss_curve_path = os.path.join(figures_dir, "model_a_loss_curve.png")
    plot_training_curves(history, loss_curve_path, title_suffix="Model A (Standard CNN)")
    print(f"  Saved training curves to: {loss_curve_path}")

    # Confusion Matrix Heatmap
    cm_path = os.path.join(figures_dir, "model_a_confusion_matrix.png")
    plot_confusion_matrix_heatmap(
        cm=test_metrics["confusion_matrix"],
        classes=list(CLASSES),
        save_path=cm_path,
        title="Model A Test Confusion Matrix (9 Classes)",
    )
    print(f"  Saved confusion matrix to: {cm_path}")

    # Serialize JSON Metrics
    serializable_results = {
        "model_name": "Model A (Standard CNN)",
        "num_classes": len(CLASSES),
        "classes": list(CLASSES),
        "parameter_counts": param_counts,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "seed": seed,
        },
        "training_history": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
            "epoch_times": history["epoch_times"],
            "best_epoch": history["best_epoch"],
            "best_val_acc": history["best_val_acc"],
            "mean_epoch_time": history["mean_epoch_time"],
            "total_training_time": history["total_training_time"],
        },
        "test_metrics": {
            "accuracy": test_metrics["test_accuracy"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"],
            "f1_macro": test_metrics["f1_macro"],
            "precision_weighted": test_metrics["precision_weighted"],
            "recall_weighted": test_metrics["recall_weighted"],
            "f1_weighted": test_metrics["f1_weighted"],
            "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        },
    }

    metrics_json_path = os.path.join(results_dir, "model_a_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"  Saved JSON metrics to: {metrics_json_path}")
    print("\n✅ Model A training experiment completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model A on RealWaste 64x64")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--figures-dir", type=str, default="figures", help="Directory for output figures")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for JSON results")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    args = parser.parse_args()

    run_experiment(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        figures_dir=args.figures_dir,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
    )
