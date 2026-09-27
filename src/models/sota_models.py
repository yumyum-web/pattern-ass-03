"""
SOTA Transfer Learning Wrappers for RealWaste 9-class classification.

Provides factory functions for two lightweight pretrained architectures:
  - MobileNetV2  (~2.2M–3.5M params): Inverted residual blocks with linear bottlenecks.
  - ShuffleNetV2 (~1.4M–2.3M params): Channel-split + channel-shuffle, optimised for MAC cost.

Both models are initialised with ImageNet pretrained weights and their final
classification heads are replaced with a linear layer outputting 9 logits
(one per RealWaste class).

The 64×64 input resolution used in this project is intentionally smaller than
the ImageNet default (224×224), but both architectures employ global average
pooling before the head, so they accept arbitrary spatial input without
modification.

Classes (9, canonical ordering):
    0: Cardboard          1: Food Organics      2: Glass
    3: Metal              4: Miscellaneous Trash 5: Paper
    6: Plastic            7: Textile Trash       8: Vegetation

Usage
-----
>>> from src.models.sota_models import get_mobilenet_v2, get_shufflenet_v2
>>> model = get_mobilenet_v2(num_classes=9)
>>> model = get_shufflenet_v2(num_classes=9)
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


# ---------------------------------------------------------------------------
# MobileNetV2
# ---------------------------------------------------------------------------

def get_mobilenet_v2(num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    """
    Returns a MobileNetV2 with its classifier head replaced for `num_classes` outputs.

    Architecture summary (from torchvision):
        - 19 bottleneck residual blocks (inverted residuals with linear bottlenecks)
        - Depthwise separable convolutions throughout
        - Global Average Pooling → 1280-dim feature vector
        - Original classifier: Linear(1280, 1000)   ← replaced below

    Modification:
        model.classifier = Sequential(Dropout(0.2), Linear(1280, num_classes))

    Args:
        num_classes: Number of output logits (default 9 for RealWaste).
        pretrained:  Load ImageNet-1k pretrained weights (default True).

    Returns:
        nn.Module: Modified MobileNetV2 ready for fine-tuning.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Replace the final classifier head
    # Original: Sequential(Dropout(0.2), Linear(1280, 1000))
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


# ---------------------------------------------------------------------------
# ShuffleNetV2
# ---------------------------------------------------------------------------

def get_shufflenet_v2(num_classes: int = 9, pretrained: bool = True, width_mult: str = "1_0") -> nn.Module:
    """
    Returns a ShuffleNetV2 (1.0× width) with its FC head replaced for `num_classes` outputs.

    Architecture summary (from torchvision):
        - Channel-split and channel-shuffle operations to minimise Memory Access Cost (MAC)
        - Pointwise group convolutions throughout
        - Global Average Pooling → 1024-dim feature vector (1.0× variant)
        - Original classifier: Linear(1024, 1000)   ← replaced below

    Modification:
        model.fc = Linear(1024, num_classes)

    Args:
        num_classes: Number of output logits (default 9 for RealWaste).
        pretrained:  Load ImageNet-1k pretrained weights (default True).
        width_mult:  Width multiplier variant — "0_5" (0.5×) or "1_0" (1.0×).
                     Affects channel count and parameter budget.

    Returns:
        nn.Module: Modified ShuffleNetV2 ready for fine-tuning.
    """
    if width_mult == "0_5":
        weights = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.shufflenet_v2_x0_5(weights=weights)
    else:  # default: 1.0×
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)

    # Replace the final fully-connected head
    # Original: Linear(1024, 1000)  for 1.0×  |  Linear(1024, 1000)  for 0.5×
    in_features = model.fc.in_features  # 1024 for 1.0×, 1024 for 0.5×
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Counts trainable and total parameters of a model.

    Args:
        model: Any nn.Module instance.

    Returns:
        (trainable_params, total_params): Tuple of integers.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_model_size_mb(model: nn.Module) -> float:
    """
    Estimates the on-disk size of a model's state_dict in megabytes.

    This matches the size of the `.pth` file produced by
    `torch.save(model.state_dict(), path)`.

    Args:
        model: Any nn.Module instance.

    Returns:
        float: Estimated disk size in MB (float, 2 decimal places).
    """
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    return round(size_bytes / (1024 ** 2), 2)


def print_model_summary(model: nn.Module, model_name: str) -> None:
    """
    Prints a concise summary: parameter counts and estimated disk size.

    Args:
        model:      nn.Module to summarise.
        model_name: Human-readable label printed in the header.
    """
    trainable, total = count_parameters(model)
    size_mb = get_model_size_mb(model)
    frozen = total - trainable

    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}")
    print(f"  Trainable parameters : {trainable:>12,}")
    print(f"  Frozen parameters    : {frozen:>12,}")
    print(f"  Total parameters     : {total:>12,}")
    print(f"  Estimated disk size  : {size_mb:>11.2f} MB")
    print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SOTA Transfer Learning Model Verification ===")
    print("Loading pretrained weights (downloads on first run)...\n")

    # MobileNetV2
    mobilenet = get_mobilenet_v2(num_classes=9, pretrained=True)
    mobilenet.eval()
    print_model_summary(mobilenet, "MobileNetV2 (fine-tune head for 9 classes)")

    # Verify forward pass with 64×64 input
    dummy = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        out = mobilenet(dummy)
    assert out.shape == (2, 9), f"Expected (2, 9), got {out.shape}"
    print(f"  Forward pass (2×3×64×64) → output shape: {tuple(out.shape)}  ✅")

    # ShuffleNetV2 (1.0×)
    shufflenet = get_shufflenet_v2(num_classes=9, pretrained=True, width_mult="1_0")
    shufflenet.eval()
    print_model_summary(shufflenet, "ShuffleNetV2 ×1.0 (fine-tune head for 9 classes)")

    with torch.no_grad():
        out = shufflenet(dummy)
    assert out.shape == (2, 9), f"Expected (2, 9), got {out.shape}"
    print(f"  Forward pass (2×3×64×64) → output shape: {tuple(out.shape)}  ✅")

    print("\n✅ All SOTA model wrappers verified successfully.")
