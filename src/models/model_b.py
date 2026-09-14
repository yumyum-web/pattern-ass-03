"""
Model B: Lightweight Depthwise Separable CNN for RealWaste 64x64 classification.

Design goal: strictly < 100,000 trainable parameters (Assignment 03, Section 2,
Model B requirement), suitable for resource-constrained edge devices
(microcontrollers, Raspberry Pi-class hardware).

Architecture
------------
Stem:      Conv2d(3, 24, k=3, s=1, p=1)        + BN(24)  + ReLU6 + MaxPool(2) -> 24 x 32 x 32
DS Block1: Depthwise Conv2d(24, 24, k=3, groups=24) + BN(24)  + ReLU6
           Pointwise Conv2d(24, 48, k=1)             + BN(48)  + ReLU6 + MaxPool(2) -> 48 x 16 x 16
DS Block2: Depthwise Conv2d(48, 48, k=3, groups=48) + BN(48)  + ReLU6
           Pointwise Conv2d(48, 96, k=1)             + BN(96)  + ReLU6 + MaxPool(2) -> 96 x 8 x 8
DS Block3: Depthwise Conv2d(96, 96, k=3, groups=96) + BN(96)  + ReLU6
           Pointwise Conv2d(96, 128, k=1)            + BN(128) + ReLU6 + AdaptiveAvgPool((1,1)) -> 128
Head:      Dropout(0.2) + Linear(128, 9)

All conv layers that feed directly into a BatchNorm use bias=False: BatchNorm's
learned shift (beta) makes an immediately-preceding conv bias mathematically
redundant, so dropping it saves parameters at zero accuracy cost.

Depthwise-separable parameter savings (per block, Cin -> Cout channels, k=3):
    Standard conv:      Cin * Cout * k^2                      params
    Depthwise separable: Cin * k^2 (depthwise) + Cin * Cout (pointwise) params
    e.g. Block 2 (48 -> 96): standard = 48*96*9 = 41,472 vs DS = 48*9 + 48*96 = 4,896
         (~8.5x fewer multiply-accumulate weights in that block alone)

Exact trainable parameter count for this network (computed programmatically,
see `count_trainable_parameters` / `if __name__ == "__main__"` block below):

    Stem   (Conv 3->24, bias=False):      3*24*3*3            =    648
             BatchNorm2d(24):              24*2               =     48
    Block1  Depthwise Conv(24, groups=24): 24*3*3             =    216
             BatchNorm2d(24):               24*2               =     48
             Pointwise Conv(24->48):        24*48              =  1,152
             BatchNorm2d(48):               48*2               =     96
    Block2  Depthwise Conv(48, groups=48): 48*3*3             =    432
             BatchNorm2d(48):               48*2               =     96
             Pointwise Conv(48->96):        48*96              =  4,608
             BatchNorm2d(96):               96*2               =    192
    Block3  Depthwise Conv(96, groups=96): 96*3*3             =    864
             BatchNorm2d(96):               96*2               =    192
             Pointwise Conv(96->128):       96*128             = 12,288
             BatchNorm2d(128):              128*2              =    256
    Head    Linear(128, 9): 128*9 + 9                          =  1,161
    -----------------------------------------------------------------
    TOTAL                                                       = 22,297 trainable params

This is well under the 100,000 ceiling (~4.5x margin), leaving headroom that
was intentionally *not* spent on extra width -- keeping the model this lean
is itself the point of Section 2's "resource-constrained" brief.

Hardware-aware activation choice: ReLU6
----------------------------------------
Model B uses ReLU6 (clip(x, 0, 6)) instead of Sigmoid, Tanh, GELU or Swish:
  - Sigmoid/Tanh/GELU/Swish all require an exp(x) evaluation per element.
    Microcontrollers and many edge NPUs/FPGAs have no dedicated transcendental
    (exp/log) hardware unit, so exp() is emulated in software (polynomial /
    lookup-table approximation) at a large ALU cycle cost relative to a single
    compare-and-clamp instruction for ReLU6.
  - ReLU6's *bounded* output range (0-6) additionally exposes a fixed
    dynamic range, which is exactly the property INT8 post-training
    quantization tooling (e.g. TensorFlow Lite, PyTorch Mobile) relies on to
    map activations into 8-bit integers with minimal clipping error -- this
    is why ReLU6 specifically (not plain unbounded ReLU) is MobileNet's
    signature activation choice for quantized edge deployment.
"""

from typing import Tuple

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """
    One depthwise-separable convolution block:
        Depthwise Conv2d(groups=in_channels) -> BN -> ReLU6
        Pointwise Conv2d(k=1)                -> BN -> ReLU6
    Optionally followed by a 2x2 max-pool for spatial downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False,
        )
        self.bn_depthwise = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_pointwise = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU6(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn_depthwise(self.depthwise(x)))
        x = self.act(self.bn_pointwise(self.pointwise(x)))
        return self.pool(x)


class ModelB(nn.Module):
    """
    Lightweight Depthwise Separable CNN, < 100,000 trainable parameters.
    Input:  (B, 3, 64, 64)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.2):
        super().__init__()

        # Stem: standard conv, 3 -> 24 channels, then downsample 64x64 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Three depthwise-separable blocks, each halving spatial size and
        # widening channels: 24 -> 48 -> 96 -> 128
        self.block1 = DepthwiseSeparableBlock(24, 48, pool=True)    # -> 48 x 16 x 16
        self.block2 = DepthwiseSeparableBlock(48, 96, pool=True)    # -> 96 x 8 x 8
        self.block3 = DepthwiseSeparableBlock(96, 128, pool=False)  # -> 128 x 8 x 8 (pool replaced by GAP)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> 128 x 1 x 1
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable (requires_grad=True) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize_layer_shapes(model: "ModelB", input_size: Tuple[int, int, int, int] = (1, 3, 64, 64)) -> None:
    """Prints per-stage output shape and running trainable-parameter total."""
    x = torch.zeros(input_size)
    running_total = 0

    def report(name: str, tensor: torch.Tensor, module: nn.Module) -> None:
        nonlocal running_total
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        running_total += n_params
        print(f"  {name:<10s} -> shape {tuple(tensor.shape)!s:<20s} | +{n_params:>6,} params | total {running_total:>7,}")

    print(f"Input      -> shape {tuple(x.shape)}")
    x = model.stem(x); report("stem", x, model.stem)
    x = model.block1(x); report("block1", x, model.block1)
    x = model.block2(x); report("block2", x, model.block2)
    x = model.block3(x); report("block3", x, model.block3)
    x = model.global_pool(x); x = torch.flatten(x, 1)
    print(f"  {'gap+flat':<10s} -> shape {tuple(x.shape)!s:<20s} | +{0:>6,} params | total {running_total:>7,}")
    logits = model.classifier(model.dropout(x))
    report("head", logits, model.classifier)


if __name__ == "__main__":
    model = ModelB(num_classes=9)
    total_params = count_trainable_parameters(model)

    print("=== Model B: Depthwise Separable CNN ===")
    summarize_layer_shapes(model)
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Budget ceiling:              100,000")
    print(f"Margin under budget:         {100_000 - total_params:,} ({(1 - total_params / 100_000) * 100:.1f}% headroom)")
    assert total_params < 100_000, "Model B exceeds the 100,000 trainable parameter budget!"
    print("\n✅ Verified: Model B is strictly under the 100,000 parameter budget.")
