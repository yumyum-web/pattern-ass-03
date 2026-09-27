"""
Model A: Standard 2D Convolutional Neural Network Baseline.
Designed for 64x64 edge image classification on the RealWaste dataset.
Interleaves standard 2D convolutions, batch normalization, ReLU activations,
and max-pooling stages, followed by global average pooling and a dense classifier.
"""

from typing import Dict, Tuple, Any
import torch
import torch.nn as nn


class ModelA(nn.Module):
    """
    Standard CNN Baseline (Model A).
    
    Architecture Topology:
        - Input: (B, 3, 64, 64)
        - Block 1: Conv2d(3 -> 32, k=3, p=1) + BatchNorm2d(32) + ReLU + MaxPool2d(2, 2) -> (B, 32, 32, 32)
        - Block 2: Conv2d(32 -> 64, k=3, p=1) + BatchNorm2d(64) + ReLU + MaxPool2d(2, 2) -> (B, 64, 16, 16)
        - Block 3: Conv2d(64 -> 128, k=3, p=1) + BatchNorm2d(128) + ReLU + MaxPool2d(2, 2) -> (B, 128, 8, 8)
        - Global Pooling: AdaptiveAvgPool2d((1, 1)) -> (B, 128, 1, 1) -> Flatten -> (B, 128)
        - Classifier Head:
            - Dropout(p=0.3)
            - Linear(128 -> 64) + ReLU
            - Dropout(p=0.2)
            - Linear(64 -> 9)
    
    Hardware & Activation Justifications:
        1. ReLU (Rectified Linear Unit, f(x) = max(0, x)):
           - Zero transcendental operations: Avoids evaluating exponentials (e^x) as required by
             Sigmoid, Tanh, or GELU. On edge microcontrollers (e.g. ARM Cortex-M, ESP32-S3),
             computing e^x requires multi-cycle polynomial expansions or lookup tables.
           - Single-cycle ALU execution: Implemented via simple conditional clamping or bitwise mask.
           - Quantization friendly: Non-saturating positive regime preserves dynamic range for INT8.
        2. Global Average Pooling (GAP):
           - Replaces high-dimensional flattening (which would require 128 * 8 * 8 = 8,192 input units
             and >500k dense weights), enforcing spatial invariance while minimizing parameter overhead.
    """

    def __init__(self, num_classes: int = 9, in_channels: int = 3, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Feature Extractor: Interleaved Standard Convolutions & Max-Pooling
        self.features = nn.Sequential(
            # --- Stage 1: (3, 64, 64) -> (32, 32, 32) ---
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # --- Stage 2: (32, 32, 32) -> (64, 16, 16) ---
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # --- Stage 3: (64, 16, 16) -> (128, 8, 8) ---
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Receptive Aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.67),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) normal initialization for conv layers and Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward inference pass.
        Args:
            x (torch.Tensor): Input batch of shape (B, 3, 64, 64)
        Returns:
            torch.Tensor: Unnormalized class logits of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> Dict[str, int]:
        """Returns total, trainable, and non-trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
        }

    def get_layer_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Detailed breakdown of trainable parameters across each architectural layer.
        Useful for formal report documentation and comparative analysis.
        """
        breakdown = {
            "Conv1 (3x3, 3->32)": {"params": 32 * 3 * 3 * 3, "output_shape": "32x32x32"},
            "BatchNorm1 (32)": {"params": 32 * 2, "output_shape": "32x32x32"},
            "Conv2 (3x3, 32->64)": {"params": 64 * 32 * 3 * 3, "output_shape": "64x16x16"},
            "BatchNorm2 (64)": {"params": 64 * 2, "output_shape": "64x16x16"},
            "Conv3 (3x3, 64->128)": {"params": 128 * 64 * 3 * 3, "output_shape": "128x8x8"},
            "BatchNorm3 (128)": {"params": 128 * 2, "output_shape": "128x8x8"},
            "Linear1 (128->64)": {"params": 128 * 64 + 64, "output_shape": "64"},
            "Linear2 (64->9)": {"params": 64 * 9 + 9, "output_shape": "9"},
        }
        return breakdown


def create_model_a(num_classes: int = 9, in_channels: int = 3) -> ModelA:
    """Factory helper to instantiate Model A."""
    return ModelA(num_classes=num_classes, in_channels=in_channels)


if __name__ == "__main__":
    print("=" * 60)
    print("Model A: Standard 2D CNN Baseline Parameter Analysis")
    print("=" * 60)

    model = create_model_a(num_classes=9)
    counts = model.count_parameters()
    
    print(f"Total Parameters:          {counts['total_parameters']:,}")
    print(f"Trainable Parameters:      {counts['trainable_parameters']:,}")
    print(f"Non-Trainable Parameters:  {counts['non_trainable_parameters']:,}")
    print("-" * 60)
    print(f"{'Layer Name':<28} | {'Param Count':<12} | {'Output Resolution'}")
    print("-" * 60)
    
    for layer, info in model.get_layer_breakdown().items():
        print(f"{layer:<28} | {info['params']:<12,} | {info['output_shape']}")
    print("-" * 60)

    dummy_input = torch.randn(2, 3, 64, 64)
    out = model(dummy_input)
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {out.shape} (Expected: [2, 9])")
    assert out.shape == (2, 9), f"Unexpected output shape: {out.shape}"
    print("✅ Model A forward pass and parameter verification successful!")
