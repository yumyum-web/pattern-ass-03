# EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification

**Module:** EN3150 — Pattern Recognition  
**Team:** Outliers  
**Members:**
- Yumeth
- Ravishan
- Shanuka
- Damindu

---

## 📌 Project Overview

This project focuses on the design, optimization, and edge benchmarking of lightweight Convolutional Neural Networks (CNNs) for resource-constrained embedded environments (e.g., smart waste sorting sensor nodes on microcontrollers or Raspberry Pi).

Key components:
1. **Dataset Selection & Preparation:** Real-world image classification using the **RealWaste** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste) downscaled to $64 \times 64$ pixels with a deterministic 70% / 15% / 15% train/val/test split.
2. **Custom CNN Architectures:**
   - **Model A (Standard CNN):** Interleaved standard 2D convolutions and pooling layers.
   - **Model B (Lightweight CNN):** Depthwise Separable Convolutions strictly constrained to $<100,000$ trainable parameters and hardware-friendly activations (ReLU/ReLU6).
3. **Optimizer Tuning & Momentum Study:** Empirical convergence comparison among Adam, standard SGD, and SGD with Momentum ($\beta = 0.9$).
4. **SOTA Edge Model Fine-Tuning:** Transfer learning and benchmarking with lightweight architectures (MobileNetV2 and ShuffleNetV2).
5. **Hardware Profiling & Trade-offs:** Evaluation of memory footprint (KB/MB), parameter efficiency, computational cost (MACs/FLOPs), and test accuracy.

---

## 🚀 Deliverables

- **Jupyter Notebook:** `Outliers_A03_EN3150.ipynb` (complete, self-contained, and ready to run)
- **Report Document:** `Outliers_A03_EN3150.pdf`

---

## 🛠️ Environment Setup

We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management.

```bash
# Clone the repository
git clone git@github.com:yumyum-web/pattern-ass-03.git
cd pattern-ass-03

# Install dependencies and sync virtual environment
uv sync
```
