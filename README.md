# CIFAR-10 CNN Scaling Law Experiments


**###########################################################  
TODO: Update README  
This is a work in progress and will be double-checked later  
Check Research_Log.md in Notes/ for more information  
###########################################################**

## Overview

This project studies **empirical scaling laws** in convolutional neural networks (CNNs) trained on CIFAR-10. The objective is to understand how test loss scales with:

- **Model size (N)** — controlled via width and depth  
- **Dataset size (D)** — controlled via deterministic subsampling  
- **Compute (C)** — approximated as `C ∝ N × D × epochs`

The experiments are designed to measure scaling trends and analyze optimal tradeoffs under fixed compute constraints.

---

## Research Goals

1. Measure loss scaling with **width** (fixed depth, fixed dataset size).
2. Measure loss scaling with **depth** (fixed width, fixed dataset size).
3. Measure loss scaling with **dataset size** (fixed model).
4. Evaluate performance under **fixed compute**, testing:
   - Model size vs dataset size tradeoffs
   - Model size vs training duration (epochs) tradeoffs

The broader goal is to estimate scaling exponents and compare behavior to known scaling-law results, using small CNNs (~300k parameters).

---

## Experimental Design

### Model Architecture

- Configurable 3-stage CNN
- Variable **width**
- Variable **blocks per stage (depth)**
- Global average pooling
- Final linear classifier

This allows controlled scaling of parameter count.

### Dataset

- CIFAR-10
- Deterministic subsampling using a fixed seed
- Dataset scaling implemented via `torch.utils.data.Subset`

### Training Setup

- Optimizer: SGD
- Scheduler: CosineAnnealingLR
- Fixed batch size
- Reproducible configuration:
  - Fixed random seed
  - Deterministic CuDNN settings

---

## Compute Approximation

Compute is approximated as:

```
C ∝ N × D × epochs
```

Where:
- `N` = number of model parameters  
- `D` = dataset size  
- `epochs` = number of full passes over data  

This proxy is used for fixed-compute experiments.

---

## Logging and Outputs

For each run:

- Per-epoch metrics saved to CSV
- Summary CSV includes:
  - Parameter count (N)
  - Dataset size (D)
  - Estimated compute proxy
  - Final loss (average of last 5 epochs)
  - Best validation accuracy
  - Regression statistics (A, exponent α, R² when applicable)

All experiment outputs are stored in a structured `results/` directory.

---

## Fixed Compute Experiments

Under a constant compute budget:

- Inversely scale **N and D**
- Inversely scale **N and epochs**

Then measure which allocation minimizes final loss.

This enables empirical estimation of the optimal proportion between model size and dataset size under constrained compute.

---

## Code Structure (High-Level)

- `model.py` — Configurable CNN architecture
- `train.py` — Training loop, logging, checkpointing
- `utils.py` — Seed control, dataset handling, compute estimation
- `analysis.ipynb` — Regression fitting and scaling visualization
- `results/` — Stored experiment outputs

Experiments are run via command-line arguments (`argparse`) to ensure reproducibility.

---

## Summary

This project builds a controlled experimental pipeline to study:

- Loss vs model size  
- Loss vs dataset size  
- Optimal scaling under fixed compute  

It provides a reproducible small-scale empirical study of neural scaling behavior using CNNs trained on CIFAR-10.


