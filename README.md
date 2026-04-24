# Self-Pruning Neural Network

A neural network that learns to prune its own weights **during training** using
learnable gate parameters and L1 sparsity regularisation, trained on CIFAR-10.

---

## Concept

Each weight `w_ij` in a `PrunableLinear` layer has a corresponding learnable gate
score `g_ij`. During the forward pass:

```
gates       = sigmoid(gate_scores)       # values in (0, 1)
pruned_W    = weight * gates             # dead gates zero out weights
output      = pruned_W @ x + bias
```

The training objective combines two terms:

```
Total Loss = CrossEntropy(logits, labels) + λ * Σ sigmoid(gate_scores)
```

The L1 penalty on gates creates constant gradient pressure pushing them toward zero.
Gates whose weights don't help classification lose the tug-of-war and collapse —
the weight is effectively pruned.

---

## Project Structure

```
self-pruning-nn/
├── src/
│   ├── prunable_layer.py   # PrunableLinear — the gated weight layer
│   ├── model.py            # SelfPruningNet — CNN + prunable FC head
│   ├── loss.py             # Sparsity loss + total loss combiner
│   ├── train.py            # Training loop and data loading
│   └── evaluate.py         # Metrics, plots, CSV export
│
├── experiments/
│   └── run_lambda_sweep.py # Full λ comparison run
│
├── outputs/                # Generated after training
│   ├── results.csv
│   ├── gate_dist_lambda_*.png
│   └── training_curves.png
│
├── report/
│   └── report.md           # Analysis and results writeup
│
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Single run

```bash
# Train with one lambda value
python src/train.py --lambda_ 1e-3 --epochs 30

# Options:
#   --lambda_     sparsity weight       (default: 1e-3)
#   --epochs      number of epochs      (default: 30)
#   --lr          Adam learning rate    (default: 1e-3)
#   --batch_size  mini-batch size       (default: 128)
#   --dropout     dropout rate in FC    (default: 0.3)
```

### Full lambda sweep (recommended)

```bash
# Trains for λ ∈ {1e-4, 1e-3, 1e-2}, generates all outputs and plots
python experiments/run_lambda_sweep.py

# Quick smoke-test (5 epochs each, ~2 min on CPU)
python experiments/run_lambda_sweep.py --quick
```

---

## Outputs

| File                          | Description                            |
|-------------------------------|----------------------------------------|
| `outputs/results.csv`         | Lambda / Test Accuracy / Sparsity table|
| `outputs/gate_dist_*.png`     | Gate value histogram per lambda        |
| `outputs/training_curves.png` | Accuracy + sparsity over epochs        |
| `report/report.md`            | Full written report                    |

---

## Architecture

```
Input (3×32×32)
    ↓
Conv(3→32) → BN → ReLU → MaxPool        # (B, 32, 16, 16)
Conv(32→64) → BN → ReLU → MaxPool       # (B, 64, 8, 8)
Conv(64→128) → BN → ReLU → AvgPool      # (B, 128, 1, 1)
Flatten                                  # (B, 128)
    ↓
PrunableLinear(128→256) → ReLU → Dropout   ← prunable
PrunableLinear(256→128) → ReLU → Dropout   ← prunable
PrunableLinear(128→10)                     ← prunable
    ↓
Logits (10 classes)
```

**Prunable parameters:** 66,816 gated weights across three FC layers.

---

## Expected Results

| Lambda | Test Accuracy | Sparsity |
|--------|:-------------:|:--------:|
| 1e-4   | ~83–85%       | ~20–35%  |
| 1e-3   | ~78–82%       | ~55–75%  |
| 1e-2   | ~60–68%       | ~85–95%  |
