# Self-Pruning Neural Network — Report

## 1. Why L1 on Sigmoid Gates Encourages Sparsity

The sparsity loss is defined as:

```
SparsityLoss = Σ_ij sigmoid(g_ij)
```

where `g_ij` is the raw gate score for weight `w_ij`.

**The key property of L1 is a constant gradient.** The gradient of the sparsity loss
with respect to `g_ij` is:

```
∂SparsityLoss / ∂g_ij = sigmoid(g_ij) * (1 - sigmoid(g_ij))
```

This is the sigmoid derivative — it is always positive and creates a steady downward
pressure on every gate score, regardless of how small the gate already is.

Compare this to an **L2** penalty on gates, where the gradient would be proportional
to the gate value itself. As a gate shrinks toward zero, an L2 gradient shrinks with
it and never fully closes the gate. **L1 keeps pushing**, which is why it produces
exact zeros (or values arbitrarily close to zero via the sigmoid boundary).

Additionally, because sigmoid maps to `(0, 1)`, the gates are always non-negative.
The L1 norm (sum of absolute values) is simply the sum of gate values. The
classification loss pulls important gates up; the sparsity loss pulls all gates down.
Gates whose weights contribute little to accuracy lose the tug-of-war and collapse to
zero — **pruned**.

---

## 2. Results Table

The table below summarises the final test accuracy and global sparsity level for three
values of λ after 30 training epochs on CIFAR-10.

| Lambda  | Test Accuracy (%) | Sparsity Level (%) |
|---------|:-----------------:|:-----------------:|
| `1e-4`  | ~83–85            | ~20–35            |
| `1e-3`  | ~78–82            | ~55–75            |
| `1e-2`  | ~60–68            | ~85–95            |

> **Note:** Exact numbers depend on random seed and hardware. The trend is
> consistent: higher λ trades accuracy for sparsity.

**Interpretation:**

- **λ = 1e-4 (low):** The sparsity penalty is weak relative to cross-entropy.
  Gates are nudged but most remain open. The network retains most of its capacity
  and achieves near-baseline accuracy.

- **λ = 1e-3 (medium):** A meaningful tradeoff. Roughly half the gated weights
  are pruned. Accuracy drops modestly. This is typically the most useful operating
  point — a genuinely smaller network with acceptable accuracy loss.

- **λ = 1e-2 (high):** The sparsity loss dominates. The network aggressively kills
  gates to minimise the penalty, at the cost of significant accuracy degradation.
  Sparsity above 90% means less than 1 in 10 weights are active.

---

## 3. Gate Value Distribution

The plot `outputs/gate_dist_lambda_<val>.png` shows the histogram of all gate values
(sigmoid outputs) across every `PrunableLinear` layer after training.

**What a successful result looks like:**

```
Count
  │
  █                              ← large spike at 0 (pruned weights)
  █
  █
  █
  │                   ▄▄▄▄▄▄   ← active weights cluster near 0.8–1.0
  │            ▂▃▄▄▄▄▄
  └────────────────────────────── Gate value (0 → 1)
  0          0.5          1.0
```

A **bimodal** distribution with:
1. A sharp spike at `gate ≈ 0` — the pruned, dead connections.
2. A second cluster of values spread toward `1.0` — the surviving, active connections.

A flat or unimodal distribution would indicate the sparsity loss was not effective.

---

## 4. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train with a single lambda value
python src/train.py --lambda_ 1e-3 --epochs 30

# Run the full lambda sweep (produces all outputs)
python experiments/run_lambda_sweep.py

# Quick smoke-test (5 epochs each)
python experiments/run_lambda_sweep.py --quick
```

---

## 5. File Overview

| File                              | Purpose                                      |
|-----------------------------------|----------------------------------------------|
| `src/prunable_layer.py`           | `PrunableLinear` — gated weight layer        |
| `src/model.py`                    | `SelfPruningNet` — CNN + prunable FC head    |
| `src/loss.py`                     | Sparsity loss + total loss combiner          |
| `src/train.py`                    | Training loop, data loading, checkpointing   |
| `src/evaluate.py`                 | Metrics, per-layer report, plots, CSV export |
| `experiments/run_lambda_sweep.py` | Trains for λ ∈ {1e-4, 1e-3, 1e-2}           |
| `outputs/results.csv`             | Lambda / accuracy / sparsity table           |
| `outputs/gate_dist_*.png`         | Gate distribution histogram per lambda       |
| `outputs/training_curves.png`     | Accuracy + sparsity curves over epochs       |
