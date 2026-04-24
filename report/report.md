# Self-Pruning Neural Network — Report

## 1. Why L1 on Sigmoid Gates Encourages Sparsity

The sparsity loss is defined as:

SparsityLoss = Σ_ij sigmoid(g_ij)

where g_ij is the raw gate score corresponding to weight w_ij.

Each gate is obtained by applying a sigmoid function to the gate scores, ensuring
values lie in the range (0, 1). These gates multiplicatively scale the weights,
so values close to zero effectively remove connections.

The L1 penalty (sum of gate values) encourages sparsity by applying a consistent
downward pressure on all gates. Unlike L2 regularization, whose gradient diminishes
as values approach zero, the L1 penalty continues to encourage shrinkage even for
small values.

As a result:

* Important weights are preserved by the classification loss.
* Unimportant weights are pushed toward zero by the sparsity loss.

Because sigmoid outputs never reach exact zero, gates below a small threshold
(e.g., 1e-2) are treated as pruned during evaluation.

---

## 2. Results

The model was trained on CIFAR-10 for 30 epochs with different values of λ
to study the sparsity–accuracy trade-off.

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| ------ | ----------------- | ------------------ |
| 0.1    | 77.01%            | 67.76%             |
| 1.0    | 75.80%            | 92.61%             |
| 5.0    | 73.57%            | 98.13%             |

### Interpretation

* **λ = 0.1 (low):**
  Moderate sparsity with minimal impact on accuracy.

* **λ = 1.0 (medium):**
  Strong pruning (>90%) while retaining most predictive performance.

* **λ = 5.0 (high):**
  Extreme pruning (~98%) with a modest drop in accuracy.

These results demonstrate that the model successfully learns to prune itself
during training. Even at very high sparsity levels, performance remains
reasonably strong, indicating significant redundancy in the network.

---

## 3. Gate Value Distribution

Below is the distribution of gate values for the best model:

![Gate Distribution](gate_dist.png)

### Observations

The distribution shows a clear bimodal pattern:

1. A large spike near 0
   → represents pruned or inactive connections

2. A second cluster away from 0 (toward higher values)
   → represents important, active weights

This separation confirms that the sparsity loss is effective in distinguishing
between useful and redundant parameters.

---

## 4. Key Insight

Layer-wise analysis shows that:

* Early fully connected layers are pruned very aggressively (~98%)
* The final classification layer is pruned less (~70%)

This suggests:

* Earlier layers are highly overparameterized
* Later layers retain more task-critical information

---

## 5. Conclusion

This project demonstrates a self-pruning neural network that learns to remove
unnecessary connections during training using learnable gates and L1
regularization.

Key outcomes:

* Up to **98% sparsity achieved**
* Only a small drop in accuracy (~3–4%)
* No post-training pruning required

This highlights the effectiveness of differentiable pruning mechanisms for
building compact and efficient neural networks.

---

## 6. How to Run

```bash
pip install -r requirements.txt

# Train for a single lambda
python self_pruning_nn.py --lambda_ 1.0 --epochs 30

# Or run multiple lambdas (if included)
python self_pruning_nn.py --sweep
```
