"""
self_pruning_nn.py
==================
Self-Pruning Neural Network on CIFAR-10
Author: Tredence AI Engineer Case Study

A neural network that learns to prune its own weights DURING training using
learnable gate parameters and L1 sparsity regularisation.

Core mechanism:
    Each weight w_ij has a learnable gate score g_ij (same shape as weight).
    Forward pass:  gates = sigmoid(g_ij)          → values in (0, 1)
                   pruned_W = weight * gates       → dead gates zero weights
                   output   = pruned_W @ x + bias
    Loss:          CrossEntropy + λ * mean(sigmoid(gate_scores))

Usage:
    # Full lambda sweep (produces all outputs)
    python self_pruning_nn.py

    # Single lambda run
    python self_pruning_nn.py --lambda_ 1.0 --epochs 30

    # Quick smoke-test (5 epochs each lambda)
    python self_pruning_nn.py --quick
"""

import os
import csv
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — PrunableLinear Layer
# ══════════════════════════════════════════════════════════════════════════════

class PrunableLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a learnable gate per weight.

    Each weight w_ij has a corresponding gate_score g_ij (also a learned
    parameter). During the forward pass, gates = sigmoid(gate_scores) are
    element-wise multiplied with the weights before the linear operation.

    When a gate collapses to ~0, the corresponding weight is effectively
    pruned — it contributes nothing to the output.

    Args:
        in_features  : number of input features
        out_features : number of output features
        bias         : whether to add a learnable bias (default: True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight parameter — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Gate scores: same shape as weight, also fully learnable.
        # Initialized to 2.0 so sigmoid(2.0) = 0.88 — gates start nearly open.
        # The sparsity loss will drive unimportant gate_scores toward -inf,
        # making sigmoid(gate_scores) → 0 (pruned).
        self.gate_scores = nn.Parameter(
            torch.ones(out_features, in_features) * 2.0
        )

        # Bias — not gated, only weights are pruned
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform init — identical to nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: squash gate scores to (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: element-wise multiply — gates near 0 kill their weight
        pruned_weights = self.weight * gates

        # Step 3: standard linear operation with pruned weight matrix
        # Gradients flow through BOTH self.weight and self.gate_scores here
        return F.linear(x, pruned_weights, self.bias)

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Current gate values in (0, 1). Detached from computation graph."""
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below threshold."""
        return (self.get_gates() < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 (cont.) — Network Definition
# ══════════════════════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    """
    CNN + Prunable FC classifier for CIFAR-10.

    Architecture:
        Conv feature extractor (standard layers — spatial feature learning)
            Conv(3→32) → BN → ReLU → MaxPool
            Conv(32→64) → BN → ReLU → MaxPool
            Conv(64→128) → BN → ReLU → AdaptiveAvgPool
            → output: (B, 128)

        Prunable classifier (PrunableLinear layers — where pruning happens)
            PrunableLinear(128→256) → ReLU → Dropout
            PrunableLinear(256→128) → ReLU → Dropout
            PrunableLinear(128→10)  → logits

    Why prune only FC layers?
        Conv layers handle spatial structure efficiently with few parameters.
        FC layers are where redundant capacity concentrates — ideal prune targets.
        Total prunable parameters: 128×256 + 256×128 + 128×10 = 66,816
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        # Feature extractor — standard Conv stack
        self.features = nn.Sequential(
            # Block 1: (B, 3, 32, 32) → (B, 32, 16, 16)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (B, 32, 16, 16) → (B, 64, 8, 8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (B, 64, 8, 8) → (B, 128, 1, 1)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten = nn.Flatten()  # → (B, 128)

        # Prunable classifier
        self.classifier = nn.Sequential(
            PrunableLinear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            PrunableLinear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            PrunableLinear(128, 10),   # 10 CIFAR-10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)    # raw logits

    def prunable_layers(self) -> list:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def all_gates(self) -> torch.Tensor:
        return torch.cat([l.get_gates().flatten() for l in self.prunable_layers()])

    @torch.no_grad()
    def global_sparsity(self, threshold: float = 1e-2) -> float:
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()

    def count_prunable_params(self) -> int:
        return sum(l.weight.numel() for l in self.prunable_layers())


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Sparsity Regularisation Loss
# ══════════════════════════════════════════════════════════════════════════════

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    Computes the normalised L1 sparsity penalty.

    SparsityLoss = mean of sigmoid(gate_scores) across all PrunableLinear layers.

    Why normalise by count?
        Raw sum grows with number of parameters (66,816 * 0.5 ≈ 33,000 at init).
        Normalised mean stays in (0, 1) — same scale as CrossEntropy (~1.7 at init).
        This makes λ values interpretable and consistent regardless of model size.

    Why L1 (sum/mean) encourages sparsity:
        Gradient of mean(sigmoid(g)) w.r.t. g_ij = sigmoid(g_ij)*(1-sigmoid(g_ij))/N
        This creates a CONSTANT downward pressure on every gate score, regardless of
        how small the gate already is. Unlike L2, this gradient doesn't vanish near
        zero — it keeps pushing until the gate fully collapses. Gates whose weights
        help classification resist this pressure; useless gates surrender.

    Returns:
        Scalar tensor in (0, 1). Differentiable — gradients flow to gate_scores.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0

    for layer in model.prunable_layers():
        gates  = torch.sigmoid(layer.gate_scores)
        total  = total + gates.sum()
        count += gates.numel()

    return total / count   # normalised to (0, 1)


def total_loss(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    model:   SelfPruningNet,
    lambda_: float,
) -> tuple:
    """
    Total Loss = CrossEntropy(logits, targets) + λ * SparsityLoss(model)

    Returns (total, cls_loss, sp_loss) — all scalar tensors.
    Returning components separately makes per-epoch logging clean.
    """
    cls = nn.CrossEntropyLoss()(logits, targets)
    sp  = sparsity_loss(model)
    return cls + lambda_ * sp, cls, sp


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(batch_size: int = 128):
    """CIFAR-10 train and test DataLoaders with standard normalisation."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    # num_workers=0: avoids Windows multiprocessing MemoryError with spawn
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(model, loader, opt_weights, opt_gates, lambda_, device):
    """One full pass over the training set. Returns avg losses and train accuracy."""
    model.train()
    total_sum = cls_sum = sp_sum = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        opt_weights.zero_grad()
        opt_gates.zero_grad()

        logits = model(images)
        t_loss, c_loss, s_loss = total_loss(logits, labels, model, lambda_)

        t_loss.backward()
        opt_weights.step()
        opt_gates.step()

        b = images.size(0)
        total_sum += t_loss.item() * b
        cls_sum   += c_loss.item() * b
        sp_sum    += s_loss.item() * b
        correct   += logits.argmax(1).eq(labels).sum().item()
        total     += b

    n = len(loader.dataset)
    return total_sum/n, cls_sum/n, sp_sum/n, correct/total


@torch.no_grad()
def evaluate(model, loader, device):
    """Test accuracy and global sparsity on the test set."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += model(images).argmax(1).eq(labels).sum().item()
        total   += images.size(0)
    return correct / total, model.global_sparsity()


def train_model(lambda_=1e-3, epochs=30, lr=1e-3, batch_size=128,
                dropout=0.3, device=None, verbose=True) -> dict:
    """
    Full training run for one lambda value.

    Key design: TWO separate Adam optimizers — one for weights, one for gates.

    Why two optimizers?
        Gate gradients are ~17x smaller than weight gradients (confirmed empirically).
        With a single optimizer, Adam normalises all gradients to the same step size,
        so every gate moves by exactly lr regardless of which connections matter.
        This causes all gates to drift uniformly rather than differentiating.

        Solution: gate optimizer uses lr * 50 so gates can move decisively.
        Weights stay at lr=1e-3 for stable classification learning.

    Returns dict: model, history, test_accuracy, sparsity, lambda_
    """
    if device is None:
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else "cpu")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training  |  lambda={lambda_}  |  device={device}")
        print(f"{'='*60}")

    train_loader, test_loader = get_dataloaders(batch_size)
    model = SelfPruningNet(dropout_rate=dropout).to(device)

    # Separate parameter groups
    gate_params   = [l.gate_scores for l in model.prunable_layers()]
    weight_params = [p for p in model.parameters()
                     if not any(p is g for g in gate_params)]

    opt_weights = optim.Adam(weight_params, lr=lr)
    opt_gates   = optim.Adam(gate_params,   lr=lr * 50)  # 50x — compensates weak grads

    sched_w = optim.lr_scheduler.CosineAnnealingLR(opt_weights, T_max=epochs)
    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_gates,   T_max=epochs)

    history  = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        avg_total, avg_cls, avg_sp, train_acc = train_one_epoch(
            model, train_loader, opt_weights, opt_gates, lambda_, device)
        test_acc, sparsity = evaluate(model, test_loader, device)
        sched_w.step()
        sched_g.step()

        history.append(dict(epoch=epoch, total_loss=avg_total, cls_loss=avg_cls,
                            sp_loss=avg_sp, train_acc=train_acc,
                            test_acc=test_acc, sparsity=sparsity))
        if verbose:
            print(f"Epoch {epoch:>3}/{epochs} | "
                  f"Loss {avg_total:.4f} (cls {avg_cls:.4f}, sp {avg_sp:.2f}) | "
                  f"Train {train_acc*100:.1f}% | Test {test_acc*100:.1f}% | "
                  f"Sparsity {sparsity*100:.1f}% | {time.time()-t0:.1f}s")

    final_acc, final_sp = evaluate(model, test_loader, device)
    if verbose:
        print(f"\nFinal  →  Test Accuracy: {final_acc*100:.2f}%  |  "
              f"Sparsity: {final_sp*100:.2f}%")

    return dict(model=model, history=history,
                test_accuracy=final_acc, sparsity=final_sp, lambda_=lambda_)


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 (cont.) — Evaluation & Reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_sparsity_report(model, lambda_, device, threshold=1e-2):
    """Prints per-layer sparsity breakdown."""
    _, test_loader = get_dataloaders()
    test_acc, global_sp = evaluate(model, test_loader, device)

    print(f"\n{'='*58}")
    print(f"  Evaluation Report  |  lambda = {lambda_}")
    print(f"{'='*58}")
    print(f"  Test Accuracy   : {test_acc*100:.2f}%")
    print(f"  Global Sparsity : {global_sp*100:.2f}%")

    total_prunable = total_pruned = 0
    rows = []
    for i, layer in enumerate(model.prunable_layers()):
        gates    = layer.get_gates()
        n        = gates.numel()
        pruned   = (gates < threshold).sum().item()
        rows.append(dict(i=i, shape=tuple(gates.shape), n=n, pruned=pruned,
                         sp=pruned/n*100, mu=gates.mean().item(),
                         sigma=gates.std().item()))
        total_prunable += n
        total_pruned   += pruned

    print(f"  Pruned weights  : {total_pruned:,} / {total_prunable:,}")
    print(f"\n  {'Layer':<8} {'Shape':<15} {'Params':>8} {'Pruned':>8} "
          f"{'Sparsity':>10} {'Gate μ':>8} {'Gate σ':>8}")
    print(f"  {'-'*67}")
    for r in rows:
        print(f"  FC-{r['i']:<5} {str(r['shape']):<15} {r['n']:>8,} "
              f"{r['pruned']:>8,} {r['sp']:>9.1f}% {r['mu']:>8.4f} {r['sigma']:>8.4f}")
    print(f"{'='*58}\n")


def plot_gate_distribution(model, lambda_, out_path="outputs/gate_distribution.png"):
    """
    Histogram of all gate values across every PrunableLinear layer.

    A successful result shows:
        - Large spike near 0  (pruned/dead weights)
        - Cluster near 0.5–1  (active/surviving weights)
        - Clear bimodal separation
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    gates = model.all_gates().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Gate Value Distribution  —  λ = {lambda_}",
                 fontsize=14, fontweight="bold")

    # Left: full range
    ax = axes[0]
    ax.hist(gates, bins=100, color="#2563eb", edgecolor="white",
            linewidth=0.3, alpha=0.9)
    ax.axvline(0.01, color="#ef4444", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.set_xlabel("Gate value  (sigmoid output)")
    ax.set_ylabel("Count")
    ax.set_title("All gate values")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{int(x):,}"))

    # Right: zoomed near 0
    ax2 = axes[1]
    near_zero = gates[gates < 0.1]
    pct = (gates < 0.01).mean() * 100
    ax2.hist(near_zero, bins=60, color="#16a34a", edgecolor="white",
             linewidth=0.3, alpha=0.9)
    ax2.set_xlabel("Gate value")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Zoomed — gates < 0.1\n({pct:.1f}% of all gates below 0.01)")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gate distribution plot saved → {out_path}")


def plot_training_curves(all_histories, out_path="outputs/training_curves.png"):
    """Accuracy and sparsity over epochs for all lambda values."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training Curves — Lambda Sweep", fontsize=14, fontweight="bold")

    for (lam, hist), color in zip(all_histories.items(), colors):
        ep  = [h["epoch"]        for h in hist]
        acc = [h["test_acc"]*100 for h in hist]
        sp  = [h["sparsity"]*100 for h in hist]
        axes[0].plot(ep, acc, color=color, label=f"λ={lam}", linewidth=1.8)
        axes[1].plot(ep, sp,  color=color, label=f"λ={lam}", linewidth=1.8)

    for ax, title, ylabel in zip(
            axes,
            ["Test Accuracy over Epochs", "Sparsity Level over Epochs"],
            ["Test Accuracy (%)", "Sparsity (%)"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {out_path}")


def save_results_csv(lambda_, test_acc, sparsity, out_path="outputs/results.csv"):
    """Appends one result row to CSV. Creates file with header if needed."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_header = not os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Lambda", "Test Accuracy (%)", "Sparsity Level (%)"])
        w.writerow([lambda_, round(test_acc*100, 2), round(sparsity*100, 2)])
    print(f"Results appended → {out_path}")


def print_summary_table(all_results):
    print("\n" + "="*52)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity':>15}")
    print("="*52)
    for r in all_results:
        print(f"  {r['lambda_']:<12.0e} "
              f"{r['test_accuracy']*100:>14.2f}% "
              f"{r['sparsity']*100:>14.2f}%")
    print("="*52 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Lambda Sweep Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(lambdas=(0.1, 1.0, 5.0), epochs=30, quick=False):
    """Trains one model per lambda, evaluates, and generates all outputs."""
    if quick:
        epochs = 5
        print("Quick mode: running 5 epochs per lambda")

    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")

    csv_path = "outputs/results.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    all_histories = {}
    all_results   = []

    for lambda_ in lambdas:
        result = train_model(lambda_=lambda_, epochs=epochs, verbose=True)

        all_histories[lambda_] = result["history"]

        # Per-lambda gate distribution plot
        plot_gate_distribution(
            result["model"], lambda_,
            out_path=f"outputs/gate_dist_lambda_{lambda_:.0e}.png"
        )
        print_sparsity_report(result["model"], lambda_, device)
        save_results_csv(lambda_, result["test_accuracy"], result["sparsity"])

        all_results.append({
            "lambda_":       lambda_,
            "test_accuracy": result["test_accuracy"],
            "sparsity":      result["sparsity"],
        })

    plot_training_curves(all_histories, out_path="outputs/training_curves.png")
    print_summary_table(all_results)
    print("All outputs saved to ./outputs/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network")
    parser.add_argument("--lambda_",    type=float, default=None,
                        help="Single lambda value (skips sweep)")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--quick",      action="store_true",
                        help="5 epochs per lambda (smoke test)")
    args = parser.parse_args()

    if args.lambda_ is not None:
        # Single run mode
        device = ("cuda" if torch.cuda.is_available() else
                  "mps"  if torch.backends.mps.is_available() else "cpu")
        result = train_model(lambda_=args.lambda_, epochs=args.epochs,
                             lr=args.lr, batch_size=args.batch_size)
        plot_gate_distribution(result["model"], args.lambda_)
        print_sparsity_report(result["model"], args.lambda_, device)
        save_results_csv(args.lambda_, result["test_accuracy"], result["sparsity"])
    else:
        # Full sweep mode
        run_sweep(lambdas=(0.1, 1.0, 5.0), epochs=args.epochs, quick=args.quick)