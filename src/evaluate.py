"""
evaluate.py
-----------
Post-training evaluation, sparsity reporting, and gate distribution plotting.

What this module does:
    1. compute_metrics()      — test accuracy + per-layer and global sparsity
    2. print_sparsity_report()— human-readable breakdown of gate stats
    3. plot_gate_distribution()— matplotlib histogram of all gate values
    4. save_results_csv()     — appends one row per lambda run to results.csv
"""

import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import sys
sys.path.insert(0, os.path.dirname(__file__))

from model import SelfPruningNet
from train import get_dataloaders, evaluate


# ------------------------------------------------------------------
# 1. Metrics
# ------------------------------------------------------------------
def compute_metrics(
    model:     SelfPruningNet,
    device:    str,
    threshold: float = 1e-2,
) -> dict:
    """
    Computes full evaluation metrics on the CIFAR-10 test set.

    Args:
        model     : trained SelfPruningNet
        device    : torch device string
        threshold : gates below this are counted as pruned

    Returns dict with:
        test_accuracy    : float (0–1)
        global_sparsity  : float (0–1)
        per_layer        : list of dicts, one per PrunableLinear
        total_prunable   : total prunable weight count
        total_pruned     : number of weights whose gate < threshold
    """
    _, test_loader = get_dataloaders()
    test_acc, global_sp = evaluate(model, test_loader, device)

    per_layer = []
    total_prunable = 0
    total_pruned   = 0

    for i, layer in enumerate(model.prunable_layers()):
        gates    = layer.get_gates()               # detached tensor
        n_params = gates.numel()
        n_pruned = (gates < threshold).sum().item()

        per_layer.append({
            "layer_idx":    i,
            "shape":        tuple(gates.shape),
            "n_params":     n_params,
            "n_pruned":     n_pruned,
            "sparsity_pct": n_pruned / n_params * 100,
            "gate_mean":    gates.mean().item(),
            "gate_std":     gates.std().item(),
        })

        total_prunable += n_params
        total_pruned   += n_pruned

    return {
        "test_accuracy":   test_acc,
        "global_sparsity": global_sp,
        "per_layer":       per_layer,
        "total_prunable":  total_prunable,
        "total_pruned":    total_pruned,
    }


# ------------------------------------------------------------------
# 2. Human-readable report
# ------------------------------------------------------------------
def print_sparsity_report(metrics: dict, lambda_: float):
    """Prints a formatted sparsity breakdown to stdout."""
    print(f"\n{'='*58}")
    print(f"  Evaluation Report  |  lambda = {lambda_}")
    print(f"{'='*58}")
    print(f"  Test Accuracy   : {metrics['test_accuracy']*100:.2f}%")
    print(f"  Global Sparsity : {metrics['global_sparsity']*100:.2f}%")
    print(f"  Pruned weights  : {metrics['total_pruned']:,} / "
          f"{metrics['total_prunable']:,}")
    print(f"\n  Per-layer breakdown:")
    print(f"  {'Layer':<8} {'Shape':<15} {'Params':>8} "
          f"{'Pruned':>8} {'Sparsity':>10} {'Gate μ':>8} {'Gate σ':>8}")
    print(f"  {'-'*67}")

    for lyr in metrics["per_layer"]:
        print(
            f"  FC-{lyr['layer_idx']:<5} "
            f"{str(lyr['shape']):<15} "
            f"{lyr['n_params']:>8,} "
            f"{lyr['n_pruned']:>8,} "
            f"{lyr['sparsity_pct']:>9.1f}% "
            f"{lyr['gate_mean']:>8.4f} "
            f"{lyr['gate_std']:>8.4f}"
        )
    print(f"{'='*58}\n")


# ------------------------------------------------------------------
# 3. Gate distribution plot
# ------------------------------------------------------------------
def plot_gate_distribution(
    model:    SelfPruningNet,
    lambda_:  float,
    out_path: str = "outputs/gate_distribution.png",
):
    """
    Plots a histogram of all gate values across every PrunableLinear layer.

    A well-trained model shows:
        - A large spike near 0 (pruned weights)
        - A cluster of values spread toward 1 (active weights)
        - A clear bimodal separation

    Args:
        model    : trained SelfPruningNet
        lambda_  : used only for the plot title
        out_path : where to save the PNG
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Collect all gate values
    all_gates = model.all_gates().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Gate Value Distribution — λ = {lambda_}",
        fontsize=14, fontweight="bold", y=1.01
    )

    # --- Left: full histogram ---
    ax = axes[0]
    ax.hist(all_gates, bins=100, color="#2563eb", edgecolor="white",
            linewidth=0.3, alpha=0.9)
    ax.set_title("All gate values")
    ax.set_xlabel("Gate value  (sigmoid output)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax.axvline(0.01, color="#ef4444", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    # --- Right: zoom into [0, 0.1] to show the spike at 0 clearly ---
    ax2 = axes[1]
    near_zero = all_gates[all_gates < 0.1]
    ax2.hist(near_zero, bins=60, color="#16a34a", edgecolor="white",
             linewidth=0.3, alpha=0.9)
    ax2.set_title("Zoomed — gates < 0.1")
    ax2.set_xlabel("Gate value")
    ax2.set_ylabel("Count")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    pct_near_zero = (all_gates < 0.01).mean() * 100
    ax2.set_title(
        f"Zoomed — gates < 0.1\n"
        f"({pct_near_zero:.1f}% of all gates are below 0.01)"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gate distribution plot saved → {out_path}")


# ------------------------------------------------------------------
# 4. CSV results logger
# ------------------------------------------------------------------
def save_results_csv(
    lambda_:  float,
    metrics:  dict,
    out_path: str = "outputs/results.csv",
):
    """
    Appends one result row to a CSV file.
    Creates the file with header if it doesn't exist yet.

    Columns: Lambda, Test Accuracy (%), Sparsity Level (%)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Lambda", "Test Accuracy (%)", "Sparsity Level (%)"])
        writer.writerow([
            lambda_,
            round(metrics["test_accuracy"] * 100, 2),
            round(metrics["global_sparsity"] * 100, 2),
        ])

    print(f"Results appended → {out_path}")


# ------------------------------------------------------------------
# Convenience: run full eval pipeline in one call
# ------------------------------------------------------------------
def full_eval(
    model:    SelfPruningNet,
    lambda_:  float,
    device:   str,
    plot:     bool = True,
    csv_path: str  = "outputs/results.csv",
    plot_path:str  = "outputs/gate_distribution.png",
) -> dict:
    """
    Runs compute_metrics → print_sparsity_report → plot → csv in one shot.

    Returns the metrics dict.
    """
    metrics = compute_metrics(model, device)
    print_sparsity_report(metrics, lambda_)

    if plot:
        plot_gate_distribution(model, lambda_, out_path=plot_path)

    save_results_csv(lambda_, metrics, out_path=csv_path)
    return metrics
