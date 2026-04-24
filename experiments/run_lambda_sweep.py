import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train    import train_model
from evaluate import full_eval, plot_gate_distribution


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
LAMBDAS    = [0.1, 1.0, 5.0]
EPOCHS     = 30
LR         = 1e-3
BATCH_SIZE = 128
OUT_DIR    = "outputs"


# ------------------------------------------------------------------
# Training curves plot
# ------------------------------------------------------------------
def plot_training_curves(all_histories: dict, out_path: str):
    """
    Plots test accuracy and sparsity over epochs for all lambda values.

    Args:
        all_histories : {lambda_val: [epoch_log_dict, ...]}
        out_path      : save location
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training Curves — Lambda Sweep", fontsize=14,
                 fontweight="bold")

    colors = ["#2563eb", "#16a34a", "#dc2626"]
    labels = [f"λ = {l}" for l in LAMBDAS]

    # --- Test accuracy ---
    ax = axes[0]
    for (lambda_, history), color, label in zip(
            all_histories.items(), colors, labels):
        epochs = [h["epoch"]    for h in history]
        accs   = [h["test_acc"] * 100 for h in history]
        ax.plot(epochs, accs, color=color, label=label, linewidth=1.8)

    ax.set_title("Test Accuracy over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    # --- Sparsity level ---
    ax2 = axes[1]
    for (lambda_, history), color, label in zip(
            all_histories.items(), colors, labels):
        epochs    = [h["epoch"]    for h in history]
        sparsity  = [h["sparsity"] * 100 for h in history]
        ax2.plot(epochs, sparsity, color=color, label=label, linewidth=1.8)

    ax2.set_title("Sparsity Level over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Sparsity (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {out_path}")


# ------------------------------------------------------------------
# Summary table printer
# ------------------------------------------------------------------
def print_summary_table(all_results: list[dict]):
    """Prints the final lambda vs accuracy vs sparsity table."""
    print("\n" + "="*52)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity':>15}")
    print("="*52)
    for r in all_results:
        print(
            f"  {r['lambda_']:<12.0e} "
            f"{r['test_accuracy']*100:>14.2f}% "
            f"{r['sparsity']*100:>14.2f}%"
        )
    print("="*52 + "\n")


# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
def run_sweep(epochs: int = EPOCHS, quick: bool = False):
    """
    Trains one model per lambda, evaluates, and generates all outputs.

    Args:
        epochs : epochs per run (use fewer for a quick smoke-test)
        quick  : if True, uses only 5 epochs (for testing the pipeline)
    """
    if quick:
        epochs = 5
        print("Quick mode: running 5 epochs per lambda")

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # Clear any previous CSV so we don't append duplicate rows
    csv_path = os.path.join(OUT_DIR, "results.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    all_histories = {}
    all_results   = []

    for lambda_ in LAMBDAS:
        # --- Train ---
        result = train_model(
            lambda_=lambda_,
            epochs=epochs,
            lr=LR,
            batch_size=BATCH_SIZE,
            verbose=True,
        )

        all_histories[lambda_] = result["history"]

        # --- Evaluate + save per-lambda gate plot ---
        plot_path = os.path.join(OUT_DIR, f"gate_dist_lambda_{lambda_:.0e}.png")
        metrics   = full_eval(
            model=result["model"],
            lambda_=lambda_,
            device=device,
            plot=True,
            csv_path=csv_path,
            plot_path=plot_path,
        )

        all_results.append({
            "lambda_":       lambda_,
            "test_accuracy": metrics["test_accuracy"],
            "sparsity":      metrics["global_sparsity"],
        })

    # --- Training curves across all lambdas ---
    plot_training_curves(
        all_histories,
        out_path=os.path.join(OUT_DIR, "training_curves.png"),
    )

    # --- Final summary ---
    print_summary_table(all_results)
    print(f"All outputs saved to ./{OUT_DIR}/")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lambda sweep experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Epochs per lambda run (default: 30)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 5 epochs per lambda (smoke test)")
    args = parser.parse_args()

    run_sweep(epochs=args.epochs, quick=args.quick)