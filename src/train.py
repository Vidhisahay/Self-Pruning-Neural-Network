import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model import SelfPruningNet
from loss  import total_loss


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def get_dataloaders(batch_size: int = 128, num_workers: int = 2):
    """
    Returns CIFAR-10 train and test DataLoaders.

    Train transforms: random crop + horizontal flip for regularisation.
    Test  transforms: only normalise (no augmentation).
    """
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
        root="./data", train=True,  download=True, transform=train_tf
    )
    test_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ------------------------------------------------------------------
# Single epoch helpers
# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, lambda_, device):
    """
    Runs one full pass over the training set.

    Returns:
        avg_total_loss, avg_cls_loss, avg_sp_loss, train_accuracy
    """
    model.train()

    total_loss_sum = cls_loss_sum = sp_loss_sum = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        t_loss, c_loss, s_loss = total_loss(logits, labels, model, lambda_)

        t_loss.backward()
        optimizer.step()

        # Hard pruning: gates that have fallen below 0.05 get pushed to -10.
        # sigmoid(-10) ≈ 0.00005 — decisively dead.
        # Without this, sigmoid asymptotically approaches 0 but never reaches
        # the 1e-2 threshold, so sparsity stays at 0% forever.
        with torch.no_grad():
            for layer in model.prunable_layers():
                gates = torch.sigmoid(layer.gate_scores)
                dead  = gates < 0.05
                layer.gate_scores.data[dead] = -10.0

        # Accumulate metrics
        batch = images.size(0)
        total_loss_sum += t_loss.item() * batch
        cls_loss_sum   += c_loss.item() * batch
        sp_loss_sum    += s_loss.item() * batch

        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += batch

    n = len(loader.dataset)
    return (
        total_loss_sum / n,
        cls_loss_sum   / n,
        sp_loss_sum    / n,
        correct / total,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Runs inference on the test set.

    Returns:
        test_accuracy (float), global_sparsity (float)
    """
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds  = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return correct / total, model.global_sparsity()


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------
def train_model(
    lambda_:     float = 1e-3,
    epochs:      int   = 30,
    lr:          float = 1e-3,
    batch_size:  int   = 128,
    dropout:     float = 0.3,
    device:      str   = None,
    verbose:     bool  = True,
) -> dict:
    """
    Full training run for one lambda value.

    Args:
        lambda_    : sparsity regularisation weight
        epochs     : number of training epochs
        lr         : Adam learning rate
        batch_size : mini-batch size
        dropout    : dropout rate in classifier
        device     : 'cuda', 'mps', or 'cpu' (auto-detected if None)
        verbose    : print per-epoch logs

    Returns:
        dict with keys:
            model          : trained SelfPruningNet
            history        : list of per-epoch metric dicts
            test_accuracy  : final test accuracy
            sparsity       : final global sparsity level
            lambda_        : the lambda used
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training  |  lambda={lambda_}  |  device={device}")
        print(f"{'='*60}")

    # --- Data ---
    train_loader, test_loader = get_dataloaders(batch_size)

    # --- Model + Optimiser ---
    model     = SelfPruningNet(dropout_rate=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Cosine annealing: smoothly decays LR to near-zero over training.
    # Helps gates settle cleanly at 0 or 1 by the end.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        avg_total, avg_cls, avg_sp, train_acc = train_one_epoch(
            model, train_loader, optimizer, lambda_, device
        )

        # --- Evaluate ---
        test_acc, sparsity = evaluate(model, test_loader, device)

        scheduler.step()

        # --- Log ---
        epoch_log = {
            "epoch":      epoch,
            "total_loss": avg_total,
            "cls_loss":   avg_cls,
            "sp_loss":    avg_sp,
            "train_acc":  train_acc,
            "test_acc":   test_acc,
            "sparsity":   sparsity,
            "lr":         scheduler.get_last_lr()[0],
        }
        history.append(epoch_log)

        if test_acc > best_acc:
            best_acc = test_acc

        if verbose:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"Loss {avg_total:.4f} (cls {avg_cls:.4f}, sp {avg_sp:.1f}) | "
                f"Train {train_acc*100:.1f}% | "
                f"Test {test_acc*100:.1f}% | "
                f"Sparsity {sparsity*100:.1f}% | "
                f"{elapsed:.1f}s"
            )

    final_acc, final_sparsity = evaluate(model, test_loader, device)

    if verbose:
        print(f"\nFinal  →  Test Accuracy: {final_acc*100:.2f}%  |  "
              f"Sparsity: {final_sparsity*100:.2f}%")

    return {
        "model":         model,
        "history":       history,
        "test_accuracy": final_acc,
        "sparsity":      final_sparsity,
        "lambda_":       lambda_,
    }


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train self-pruning network")
    parser.add_argument("--lambda_",    type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--dropout",    type=float, default=0.3)
    args = parser.parse_args()

    train_model(
        lambda_=args.lambda_,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
    )