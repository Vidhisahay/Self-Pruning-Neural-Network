import torch
import torch.nn as nn
from model import SelfPruningNet


def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0

    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total = total + gates.sum()
        count += gates.numel()

    return total / count   # ← normalize: now always in (0, 1)


def total_loss(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    model:   SelfPruningNet,
    lambda_: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combines classification loss and sparsity penalty.

    Args:
        logits   : raw model outputs, shape (B, 10)
        targets  : ground-truth class indices, shape (B,)
        model    : the SelfPruningNet (needed to access gate params)
        lambda_  : sparsity weight — higher = more aggressive pruning

    Returns:
        (total, cls_loss, sp_loss) — all scalar tensors
        Returning the components separately makes logging clean.
    """
    cls_loss = nn.CrossEntropyLoss()(logits, targets)
    sp_loss  = sparsity_loss(model)
    total    = cls_loss + lambda_ * sp_loss
    return total, cls_loss, sp_loss


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    from model import SelfPruningNet
    import torch

    model   = SelfPruningNet()
    logits  = torch.randn(8, 10)              # fake batch of 8
    targets = torch.randint(0, 10, (8,))

    # At init: all gate_scores = 0 → all gates = 0.5
    # SparsityLoss should equal: num_prunable_params * 0.5
    expected_sp = model.count_prunable_params() * 0.5
    sp = sparsity_loss(model)
    print(f"Expected sparsity loss : {expected_sp:.1f}")
    print(f"Actual sparsity loss   : {sp.item():.1f}")

    t, c, s = total_loss(logits, targets, model, lambda_=1e-3)
    print(f"\nlambda=1e-3")
    print(f"  cls_loss    : {c.item():.4f}")
    print(f"  sp_loss     : {s.item():.2f}")
    print(f"  total_loss  : {t.item():.4f}")

    # Check gradient flows back to gate_scores
    t.backward()
    for layer in model.prunable_layers():
        assert layer.gate_scores.grad is not None, "No grad on gate_scores!"
        assert layer.weight.grad is not None,      "No grad on weight!"
    print("\nGradient check: PASSED — gate_scores and weight both receive grads")
