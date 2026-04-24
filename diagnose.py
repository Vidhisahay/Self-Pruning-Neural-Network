"""
diagnose.py
-----------
Runs ONE forward+backward pass and prints:
- gradient magnitude on gate_scores vs weight
- what the sparsity gradient looks like at different gate values
- whether gate_scores are actually being updated by optimizer

Run from project root:
    python diagnose.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import SelfPruningNet
from loss  import total_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- Tiny dataloader ---
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                       download=False, transform=tf)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    # ---------------------------------------------------------------
    # TEST 1: Gradient magnitudes at different gate initializations
    # ---------------------------------------------------------------
    print("=" * 60)
    print("TEST 1: Gradient magnitudes at different gate_score inits")
    print("=" * 60)

    for init_val in [0.0, 2.0, -2.0]:
        model = SelfPruningNet().to(device)

        # Set all gate_scores to init_val
        with torch.no_grad():
            for layer in model.prunable_layers():
                layer.gate_scores.fill_(init_val)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        logits = model(images)
        t_loss, c_loss, s_loss = total_loss(logits, labels, model, lambda_=1.0)
        t_loss.backward()

        for i, layer in enumerate(model.prunable_layers()):
            gs_grad = layer.gate_scores.grad.abs()
            w_grad  = layer.weight.grad.abs()
            gate_val = torch.sigmoid(layer.gate_scores).mean().item()
            print(f"  init={init_val:+.1f} | Layer {i} | "
                  f"gate={gate_val:.4f} | "
                  f"gate_scores.grad mean={gs_grad.mean():.6f} max={gs_grad.max():.6f} | "
                  f"weight.grad mean={w_grad.mean():.6f}")

    # ---------------------------------------------------------------
    # TEST 2: Does optimizer step actually change gate_scores?
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("TEST 2: Do gate_scores actually update after optimizer.step()?")
    print("=" * 60)

    model = SelfPruningNet().to(device)
    with torch.no_grad():
        for layer in model.prunable_layers():
            layer.gate_scores.fill_(2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before = [layer.gate_scores.data.clone()
              for layer in model.prunable_layers()]

    optimizer.zero_grad()
    logits = model(images)
    t_loss, _, _ = total_loss(logits, labels, model, lambda_=1.0)
    t_loss.backward()
    optimizer.step()

    after = [layer.gate_scores.data.clone()
             for layer in model.prunable_layers()]

    for i, (b, a) in enumerate(zip(before, after)):
        delta = (a - b).abs()
        print(f"  Layer {i}: gate_scores changed by "
              f"mean={delta.mean():.6f}  max={delta.max():.6f}")

    # ---------------------------------------------------------------
    # TEST 3: Simulate 20 steps — do gates trend downward?
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("TEST 3: Gate mean over 20 optimizer steps (lambda=5.0)")
    print("=" * 60)

    model = SelfPruningNet().to(device)
    with torch.no_grad():
        for layer in model.prunable_layers():
            layer.gate_scores.fill_(2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(20):
        optimizer.zero_grad()
        logits = model(images)
        t_loss, c_loss, s_loss = total_loss(logits, labels, model, lambda_=5.0)
        t_loss.backward()
        optimizer.step()

        if step % 4 == 0 or step == 19:
            gates = model.all_gates()
            print(f"  Step {step+1:>2} | "
                  f"gate mean={gates.mean():.4f}  "
                  f"min={gates.min():.4f}  "
                  f"max={gates.max():.4f}  "
                  f"std={gates.std():.4f}  "
                  f"cls={c_loss.item():.4f}  sp={s_loss.item():.4f}")

    # ---------------------------------------------------------------
    # TEST 4: Check if gate_scores are in optimizer param groups
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("TEST 4: Are gate_scores in the optimizer?")
    print("=" * 60)
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    all_opt_ids  = {id(p) for group in optimizer.param_groups
                    for p in group["params"]}
    gate_ids     = {id(layer.gate_scores) for layer in model.prunable_layers()}
    weight_ids   = {id(layer.weight)      for layer in model.prunable_layers()}

    print(f"  gate_scores in optimizer : {gate_ids.issubset(all_opt_ids)}")
    print(f"  weights in optimizer     : {weight_ids.issubset(all_opt_ids)}")
    print(f"  Total params in optimizer: {sum(p.numel() for g in optimizer.param_groups for p in g['params']):,}")
    print(f"  Total model params       : {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()