import torch
import torch.nn as nn
from prunable_layer import PrunableLinear


class SelfPruningNet(nn.Module):
    """
    CNN + Prunable FC classifier for CIFAR-10.

    Args:
        dropout_rate : dropout probability between FC layers (default 0.3)
    """

    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()

        # ------------------------------------------------------------------
        # Feature extractor — standard Conv stack
        # Input: (B, 3, 32, 32)
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # (B, 32, 16, 16)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # (B, 64, 8, 8)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), # (B, 128, 1, 1)
        )

        self.flatten = nn.Flatten()       # (B, 128)

        # ------------------------------------------------------------------
        # Classifier — PrunableLinear layers (where self-pruning happens)
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            PrunableLinear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            PrunableLinear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            PrunableLinear(128, 10),      # 10 classes for CIFAR-10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x                          # raw logits — loss fn applies softmax

    # ------------------------------------------------------------------
    # Convenience helpers for sparsity reporting
    # ------------------------------------------------------------------
    def prunable_layers(self) -> list[PrunableLinear]:
        """Returns all PrunableLinear layers in the model."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def all_gates(self) -> torch.Tensor:
        """Concatenates all gate values across every PrunableLinear layer."""
        return torch.cat([layer.get_gates().flatten()
                          for layer in self.prunable_layers()])

    @torch.no_grad()
    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Fraction of prunable weights whose gate is below `threshold`.
        This is the key metric reported after training.
        """
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()

    def count_prunable_params(self) -> int:
        """Total number of gated (prunable) weight parameters."""
        return sum(layer.weight.numel() for layer in self.prunable_layers())


# ------------------------------------------------------------------
# Quick sanity check
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = SelfPruningNet()
    dummy = torch.randn(4, 3, 32, 32)   # batch of 4 CIFAR-10 images
    out   = model(dummy)

    print("Output shape       :", out.shape)          # expect (4, 10)
    print("Prunable layers    :", len(model.prunable_layers()))
    print("Prunable params    :", model.count_prunable_params())
    print("Global sparsity    :", model.global_sparsity())   # ~0% at init
    print()
    print(model)
