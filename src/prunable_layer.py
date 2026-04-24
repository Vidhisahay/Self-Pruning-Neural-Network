import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a learnable gate
    per weight. Gates are passed through sigmoid so they live in (0,1).

    Args:
        in_features  : number of input features
        out_features : number of output features
        bias         : whether to include a bias term (default: True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # --- Standard weight parameter (same as nn.Linear) ---
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # --- Gate scores: same shape as weight, also fully learnable ---
        # Initialized to 0 → sigmoid(0) = 0.5, so all gates start half-open.
        # The sparsity loss will drive many of these toward -inf → sigmoid → 0.
        self.gate_scores = nn.Parameter(
            torch.ones(out_features, in_features) * 2.0
        )

        # --- Optional bias (not gated — we only prune weights) ---
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialise weight and bias exactly like nn.Linear does
        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform — same default as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: turn raw gate scores into (0,1) values
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Step 2: element-wise multiply — dead gates zero-out their weight
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Step 3: standard linear operation with the pruned weight matrix
        return F.linear(x, pruned_weights, self.bias)

    # ------------------------------------------------------------------
    # Convenience: expose the current gate values (detached, for metrics)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Returns current gate values in (0,1). Detached from graph."""
        return torch.sigmoid(self.gate_scores)

    @torch.no_grad()
    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of weights whose gate is below `threshold`."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")
