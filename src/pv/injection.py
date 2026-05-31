"""Inference-time PV injection via forward hook (docs/persona_vectors.md Part 3)."""
from __future__ import annotations

import numpy as np

# 5-way orientation → (sign, magnitude-multiplier) on v_conservative.
# +sign steers 보수, −sign steers 진보; strong = 1.5× plain (M = no injection).
ORIENT_COEFF = {"VC": +1.5, "C": +1.0, "M": 0.0, "P": -1.0, "VP": -1.5}


def make_pv_hook(v_conservative: np.ndarray, coeff: float):
    """Add coeff·v to the residual stream at the hooked layer (all token positions)."""
    import torch

    def hook(_m, _i, output):
        if coeff == 0.0:
            return output
        h = output[0] if isinstance(output, tuple) else output
        vt = torch.as_tensor(v_conservative, dtype=h.dtype, device=h.device)
        h = h + coeff * vt
        return (h,) + output[1:] if isinstance(output, tuple) else h

    return hook


def register_pv(model, target_layer: int, v_conservative, coeff: float):
    """Register on model.model.layers[L-1] (its output == hidden_states[L])."""
    return model.model.layers[target_layer - 1].register_forward_hook(
        make_pv_hook(v_conservative, coeff))
