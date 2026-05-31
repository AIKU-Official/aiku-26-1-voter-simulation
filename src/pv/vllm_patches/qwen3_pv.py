"""vLLM monkey-patch: inject α·v_conservative into Qwen3-MoE residual stream
at a target decoder layer (docs/architecture.md "PV Inference Injection").

Architecture: vLLM v1's MultiProcExecutor spawns one worker subprocess per TP
rank. Each worker has its own Python module state, so patch installation AND
per-batch PV state must be broadcast via `LLMEngine.collective_rpc`, which
cloudpickles a module-level function + args and invokes it inside each worker
(with the Worker instance as the first arg). Workers must be able to import
this module — the driver sets PYTHONPATH to the package root so the spawned
workers inherit it.

Usage from the driver:
    qwen3_pv.install(llm)                     # apply patch + tag layer indices
    qwen3_pv.set_pv(llm, layer, vec, coeff)   # configure injection
    llm.generate(prompts, sp)
    qwen3_pv.unset_pv(llm)                    # clear
"""
from __future__ import annotations

import pickle

import numpy as np
import torch

# Per-worker module-level state. Populated via collective_rpc callbacks below;
# read by the patched forward at the target decoder layer.
_STATE = {"layer": None, "coeff": 0.0, "vector": None}
_PATCHED = {"done": False}


def _inject(layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
    coeff = _STATE["coeff"]
    target = _STATE["layer"]
    if target is None or coeff == 0.0 or layer_idx != target:
        return hidden
    v = _STATE["vector"]
    if v is None:
        return hidden
    v = v.to(device=hidden.device, dtype=hidden.dtype)
    return hidden + coeff * v


def _apply_class_patch() -> bool:
    """Monkey-patch Qwen3MoeDecoderLayer.forward. Idempotent per-process."""
    if _PATCHED["done"]:
        return False
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_forward = Qwen3MoeDecoderLayer.forward

    def patched_forward(self, *args, **kwargs):
        out = original_forward(self, *args, **kwargs)
        idx = getattr(self, "_pv_layer_idx", None)
        if idx is None:
            return out
        if isinstance(out, tuple):
            h = _inject(idx, out[0])
            return (h,) + out[1:]
        return _inject(idx, out)

    Qwen3MoeDecoderLayer.forward = patched_forward
    _PATCHED["done"] = True
    return True


def _worker_install(worker) -> int:
    """Patch class + attach layer index to each decoder layer instance."""
    from src.pv.vllm_patches import qwen3_pv as P  # re-import inside worker
    P._apply_class_patch()
    model = worker.model_runner.model
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        return 0
    n = 0
    for i, layer in enumerate(layers):
        layer._pv_layer_idx = i
        n += 1
    return n


def _worker_set_pv(worker, target_layer: int, vector_bytes: bytes, coeff: float) -> bool:
    """`vector_bytes` is a pickled ndarray (vLLM's msgspec encoder mangles raw
    ndarrays into a custom Ext tuple — bytes round-trip cleanly instead)."""
    from src.pv.vllm_patches import qwen3_pv as P
    arr = pickle.loads(vector_bytes)
    P._STATE["layer"] = int(target_layer)
    P._STATE["coeff"] = float(coeff)
    P._STATE["vector"] = torch.as_tensor(np.ascontiguousarray(arr))
    return True


def _worker_unset_pv(worker) -> bool:
    from src.pv.vllm_patches import qwen3_pv as P
    P._STATE["layer"] = None
    P._STATE["coeff"] = 0.0
    P._STATE["vector"] = None
    return True


# ---- driver-side API ---------------------------------------------------------

def install(llm) -> list[int]:
    """Apply the patch and tag layer indices on every worker. Call once."""
    return llm.llm_engine.collective_rpc(_worker_install)


def set_pv(llm, target_layer: int, vector: np.ndarray, coeff: float) -> None:
    """Configure PV injection on every worker for the next generate() call."""
    vec_bytes = pickle.dumps(np.ascontiguousarray(vector),
                             protocol=pickle.HIGHEST_PROTOCOL)
    llm.llm_engine.collective_rpc(
        _worker_set_pv,
        args=(int(target_layer), vec_bytes, float(coeff)),
    )


def unset_pv(llm) -> None:
    llm.llm_engine.collective_rpc(_worker_unset_pv)
