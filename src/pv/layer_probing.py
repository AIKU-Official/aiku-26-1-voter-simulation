"""PV vector extraction + separation metric from saved forced activations.

v_conservative[L] = mean(activations | forced 보수) - mean(activations | forced 진보),
over the trait-validation-filtered prompts. v_progressive[L] = -v_conservative[L].
Separation score = ||v|| / σ (σ = mean per-dim std of all activations at L).
Causal-effect and mention/V|M metrics (which need injection-time generation) live
in the probing script, not here.
"""
from __future__ import annotations

import numpy as np


def load_activations(npz_path: str):
    """Returns (index_df, {L: (N, hidden)}). index columns: item, q_id, direction, rollout."""
    import pandas as pd
    d = np.load(npz_path, allow_pickle=True)
    idx = np.array(d["index"].tolist(), dtype=object)
    index = pd.DataFrame(idx, columns=["item", "q_id", "direction", "rollout"])
    index["q_id"] = index["q_id"].astype(int)
    layers = {int(k[1:]): d[k] for k in d.files if k.startswith("L")}
    return index, layers


def extract_vectors(index, layers: dict, filtered_qids: list[int]) -> dict:
    """Per layer: v_conservative, v_progressive, separation, ||v||, σ, n_A/n_B."""
    keep = index["q_id"].isin(set(filtered_qids))
    is_a = keep & (index["direction"] == "conservative")
    is_b = keep & (index["direction"] == "progressive")
    out = {}
    for L, acts in layers.items():
        a, b = acts[is_a.values], acts[is_b.values]
        v_cons = a.mean(0) - b.mean(0)
        sigma = float(acts[keep.values].std(0).mean())
        norm = float(np.linalg.norm(v_cons))
        out[L] = {
            "v_conservative": v_cons,
            "v_progressive": -v_cons,
            "separation": norm / sigma if sigma else 0.0,
            "norm": norm, "sigma": sigma,
            "n_conservative": int(is_a.sum()), "n_progressive": int(is_b.sum()),
        }
    return out
