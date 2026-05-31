#!/usr/bin/env python
"""PV layer probing (Metric 1: separation).

Extracts v_conservative / v_progressive at each probed layer from the saved
forced activations (trait-validation-filtered prompts), computes the separation
score, saves the vectors, and reports the layer ranking. Causal-effect and
mention/V|M metrics (which need injection-time generation) run separately on the
top separation candidates via future causal-effect analysis.

    uv run python scripts/layer_probing.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pv import layer_probing as lp  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("layer_probing", "layer_probing.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(resolve(args.config))
    out = resolve("data/persona_vectors")

    index, layers = lp.load_activations(str(out / "forced_activations.npz"))
    filtered = json.loads((resolve("data/trait_validation/filtered_prompts.json"))
                          .read_text(encoding="utf-8"))["filtered_prompts"]
    log.info("Extracting vectors at %d layers from %d filtered prompts",
             len(layers), len(filtered))

    vecs = lp.extract_vectors(index, layers, filtered)
    ranking = sorted(vecs, key=lambda L: vecs[L]["separation"], reverse=True)

    metrics = {}
    for L, d in vecs.items():
        np.save(out / f"v_conservative_layer{L}.npy", d["v_conservative"])
        np.save(out / f"v_progressive_layer{L}.npy", d["v_progressive"])
        metrics[L] = {k: round(d[k], 4) for k in ("separation", "norm", "sigma")}
        metrics[L].update(n_conservative=d["n_conservative"], n_progressive=d["n_progressive"])

    (out / "layer_probing_metrics.json").write_text(
        json.dumps({"separation": metrics, "ranking_by_separation": ranking,
                    "n_filtered_prompts": len(filtered)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  PV LAYER PROBING — separation score (Metric 1)")
    print("=" * 60)
    print(f"  {'layer':>6} {'separation':>11} {'||v||':>8} {'σ':>7}")
    for L in sorted(vecs):
        m = metrics[L]
        mark = "  ← top" if L == ranking[0] else ""
        print(f"  {L:>6} {m['separation']:>11.3f} {m['norm']:>8.2f} {m['sigma']:>7.3f}{mark}")
    print(f"\n  separation ranking: {ranking}")
    print(f"  vectors saved: data/persona_vectors/v_*_layer{{L}}.npy")
    print("  next: causal-effect + mention/V|M on top candidates (future causal-effect analysis)")
    print("=" * 60)


if __name__ == "__main__":
    main()
