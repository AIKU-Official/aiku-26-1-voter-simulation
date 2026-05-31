#!/usr/bin/env python
"""Qwen3-30B forced-response generation for PV (text + activations).

One pass over (50 prompts × {conservative, progressive} × rollouts) producing
response text (for EXAONE trait validation) and per-probe-layer activations
(for PV extraction). Shardable for data-parallel runs:

    CUDA_VISIBLE_DEVICES=0,1 ... --shard-index 0 --shard-count 3
Then merge shards with --finalize.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pv import generate as g  # noqa: E402
from src.pv.contrastive import parse_contrastive  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("extract_persona_vectors", "extract_persona_vectors.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--rollouts", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    pv = cfg["persona_vector"]
    probe = pv["probe_layers"]
    rollouts = args.rollouts or int(pv["rollouts"])
    gen_batch = int(pv.get("gen_batch", 16))
    temp = 0.7  # spec: higher than main exp for diversity
    out = resolve("data/persona_vectors")
    out.mkdir(parents=True, exist_ok=True)

    qs = parse_contrastive(str(resolve("prompts/pv_contrastive/contrastive_prompts.txt")))
    items = g.build_items(qs, rollouts)

    if args.finalize:
        resp, idx, layer_acc = [], [], {L: [] for L in probe}
        for p in sorted(out.glob("_pvgen_shard*.npz")):
            d = np.load(p, allow_pickle=True)
            idx.append(d["index"])
            for L in probe:
                layer_acc[L].append(d[f"L{L}"])
        index = np.concatenate(idx)
        for p in sorted(out.glob("_pvgen_resp*.jsonl")):
            resp.extend(p.read_text(encoding="utf-8").splitlines())
        order = np.argsort([json.loads(r)["item"] for r in resp])
        (out / "forced_responses.jsonl").write_text(
            "\n".join(resp[i] for i in order) + "\n", encoding="utf-8")
        np.savez(out / "forced_activations.npz",
                 index=index, **{f"L{L}": np.concatenate(layer_acc[L]) for L in probe})
        log.info("Finalized: %d responses, activations %s",
                 len(resp), {L: np.concatenate(layer_acc[L]).shape for L in probe})
        return

    # shard the work items
    pos = np.array_split(np.arange(len(items)), args.shard_count)[args.shard_index]
    my = [items[i] for i in pos]
    log.info("Shard %d/%d: %d items (rollouts=%d, batch=%d, layers=%s)",
             args.shard_index, args.shard_count, len(my), rollouts, gen_batch, probe)

    tok, model = g.load_qwen(pv["name"]) if "name" in pv else g.load_qwen(cfg["model"]["name"])
    t0 = time.time()
    resp_lines, index, acts = [], [], {L: [] for L in probe}
    for b in range(0, len(my), gen_batch):
        batch = my[b:b + gen_batch]
        texts, a = g.generate_with_activations(
            tok, model, [it["prompt"] for it in batch], probe,
            max_new_tokens=int(pv["max_new_tokens"]), temperature=temp)
        for k, it in enumerate(batch):
            gi = int(pos[b + k])
            resp_lines.append(json.dumps(
                {"item": gi, "q_id": it["q_id"], "direction": it["direction"],
                 "rollout": it["rollout"], "text": texts[k]}, ensure_ascii=False))
            index.append((gi, it["q_id"], it["direction"], it["rollout"]))
        for L in probe:
            acts[L].append(a[L])
        log.info("  %d/%d (%.0fs)", min(b + gen_batch, len(my)), len(my), time.time() - t0)

    (out / f"_pvgen_resp{args.shard_index}.jsonl").write_text(
        "\n".join(resp_lines), encoding="utf-8")
    np.savez(out / f"_pvgen_shard{args.shard_index}.npz",
             index=np.array(index, dtype=object),
             **{f"L{L}": np.concatenate(acts[L]) for L in probe})
    log.info("Shard %d done in %.0fs", args.shard_index, time.time() - t0)


if __name__ == "__main__":
    main()
