#!/usr/bin/env python
"""α sweep using vLLM (single LLM load, all α values reuse weights).

Iterates a list of α_plain candidates on a single cell (default C_full) over a
stratified subset of agents, and writes per-α JSONL outputs that can be analyzed
like a normal inference run. ORIENT_COEFF (src/pv/injection.py) maps orientation
to a signed multiplier on α (VC=+1.5, C=+1, M=0, P=-1, VP=-1.5), so PV state is
set per orientation batch for each α.

Output layout:
    results/alpha_sweep_<cell>/alpha_<value>/outputs.jsonl
    results/alpha_sweep_<cell>/summary.csv   (after analysis)

Run from package root:
    HF_HOME=/workspace/.cache/huggingface uv run python \\
        scripts/alpha_sweep.py --cell C_full --n-per-orient 100 \\
        --alphas 0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

# Workers are spawned subprocesses (vLLM v1 MultiProcExecutor) and do not
# inherit sys.path mutations from the driver — propagate via PYTHONPATH.
_PKG_ROOT = str(Path(__file__).resolve().parents[1])
os.environ["PYTHONPATH"] = _PKG_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, _PKG_ROOT)

import numpy as np
import pandas as pd

from src.inference import parse as P  # noqa: E402
from src.inference.prompt_builder import (build_cell, load_candidate_block,  # noqa: E402
                                          load_instructions_main,
                                          load_orientation_descriptions)
from src.pv import injection as inj  # noqa: E402
from src.pv.vllm_patches import qwen3_pv  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("alpha_sweep_vllm", "alpha_sweep_vllm.log")
ORIENT_ORDER = ["VP", "P", "M", "C", "VC"]


def _stratified_subset(agents: pd.DataFrame, per_orient: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(min(per_orient, len(g)), random_state=seed)
             for _, g in agents.groupby("orientation", observed=True)]
    return pd.concat(parts).reset_index(drop=True)


def _open_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _stream_rows(f, rows):
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--cell", default="C_full",
                    choices=("C_pv", "C_full"))
    ap.add_argument("--alphas", default="0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0",
                    help="comma-separated list of α_plain values to sweep")
    ap.add_argument("--n-per-orient", type=int, default=100,
                    help="stratified subsample per orientation")
    ap.add_argument("--candidate-mode", default="named",
                    choices=("named", "name_anon", "party_anon", "full_anon"))
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--out", default=None,
                    help="output dir; default results/alpha_sweep_<cell>")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    pv_cfg = cfg["persona_vector"]
    sel = json.loads((resolve("data/persona_vectors") / "SELECTED.json").read_text("utf-8"))
    selected_layer = int(sel["selected_layer"])
    alphas = [float(x) for x in args.alphas.split(",")]
    log.info("SELECTED: layer=%d; sweeping alphas=%s on cell=%s",
             selected_layer, alphas, args.cell)

    agents = pd.read_parquet(resolve("data/agents/final_agents.parquet"))
    sub = _stratified_subset(agents, args.n_per_orient, int(cfg.get("seed", 42)))
    log.info("stratified subset: %d agents (%s)",
             len(sub), dict(sub.groupby("orientation", observed=True).size()))

    include_policy = not args.no_policy
    res = {
        "orient": load_orientation_descriptions(
            str(resolve("prompts/system_prompts/orientation_descriptions.txt"))),
        "candidate": load_candidate_block(
            str(resolve("prompts/user_prompts/candidate_info.txt")),
            include_policy=include_policy, candidate_mode=args.candidate_mode),
        "instructions_main": load_instructions_main(
            str(resolve("prompts/user_prompts/instructions_main.txt"))),
    }
    log.info("candidate block: mode=%s, %s policy text",
             args.candidate_mode, "WITH" if include_policy else "WITHOUT")

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(
        model=cfg["model"]["name"], revision=cfg["model"]["revision"],
        dtype="bfloat16", tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=True, enforce_eager=False,
        seed=int(cfg.get("seed", 42)),
    )
    qwen3_pv.install(llm)
    log.info("vLLM ready in %.0fs", time.time() - t0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"],
                                        revision=cfg["model"]["revision"])
    sp = SamplingParams(temperature=args.temperature, top_p=1.0,
                        max_tokens=args.max_tokens, n=1,
                        seed=int(cfg.get("seed", 42)))

    # Pre-build prompts once (cell config doesn't change across α).
    prompts_by_orient: dict[str, list[tuple]] = defaultdict(list)
    for a in sub.itertuples(index=False):
        s, u = build_cell(a, args.cell, res, include_policy=include_policy)
        try:
            text = tok.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True)
        prompts_by_orient[a.orientation].append((a.agent_id, text))

    v_cons = np.load(resolve("data/persona_vectors") /
                     f"v_conservative_layer{selected_layer}.npy")

    out_root = resolve(args.out or f"results/alpha_sweep_{args.cell}")
    log.info("output root: %s", out_root)

    for alpha in alphas:
        log.info("=== alpha_plain=%.2f ===", alpha)
        out_file = out_root / f"alpha_{alpha:.2f}" / "outputs.jsonl"
        n_rows = 0
        n_ok = 0
        t_a = time.time()
        with _open_jsonl(out_file) as fh:
            for orient in ORIENT_ORDER:
                batch = prompts_by_orient.get(orient, [])
                if not batch:
                    continue
                coeff = inj.ORIENT_COEFF[orient] * alpha
                if coeff == 0.0:
                    qwen3_pv.unset_pv(llm)
                else:
                    qwen3_pv.set_pv(llm, selected_layer, v_cons, coeff)
                t0 = time.time()
                outs = llm.generate([t for _, t in batch], sp)
                batch_rows = []
                for (aid, _), out in zip(batch, outs):
                    for k, s in enumerate(out.outputs):
                        batch_rows.append({
                            "cell": args.cell, "alpha_plain": alpha,
                            "agent_id": aid, "orientation": orient,
                            "k": k, "coeff": coeff, "text": s.text,
                            **P.parse_vote(s.text),
                        })
                _stream_rows(fh, batch_rows)
                n_rows += len(batch_rows)
                n_ok += sum(1 for r in batch_rows if r.get("ok"))
                log.info("  orient=%s n=%d coeff=%+.3f in %.0fs",
                         orient, len(batch_rows), coeff, time.time() - t0)
        qwen3_pv.unset_pv(llm)
        log.info("alpha=%.2f done: %d rows, parse_rate=%.3f (%.0fs)",
                 alpha, n_rows, n_ok / max(1, n_rows), time.time() - t_a)


if __name__ == "__main__":
    main()
