#!/usr/bin/env python
"""Main inference across 5 cells (docs/architecture.md + prompts/system_prompts/cell_templates/all_cells.md).

Drives Qwen3-30B-A3B (MoE) via vLLM's Python API in-process — no separate HTTP
server. For C_pv/C_full the Qwen3MoeDecoderLayer.forward is monkey-patched
(src/pv/vllm_patches/qwen3_pv.py) and PV state is set per-orientation between
generate() calls. ORIENT_COEFF (src/pv/injection.py) maps orientation → signed
multiplier on alpha_plain (VC=+1.5, C=+1.0, M=0, P=-1.0, VP=-1.5).

Outputs: results/raw_outputs/{cell}/outputs.jsonl with parsed votes per
(agent_id, k_rollout). Run from the package root.

    uv run python scripts/run_inference.py \\
        --cells C_pv --n-per-orient 10 --k-rollouts 1   # pilot
    uv run python scripts/run_inference.py     # full (all cells, all agents)
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

# Workers are spawned subprocesses (vLLM MultiProcExecutor). They re-execute
# Python startup and do NOT inherit sys.path mutations from the driver, so we
# set PYTHONPATH to the package root so they can import src.pv.vllm_patches.*
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

log = get_logger("inference", "inference.log")

ORIENT_ORDER = ["VP", "P", "M", "C", "VC"]
NON_PV_CELLS = {"C_L3", "C_base", "C_bns"}
PV_CELLS = {"C_pv", "C_full"}


def _stratified_subset(agents: pd.DataFrame, per_orient: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(min(per_orient, len(g)), random_state=seed)
             for _, g in agents.groupby("orientation", observed=True)]
    return pd.concat(parts).reset_index(drop=True)


def _build_prompts(agents_df: pd.DataFrame, cell: str, res: dict, tok,
                   include_policy: bool = True) -> list[tuple]:
    """Return list of (agent_id, orientation, chat_template_str)."""
    out = []
    for a in agents_df.itertuples(index=False):
        s, u = build_cell(a, cell, res, include_policy=include_policy)
        try:
            text = tok.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True)
        out.append((a.agent_id, a.orientation, text))
    return out


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _open_jsonl(path: Path):
    """Open a per-cell output file for streaming append. Parent dir created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _stream_rows(f, rows: list[dict]) -> None:
    """Append rows to an already-open jsonl file and fsync so a kill doesn't
    lose the batch that just finished."""
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.flush()
    try:
        os.fsync(f.fileno())
    except OSError:
        pass


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--cells", default="C_L3,C_base,C_bns,C_pv,C_full")
    ap.add_argument("--n-per-orient", type=int, default=None,
                    help="Stratified subsample per orientation (None = all agents).")
    ap.add_argument("--k-rollouts", type=int, default=5,
                    help="vLLM SamplingParams.n — samples per prompt.")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="max_new_tokens for non-L3 cells (L3 forced to 200).")
    ap.add_argument("--tp", type=int, default=4, help="tensor_parallel_size")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--out", default="results/raw_outputs")
    ap.add_argument("--no-policy", action="store_true",
                    help="Use the short candidate block (skip policy text).")
    ap.add_argument("--candidate-mode", default="name_anon",
                    choices=("named", "name_anon", "party_anon", "full_anon"),
                    help="Anonymization level: named (v1 baseline), name_anon "
                         "(default; v3/v4), party_anon (strip party name, keep "
                         "ideology), full_anon (strip everything but number).")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    pv_cfg = cfg["persona_vector"]
    sel = json.loads((resolve("data/persona_vectors") / "SELECTED.json").read_text("utf-8"))
    selected_layer = int(sel["selected_layer"])
    alpha_plain = float(sel["alpha_plain"])
    alpha_strong = float(sel["alpha_strong"])
    log.info("SELECTED: layer=%d alpha_plain=%.2f alpha_strong=%.2f",
             selected_layer, alpha_plain, alpha_strong)

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    unknown = set(cells) - (NON_PV_CELLS | PV_CELLS)
    if unknown:
        raise SystemExit(f"Unknown cells: {unknown}")

    agents = pd.read_parquet(resolve("data/agents/final_agents.parquet"))
    if args.n_per_orient is not None:
        agents = _stratified_subset(agents, args.n_per_orient, int(cfg.get("seed", 42)))
    log.info("agents: %d total (%s)", len(agents),
             dict(agents.groupby("orientation", observed=True).size()))

    from vllm import LLM, SamplingParams
    log.info("loading vLLM: model=%s rev=%s tp=%d max_len=%d",
             cfg["model"]["name"], cfg["model"]["revision"], args.tp, args.max_model_len)
    t0 = time.time()
    llm = LLM(
        model=cfg["model"]["name"],
        revision=cfg["model"]["revision"],
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=True,
        enforce_eager=False,
        trust_remote_code=False,
        seed=int(cfg.get("seed", 42)),
    )
    tagged = qwen3_pv.install(llm)
    log.info("vLLM ready in %.0fs; patched layers per worker: %s",
             time.time() - t0, tagged)

    # Build prompt resources + tokenizer for chat template wrapping.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"])
    include_policy = not args.no_policy
    res = {
        "orient": load_orientation_descriptions(
            str(resolve("prompts/system_prompts/orientation_descriptions.txt"))),
        "candidate": load_candidate_block(
            str(resolve("prompts/user_prompts/candidate_info.txt")),
            include_policy=include_policy,
            candidate_mode=args.candidate_mode),
        "instructions_main": load_instructions_main(
            str(resolve("prompts/user_prompts/instructions_main.txt"))),
    }
    log.info("candidate block: mode=%s, %s policy text",
             args.candidate_mode, "WITH" if include_policy else "WITHOUT")

    v_cons = np.load(resolve("data/persona_vectors") / f"v_conservative_layer{selected_layer}.npy")

    out_root = resolve(args.out)
    nonpv_chunk = max(1, int(args.max_num_seqs))
    for cell in cells:
        log.info("=== Cell %s ===", cell)
        prompts = _build_prompts(agents, cell, res, tok, include_policy=include_policy)
        max_new = 200 if cell == "C_L3" else args.max_tokens
        sp = SamplingParams(
            temperature=args.temperature, top_p=1.0, max_tokens=max_new,
            n=args.k_rollouts, seed=int(cfg.get("seed", 42)))

        out_file = out_root / cell / "outputs.jsonl"
        t_cell = time.time()
        n_rows = 0
        n_ok = 0
        # Stream-write per batch so a kill mid-cell preserves completed batches.
        with _open_jsonl(out_file) as fh:
            if cell in PV_CELLS:
                by_orient: dict[str, list[tuple]] = defaultdict(list)
                for aid, o, text in prompts:
                    by_orient[o].append((aid, text))

                for orient in ORIENT_ORDER:
                    batch = by_orient.get(orient, [])
                    if not batch:
                        continue
                    # M has no PV; everyone else: orientation-strength × alpha_plain.
                    # ORIENT_COEFF already encodes VC=+1.5, C=+1, M=0, P=-1, VP=-1.5,
                    # so coeff = ORIENT_COEFF[orient] * alpha_plain reproduces the
                    # spec's alpha_strong = 1.5 * alpha_plain for VC/VP.
                    coeff = inj.ORIENT_COEFF[orient] * alpha_plain
                    if coeff == 0.0:
                        qwen3_pv.unset_pv(llm)
                    else:
                        qwen3_pv.set_pv(llm, selected_layer, v_cons, coeff)
                    log.info("  cell=%s orient=%s n=%d coeff=%+.3f",
                             cell, orient, len(batch), coeff)
                    t0 = time.time()
                    outs = llm.generate([text for _, text in batch], sp)
                    batch_rows = []
                    for (aid, _), out in zip(batch, outs):
                        for k, s in enumerate(out.outputs):
                            batch_rows.append({
                                "cell": cell, "agent_id": aid, "orientation": orient,
                                "k": k, "coeff": coeff, "text": s.text,
                                **P.parse_vote(s.text),
                            })
                    _stream_rows(fh, batch_rows)
                    n_rows += len(batch_rows)
                    n_ok += sum(1 for r in batch_rows if r.get("ok"))
                    log.info("    %d gens in %.0fs (%.2fs/gen avg) → flushed",
                             len(batch_rows), time.time() - t0,
                             (time.time() - t0) / max(1, len(batch_rows)))
                qwen3_pv.unset_pv(llm)
            else:
                qwen3_pv.unset_pv(llm)
                # Chunk the single generate() so partial progress is persisted.
                for chunk_idx, chunk in enumerate(_chunked(prompts, nonpv_chunk)):
                    texts = [t for _, _, t in chunk]
                    t0 = time.time()
                    outs = llm.generate(texts, sp)
                    batch_rows = []
                    for (aid, orient, _), out in zip(chunk, outs):
                        for k, s in enumerate(out.outputs):
                            batch_rows.append({
                                "cell": cell, "agent_id": aid, "orientation": orient,
                                "k": k, "coeff": 0.0, "text": s.text,
                                **P.parse_vote(s.text),
                            })
                    _stream_rows(fh, batch_rows)
                    n_rows += len(batch_rows)
                    n_ok += sum(1 for r in batch_rows if r.get("ok"))
                    log.info("  cell=%s chunk %d (n=%d) in %.0fs → flushed",
                             cell, chunk_idx, len(batch_rows), time.time() - t0)

        ok_rate = n_ok / max(1, n_rows)
        log.info("cell %s done: %d rows, parse_rate=%.3f (%.0fs total) → %s",
                 cell, n_rows, ok_rate, time.time() - t_cell, out_file)


if __name__ == "__main__":
    main()
