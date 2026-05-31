#!/usr/bin/env python
"""Narrative judge + within-sido swap step.

Step 3 backend (--backend): `local` = EXAONE 4.0 32B on GPU (default, no API/cost);
`claude` = Claude API (needs ANTHROPIC_API_KEY). Modes:
  --sanity        judge 5 narratives, print results, validate JSON parse + spread
  --pilot N       judge first N agents only (no swap) — use to time/throughput-check
  (default)       judge all 5,000 → swap → write deliverables
  --skip-judge    reuse existing data/agents/narrative_lean.csv, run swap only

    HF_HOME=/workspace/.cache/huggingface uv run python scripts/narrative_swap.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import narrative_judge as nj  # noqa: E402
from src.agents import swap as swp  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("narrative_swap", "narrative_swap.log")
EXPECTED = {  # B-1 sanity priors (rough)
    "vp": (0.03, 0.15), "p": (0.10, 0.30), "m": (0.10, 0.30),
    "c": (0.10, 0.30), "vc": (0.03, 0.15), "ambiguous": (0.10, 0.40),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--backend", choices=["local", "claude"], default="local")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=8)   # claude backend only
    ap.add_argument("--batch-size", type=int, default=16)  # local backend
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--pilot", type=int, default=None)
    ap.add_argument("--skip-judge", action="store_true")
    # data-parallel sharding (run K processes, each CUDA_VISIBLE_DEVICES=2-GPU group)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--finalize", action="store_true",
                    help="merge _lean_shard*.csv → narrative_lean.csv → swap")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    seed = int(cfg.get("seed", 42))
    out = resolve("data/agents")
    agents = pd.read_parquet(out / "agents_pre_judge.parquet")
    model_id = args.model or (nj.LOCAL_MODEL if args.backend == "local" else nj.DEFAULT_MODEL)
    t0 = time.time()

    _loaded = None  # reuse one local model load across sanity/pilot/full

    def judge(frame, limit=None):
        if args.backend == "local":
            nonlocal _loaded
            if _loaded is None:
                log.info("Loading local judge model %s ...", model_id)
                _loaded = nj.load_local(model_id)
            return nj.run_judge_local(frame, model_id=model_id,
                                      batch_size=args.batch_size, limit=limit,
                                      log=log, _loaded=_loaded)
        return nj.run_judge(frame, model=model_id, workers=args.workers,
                            limit=limit, log=log)

    # --- sanity mode: 5 narratives -----------------------------------------
    if args.sanity:
        log.info("Sanity check: judging 5 narratives with %s (%s)", model_id, args.backend)
        res = judge(agents.head(5))
        for _, r in res.iterrows():
            print(json.dumps(r.to_dict(), ensure_ascii=False))
        ok = res["raw_lean"].notna().all()
        print(f"\nJSON parse ok: {ok}.  Inspect labels above, then run full.")
        return

    # --- shard mode: judge one contiguous block, save, exit (data-parallel)-
    if args.shard_count > 1 and args.shard_index is not None and not args.finalize:
        import numpy as np
        pos = np.array_split(np.arange(len(agents)), args.shard_count)[args.shard_index]
        block = agents.iloc[pos]
        log.info("Shard %d/%d: judging %d narratives...",
                 args.shard_index, args.shard_count, len(block))
        res = judge(block)
        p = out / f"_lean_shard{args.shard_index}.csv"
        res.to_csv(p, index=False)
        log.info("shard → %s (%d rows)", p, len(res))
        return

    # --- Step 3: narrative judge ------------------------------------------
    if args.finalize:
        parts = sorted(out.glob("_lean_shard*.csv"))
        lean = (pd.concat([pd.read_csv(p) for p in parts])
                .drop_duplicates("agent_id").sort_values("agent_id").reset_index(drop=True))
        log.info("Finalize: merged %d shards → %d rows", len(parts), len(lean))
    elif args.skip_judge:
        lean = pd.read_csv(out / "narrative_lean.csv")
        log.info("Loaded existing narrative_lean.csv (%d rows)", len(lean))
    else:
        n = args.pilot or len(agents)
        log.info("Step 3: judging %d narratives via %s (%s)...", n, args.backend, model_id)
        lean = judge(agents, limit=args.pilot)
        lean.to_csv(out / "narrative_lean.csv", index=False)
        log.info("narrative_lean → %s", out / "narrative_lean.csv")

    dist = lean["lean"].value_counts(normalize=True).to_dict()
    log.info("narrative_lean distribution: %s",
             {k: round(v, 3) for k, v in dist.items()})
    flags = [f"{k}={dist.get(k,0):.2f} outside {lo:.2f}-{hi:.2f}"
             for k, (lo, hi) in EXPECTED.items() if not (lo <= dist.get(k, 0) <= hi)]

    if args.pilot:
        print(f"\nPILOT {args.pilot}: distribution {dist}\nflags: {flags or 'none'}")
        return

    # --- Step 4: constrained pair swap ------------------------------------
    merged = agents.merge(lean[["agent_id", "lean", "confidence"]], on="agent_id")
    merged = merged.rename(columns={"lean": "narrative_lean"})
    sw = cfg.get("swap", {})
    log.info("Step 4: constrained swap within %s (dist>=%s, conf>=%s)...",
             sw.get("group_cols"), sw.get("conflict_distance", 2), sw.get("min_confidence", 4))
    swapped, swap_log, swap_summary = swp.run_swap(
        merged, group_cols=sw.get("group_cols"),
        conflict_distance=int(sw.get("conflict_distance", 2)),
        min_confidence=int(sw.get("min_confidence", 4)),
        signal_leans=tuple(sw.get("signal_leans", ("c", "p"))), seed=seed)

    # --- outputs -----------------------------------------------------------
    lean.to_csv(out / "narrative_lean.csv", index=False)
    cols = ["agent_id", "sido17", "orientation_before", "orientation",
            "narrative_lean", "has_unresolved_conflict"]
    swapped[cols].to_csv(out / "post_swap_orientations.csv", index=False)
    (out / "swap_log.json").write_text(
        json.dumps(swap_log, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "swap_summary.json").write_text(
        json.dumps(swap_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    swapped.to_parquet(out / "agents_post_swap.parquet")

    nat = swap_summary["national"]
    t1_3 = {
        "narrative_lean_distribution": {k: round(v, 4) for k, v in dist.items()},
        "narrative_lean_flags": flags,
        "orientation_before_5way": swapped["orientation_before"].value_counts().to_dict(),
        "orientation_after_5way": swapped["orientation"].value_counts().to_dict(),
        "swap_national": nat,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "t1_3_summary.json").write_text(
        json.dumps(t1_3, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("  Narrative judge + swap complete.")
    print("=" * 72)
    print(f"  narrative_lean: {t1_3['narrative_lean_distribution']}")
    print(f"  lean flags: {flags or 'none'}")
    print(f"  swap: {nat['n_swapped_pairs']} pairs, unresolved {nat['unresolved_rate_pct']}% "
          f"(escalate if >20%)")
    print(f"  sanity preserved: {nat['sanity']}")
    print("  Review distributions before BNS seed finalization.")
    print("=" * 72)


if __name__ == "__main__":
    main()
