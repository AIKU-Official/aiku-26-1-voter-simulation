#!/usr/bin/env python
"""BNS seed assignment → final_agents.parquet.

Direct-resamples each agent's factor scores from a real KGSS respondent in its
(age×sex×region5×orientation) cell, then selects primary+secondary belief seeds
(or a 중도 meta seed) from seed_templates.txt. Produces the final agent set for
persona vector setup / inference.

    uv run python scripts/assign_bns_seeds.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import bns  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("assign_bns_seeds", "assign_bns_seeds.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(resolve(args.config))
    seed = int(cfg.get("seed", 42))
    out = resolve("data/agents")
    t0 = time.time()

    agents = pd.read_parquet(out / "agents_post_swap.parquet")
    pool = bns.load_pool(str(resolve("data/kgss/group_respondent_pool.pkl")))
    fmeta = pool["factor_meta"]
    fnames = pool["factor_names"]
    templates, metas = bns.parse_seed_templates(str(resolve("prompts/bns_seeds/seed_templates.txt")))

    # verify every political seed key the selector can request actually exists
    need = [f"{bns.SEED_LABEL[m['name']]} {d} {s}"
            for m in fmeta if m["political"]
            for d in ("Progressive", "Conservative") for s in ("Strong", "Moderate")]
    missing = [k for k in need if not templates.get(k)]
    if missing:
        raise SystemExit(f"seed_templates.txt missing sections: {missing}")
    log.info("Parsed seeds: %d political keys ok, %d meta seeds", len(need), len(metas))

    rng = np.random.default_rng(seed)
    score_cols = [f"{n}_score" for n in fnames]
    recs = []
    for a in agents.itertuples(index=False):
        region5 = bns.REGION7_TO_5WAY[a.region7]
        scores, lvl = bns.resample_factor_scores(
            pool, a.age_bucket, a.sex_label, region5, a.orientation, rng)
        sel = bns.select_seeds(scores, fmeta, templates, metas, rng)
        rec = {"agent_id": a.agent_id, "resample_level": lvl}
        rec.update({c: float(scores[i]) for i, c in enumerate(score_cols)})
        rec.update(sel)
        recs.append(rec)
    bns_df = pd.DataFrame(recs)

    final = agents.merge(bns_df, on="agent_id")
    final.to_parquet(out / "final_agents.parquet")
    bns_df.to_csv(out / "bns_seeds.csv", index=False)

    summary = {
        "n_agents": len(final),
        "seed_type": bns_df["seed_type"].value_counts().to_dict(),
        "resample_level": bns_df["resample_level"].value_counts().to_dict(),
        "primary_factor": bns_df["primary_factor"].value_counts(dropna=True).to_dict(),
        "primary_direction": bns_df["primary_dir"].value_counts(dropna=True).to_dict(),
        "primary_strength": bns_df["primary_strength"].value_counts(dropna=True).to_dict(),
        "factor_score_stats": {
            c: {"mean": round(float(bns_df[c].mean()), 3),
                "std": round(float(bns_df[c].std()), 3)} for c in score_cols},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "bns_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    log.info("Done in %.1fs → %s", summary["elapsed_sec"], out / "final_agents.parquet")
    print("\n" + "=" * 72)
    print("  BNS seeds finalized — agent construction complete.")
    print("=" * 72)
    print(f"  final_agents: {len(final)} → {out / 'final_agents.parquet'}")
    print(f"  seed_type: {summary['seed_type']}")
    print(f"  primary_factor: {summary['primary_factor']}")
    print(f"  resample_level: {summary['resample_level']}")
    print("  Next: persona vector setup (trait validation / layer probing / α sweep).")
    print("=" * 72)


if __name__ == "__main__":
    main()
