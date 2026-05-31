#!/usr/bin/env python
"""Agent construction.

Steps implemented here (no API needed):
  1   Nemotron 5,000 persona sampling + narrative assembly
  2a  KGSS demographic-conditional 5-way orientation
  2b  Gallup 26/48/26 calibration + Kang belief_scores

Steps 3 (narrative judge, Claude API) and 4 (within-sido swap) run afterwards via
narrative_swap.py / narrative_swap.py once ANTHROPIC_API_KEY is set. This
script stops before the judge and writes data/agents/agents_pre_judge.parquet.

    uv run python scripts/construct_agents.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import demographics as dm  # noqa: E402
from src.agents import orientation as ori  # noqa: E402
from src.agents import sampling  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("construct_agents", "construct_agents.log")
KGSS_COLS = ["YEAR", "PARTYLR", "FINALWT", "SEX", "AGE", "EDUC", "REGION"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(resolve(args.config))
    seed = int(cfg.get("seed", 42))
    n = int(cfg.get("n_personas", 5000))
    oc = cfg.get("orientation", {})
    election_year = int(str(cfg["election"]["date"])[:4])
    half_life = float(oc.get("recency_half_life_years", 8))
    min_cell_n = int(oc.get("min_cell_n", 5))
    gallup = {k: float(v) for k, v in oc["gallup_target"].items()}

    out_dir = resolve("data/agents")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1 — Nemotron sampling -------------------------------------------
    log.info("Step 1: sampling %d personas...", n)
    agents, n_pool = sampling.sample_personas(
        cfg["paths"]["nvidia_data_glob"], n, seed=seed
    )
    miss = agents[["sex_label", "age_bucket", "region7", "edu4"]].isna().sum().to_dict()
    log.info("Sampled %d / %d pool. demographic NaNs: %s", len(agents), n_pool, miss)

    # Step 2a — KGSS conditional orientation -------------------------------
    log.info("Step 2a: building KGSS orientation model...")
    kgss_raw, _ = pyreadstat.read_sav(str(resolve(cfg["paths"]["kgss_sav"])), usecols=KGSS_COLS)
    kgss = dm.recode_kgss(kgss_raw)
    log.info("KGSS usable for conditional: %d respondents (all waves)", len(kgss))
    model = ori.OrientationModel(kgss, election_year, half_life, min_cell_n)
    agents = ori.assign_orientation(agents, model, seed=seed)
    lvl_counts = agents["orientation_cell_level"].value_counts().to_dict()
    log.info("2a cell-level usage: %s", lvl_counts)

    # Step 2b — Gallup calibration + belief_scores -------------------------
    log.info("Step 2b: Gallup calibration to %s", gallup)
    agents, gallup_log = ori.gallup_calibrate(agents, gallup, seed=seed)
    agents = ori.init_belief_scores(agents, seed=seed)
    log.info("3-way after: %s", gallup_log["after_3way"])

    # ---- write ------------------------------------------------------------
    keep = [
        "agent_id", "uuid", "sex_label", "age_bucket", "sido17", "region7", "edu4",
        "occupation", "orientation_2a", "orientation_cell_level", "orientation",
        *(f"belief_{i}" for i in range(5)), "narrative",
        *sampling.NARRATIVE_FIELDS,
    ]
    agents[keep].to_parquet(out_dir / "agents_pre_judge.parquet")

    summary = {
        "n_agents": len(agents),
        "n_pool": n_pool,
        "election_year": election_year,
        "recency_half_life_years": half_life,
        "min_cell_n": min_cell_n,
        "orientation_5way": agents["orientation"].value_counts().reindex(dm.ORIENT_5WAY).to_dict(),
        "orientation_3way": agents["orientation"].map(dm.THREEWAY).value_counts().to_dict(),
        "gallup": gallup_log,
        "cell_level_usage": lvl_counts,
        "region7_x_3way": (
            agents.assign(g=agents["orientation"].map(dm.THREEWAY))
            .groupby("region7", observed=True)["g"].value_counts().unstack().fillna(0).astype(int).to_dict("index")
        ),
        "sido17_counts": agents["sido17"].value_counts().to_dict(),
        "belief_means_by_orientation": (
            agents.groupby("orientation", observed=True)[[f"belief_{i}" for i in range(5)]]
            .mean().round(3).to_dict("index")
        ),
        "narrative_char_stats": {
            "mean": int(agents["narrative"].str.len().mean()),
            "min": int(agents["narrative"].str.len().min()),
            "max": int(agents["narrative"].str.len().max()),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "orientation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    log.info("Done in %.1fs. → %s", summary["elapsed_sec"], out_dir / "agents_pre_judge.parquet")
    print("\n" + "=" * 72)
    print("  PHASE 1.2 Steps 1-2b complete (Nemotron sample + orientation).")
    print("=" * 72)
    print(f"  5-way: {summary['orientation_5way']}")
    print(f"  3-way: {summary['orientation_3way']}  (Gallup target {gallup})")
    print(f"  agents → {out_dir / 'agents_pre_judge.parquet'}")
    print("  Next (needs ANTHROPIC_API_KEY): narrative judge → within-sido swap.")
    print("=" * 72)


if __name__ == "__main__":
    main()
