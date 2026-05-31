#!/usr/bin/env python
"""EXAONE trait validation of the 50 contrastive prompts.

Loads the Qwen forced responses, has EXAONE score each on (보수, 진보), applies
the relative-separation criterion, and writes the filtered prompt set + summary.

    CUDA_VISIBLE_DEVICES=0,1 HF_HOME=... uv run python scripts/trait_validation.py
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

from src.agents.narrative_judge import LOCAL_MODEL, load_local  # noqa: E402
from src.pv import trait_validation as tv  # noqa: E402
from src.pv.contrastive import parse_contrastive  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("trait_validation", "trait_validation.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(resolve(args.config))
    tvc = cfg["persona_vector"]["trait_validation"]

    pv_dir = resolve("data/persona_vectors")
    out = resolve("data/trait_validation")
    out.mkdir(parents=True, exist_ok=True)

    resp = pd.DataFrame(
        json.loads(l) for l in (pv_dir / "forced_responses.jsonl").read_text(
            encoding="utf-8").splitlines())
    qs = parse_contrastive(str(resolve("prompts/pv_contrastive/contrastive_prompts.txt")))
    qs_by_id = {q["q_id"]: q for q in qs}
    log.info("Judging %d forced responses with EXAONE...", len(resp))

    t0 = time.time()
    judged = tv.judge_responses(resp, qs_by_id, LOCAL_MODEL,
                                batch_size=args.batch_size, log=log,
                                _loaded=load_local(LOCAL_MODEL))
    judged.to_json(out / "trait_judgments.jsonl", orient="records", lines=True,
                   force_ascii=False)
    n_fail = judged["보수"].isna().sum()
    log.info("Judged in %.0fs (parse failures: %d)", time.time() - t0, n_fail)

    summary = tv.aggregate(judged, sep_margin=float(tvc["sep_margin"]),
                           min_pass_per_set=int(tvc["min_pass_per_set"]))
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "filtered_prompts.json").write_text(
        json.dumps({"filtered_prompts": summary["filtered_prompts"],
                    "n_pass": summary["n_pass"]}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print("\n" + "=" * 72)
    print("  PHASE 2 — Trait validation (EXAONE cross-judge, relative separation).")
    print("=" * 72)
    print(f"  prompts passing: {summary['n_pass']}/{summary['n_prompts']} "
          f"(threshold {summary['min_pass_per_set']}) → set_passes={summary['set_passes']}")
    print(f"  mean sep: 보수축 {summary['mean_sep_conservative_axis']}, "
          f"진보축 {summary['mean_sep_progressive_axis']} (margin {summary['sep_margin']})")
    print(f"  failed prompts: {summary['failed_prompts']}")
    print(f"  → filtered_prompts.json ({summary['n_pass']} prompts for PV extraction)")
    print("=" * 72)


if __name__ == "__main__":
    main()
