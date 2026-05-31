#!/usr/bin/env python
"""KGSS factor analysis (docs/factor_analysis.md Steps 1-9).

Runs the full EFA pipeline, writes all output artifacts, and STOPS at decision
review (factor count / naming / TBD reverse-coding / direction labels always
require manual confirmation — docs/factor_analysis.md §Review Checkpoint).

    uv run python scripts/kgss_factor_analysis.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Put the package root on sys.path so `import src...` works under `uv run`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kgss import efa, factor_sampling, report  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("kgss_factor_analysis", "kgss_factor_analysis.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--rotation", default="varimax")
    ap.add_argument(
        "--n-factors",
        type=int,
        default=None,
        help="Override the parallel-analysis factor count.",
    )
    ap.add_argument("--pa-iter", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    efa_cfg = cfg.get("efa", {})
    sav = resolve(cfg["paths"]["kgss_sav"])
    backbone_path = resolve("docs/kgss_items.csv")
    seed = int(cfg.get("seed", 42))
    n_personas = int(cfg.get("n_personas", 5000))

    wave = efa_cfg.get("wave")  # single-wave EFA (Option A) — e.g. 2016
    na_thresh = float(efa_cfg.get("na_thresh_frac", 0.30))
    rotation = args.rotation if args.rotation != "varimax" else efa_cfg.get("rotation", "varimax")
    cutoff = float(efa_cfg.get("loading_cutoff", 0.40))
    pa_iter = args.pa_iter if args.pa_iter != 100 else int(efa_cfg.get("pa_iter", 100))

    out_dir = resolve(cfg["paths"].get("personas_dir", "data/personas")).parent / "kgss"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("KGSS .sav: %s", sav)
    log.info("Output dir: %s", out_dir)
    t0 = time.time()

    # Step 1 — load + clean -------------------------------------------------
    backbone = efa.load_backbone(str(backbone_path))
    drop_items = efa_cfg.get("drop_items", [])
    if drop_items:
        backbone = backbone[~backbone["var_name"].isin(drop_items)].reset_index(drop=True)
        log.info("Dropped %d unattached items: %s", len(drop_items), drop_items)
    identities = efa_cfg.get("factor_identities", [])
    items = backbone["var_name"].tolist()
    log.info("Loading + cleaning KGSS (%d items, wave=%s)...", len(items), wave)
    clean = efa.load_and_clean(str(sav), backbone, na_thresh_frac=na_thresh, wave=wave)
    log.info("Records: %d → %d after missing-drop", clean.n_original, clean.n_after_drop)

    # Step 2-3 — reverse code + standardize ---------------------------------
    df_rev = efa.apply_reverse_coding(clean.df_clean, backbone)
    df_z = efa.standardize(df_rev, items)

    # Step 4 — KMO + Bartlett ----------------------------------------------
    kmo = efa.kmo_bartlett(df_z)
    log.info("KMO=%.3f  Bartlett p=%.1e", kmo["kmo_model"], kmo["bartlett_p"])
    if kmo["kmo_model"] <= 0.6:
        log.warning("KMO <= 0.6 — correlation structure weak (efa_spec.md Step 4).")

    # Step 5 — parallel analysis -------------------------------------------
    pa = efa.parallel_analysis(df_z, n_iter=pa_iter, seed=seed)
    n_factors = args.n_factors or pa["n_factors"]
    log.info("Parallel analysis n_factors=%d (using %d)", pa["n_factors"], n_factors)

    # Step 6-8 — EFA (varimax primary) -------------------------------------
    fa, loadings_emp, variance = efa.run_efa(df_z, n_factors, rotation=rotation)
    scores_emp = efa.factor_scores(fa, df_z, n_factors)
    log.info("Variance explained: %.1f%%", 100 * variance["prop_var"].sum())

    # Step 7b — map empirical factors to confirmed identities + orient sign -
    mapres = efa.map_factors_to_identities(loadings_emp, identities)
    loadings, scores, direction = efa.orient_factors(
        loadings_emp, scores_emp, mapres["by_factor"]
    )
    assignments, cross, none_items = efa.assign_items_to_factors(loadings, cutoff=cutoff)
    tbd = efa.resolve_tbd_reverse(loadings, assignments, backbone)
    score_dist = efa.score_distribution_report(scores)
    log.info("Factors: %s", list(loadings.columns))

    # Promax supplementary (empirical, unoriented — Varimax-vs-Promax for review)
    _, loadings_promax, _ = efa.run_efa(df_z, n_factors, rotation="promax")

    # Step 9 — group respondent pools (direct resampling) ------------------
    df_scored = clean.df_clean.copy()
    df_scored[scores.columns] = scores.values
    df_grouped = factor_sampling.add_demographics(df_scored)
    bundle = factor_sampling.build_group_respondent_pool(df_grouped, list(scores.columns))
    bundle["factor_meta"] = direction.to_dict("records")  # names + political flags for BNS
    orient_marginal = (
        df_grouped["orientation_5way"].value_counts(normalize=True).to_dict()
    )
    resampling = factor_sampling.resampling_coverage(
        bundle, orient_marginal, n_personas, seed=seed
    )
    log.info(
        "Pools: %d/%d full cells ≥min; %.1f%% in usable primary; "
        "%d agents → %d/%d unique respondents",
        bundle["sparsity_report"]["n_primary_cells_ge_min"],
        bundle["sparsity_report"]["n_distinct_full_cells"],
        bundle["sparsity_report"]["pct_in_usable_primary"],
        resampling["projected_n_agents"],
        resampling["projected_unique_respondents_used"],
        resampling["total_pool_respondents"],
    )

    # ---- write outputs ----------------------------------------------------
    df_scored.to_parquet(out_dir / "df_clean.parquet")
    loadings.round(4).to_csv(out_dir / "efa_loadings.csv")
    loadings_promax.round(4).to_csv(out_dir / "efa_loadings_promax.csv")
    scores.to_parquet(out_dir / "factor_scores.parquet")
    factor_sampling.save_group_respondent_pool(
        bundle, str(out_dir / "group_respondent_pool.pkl")
    )

    artifact = {
        "demo_cols": efa.DEMO_COLS,
        "rotation": rotation,
        "wave": wave,
        "resampling": resampling,
        "clean": {
            "n_original": clean.n_original,
            "n_after_drop": clean.n_after_drop,
            "per_item_missing": clean.per_item_missing,
            "wave_coverage": clean.wave_coverage,
        },
        "kmo": kmo,
        "pa": pa,
        "loadings": loadings,
        "variance": variance,
        "direction": direction,
        "match_score": mapres["match_score"],
        "tbd": tbd,
        "cross": cross,
        "none_items": none_items,
        "score_dist": score_dist,
        "sparsity": bundle["sparsity_report"],
    }
    html = report.build_report(artifact)
    (out_dir / "efa_report.html").write_text(html, encoding="utf-8")

    # machine-readable summary for the review checkpoint
    summary = {
        "n_original": clean.n_original,
        "n_clean": clean.n_after_drop,
        "kmo_model": kmo["kmo_model"],
        "bartlett_p": kmo["bartlett_p"],
        "pa_n_factors": pa["n_factors"],
        "kaiser_n_factors": pa["kaiser_n"],
        "n_factors_used": n_factors,
        "variance_explained": float(variance["prop_var"].sum()),
        "n_cross_loading_items": int(len(cross)),
        "dropped_items": drop_items,
        "none_items": none_items,
        "factors": direction.to_dict("records"),
        "tbd_recommendations": tbd.to_dict("records"),
        "sparsity": bundle["sparsity_report"],
        "resampling": resampling,
        "wave": wave,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "efa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    log.info("Done in %.1fs. Outputs in %s", summary["elapsed_sec"], out_dir)
    print("\n" + "=" * 72)
    print("  KGSS factor analysis complete (Option A, 2016 wave).")
    print("=" * 72)
    print(f"  report : {out_dir / 'efa_report.html'}")
    print(f"  summary: {out_dir / 'efa_summary.json'}")
    print(f"  KMO={kmo['kmo_model']:.3f}  n_factors={n_factors}  "
          f"var={100 * variance['prop_var'].sum():.1f}%  cross-load={len(cross)}")
    print(f"  factors: {list(loadings.columns)}")
    print("  Next: agent construction.")
    print("=" * 72)


if __name__ == "__main__":
    main()
