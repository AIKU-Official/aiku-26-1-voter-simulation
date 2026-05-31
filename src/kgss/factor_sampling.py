"""KGSS EFA Steps 9-10 — group respondent pools for BNS direct resampling.

Each demographic×orientation cell stores the *actual* factor-score vectors of
the real KGSS respondents in it. BNS gives an agent a real respondent's scores
(sampled with replacement), preserving real within-cell variance rather than
injecting Gaussian noise around a cell mean (efa_spec.md Step 9, Design Change
Log 2026-05-25). A fallback chain handles thin cells.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

# 5 광역 from the cumulative file's 7-way REGION codes
# (1 서울, 2 경기, 3 강원, 4 충청, 5 경상, 6 전라, 7 제주).
REGION_5WAY_MAP = {
    1: "수도권",   # 서울
    2: "수도권",   # 경기 (인천 folded into 경기 in the 7-way coding)
    5: "영남",     # 경상
    6: "호남",     # 전라
    4: "충청",     # 충청
    3: "강원·제주",  # 강원
    7: "강원·제주",  # 제주
}

SEX_MAP = {1: "남성", 2: "여성"}
ORIENT_MAP = {1: "VP", 2: "P", 3: "M", 4: "C", 5: "VC"}

# MUST match src/agents/demographics.py (AGE_BUCKETS/_AGE_BINS) so the BNS
# resampling primary key (age×sex×region5×orient) matches between KGSS pool and agents.
AGE_BINS = [18, 29, 39, 49, 59, 69, 200]
AGE_LABELS = ["19-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# Full demographic cell (efa_spec.md Step 9: region_5way, no education).
GROUP_COLS = ["age_bracket", "sex_label", "region_5way", "orientation_5way"]
MIN_CELL_SIZE = 10  # primary cell must have >= this many real respondents


def add_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Attach recoded demographic + orientation columns; drop rows that can't be
    placed (missing age/sex/region/orientation)."""
    out = df.copy()
    out["age_bracket"] = pd.cut(out["AGE"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    out["sex_label"] = out["SEX"].map(SEX_MAP)
    out["region_5way"] = out["REGION"].map(REGION_5WAY_MAP)
    out["orientation_5way"] = out["PARTYLR"].map(ORIENT_MAP)
    return out.dropna(subset=GROUP_COLS)


def build_group_respondent_pool(
    df: pd.DataFrame, factor_cols: list[str], min_cell_size: int = MIN_CELL_SIZE
) -> dict:
    """Primary (full cell) pools + 2 fallback levels (region×orient, orient).

    Each pool is an ``(n_in_cell, n_factors)`` float array of real respondents'
    factor scores. Returns the dict pickled to group_respondent_pool.pkl plus a
    sparsity / resampling-coverage report.
    """
    primary = {
        key: sub[factor_cols].to_numpy(float)
        for key, sub in df.groupby(GROUP_COLS, observed=True)
    }
    fb_region_orient = {
        key: sub[factor_cols].to_numpy(float)
        for key, sub in df.groupby(["region_5way", "orientation_5way"], observed=True)
    }
    fb_orient = {
        key: sub[factor_cols].to_numpy(float)
        for key, sub in df.groupby("orientation_5way", observed=True)
    }

    primary_ok = {k: v for k, v in primary.items() if len(v) >= min_cell_size}
    covered = sum(len(v) for v in primary_ok.values())
    report = {
        "n_respondents_grouped": int(len(df)),
        "n_distinct_full_cells": len(primary),
        "n_primary_cells_ge_min": len(primary_ok),
        "min_cell_size": min_cell_size,
        "respondents_in_usable_primary": int(covered),
        "pct_in_usable_primary": round(100 * covered / len(df), 1),
        "mean_n_per_full_cell": round(np.mean([len(v) for v in primary.values()]), 1),
        "mean_n_per_region_orient": round(
            np.mean([len(v) for v in fb_region_orient.values()]), 1
        ),
        "n_region_orient_cells": len(fb_region_orient),
        "orient_pool_sizes": {k: len(v) for k, v in sorted(fb_orient.items())},
    }

    bundle = {
        "primary": primary,
        "fallback_region_orient": fb_region_orient,
        "fallback_orient": fb_orient,
        "factor_names": [c.replace("_score", "") for c in factor_cols],
        "n_factors": len(factor_cols),
        "group_cols": GROUP_COLS,
        "min_cell_size": min_cell_size,
        "sparsity_report": report,
    }
    return bundle


def resampling_coverage(
    bundle: dict, orient_marginal: dict[str, float], n_agents: int, seed: int = 42
) -> dict:
    """Project, over ``n_agents`` drawn at the orientation-only fallback level
    (worst case for re-use), how many distinct real respondents get used and the
    heaviest re-use — a quick diversity check. The true per-cell mix is
    measured at agent construction once agents carry full demographics.
    """
    rng = np.random.default_rng(seed)
    pools = bundle["fallback_orient"]
    total_unique, max_reuse = 0, 0
    for orient, frac in orient_marginal.items():
        pool = pools.get(orient)
        if pool is None or frac <= 0:
            continue
        n = int(round(n_agents * frac))
        picks = rng.integers(0, len(pool), size=n)
        counts = np.bincount(picks, minlength=len(pool))
        total_unique += int((counts > 0).sum())
        max_reuse = max(max_reuse, int(counts.max()))
    return {
        "projected_n_agents": n_agents,
        "projected_unique_respondents_used": total_unique,
        "total_pool_respondents": int(sum(len(p) for p in pools.values())),
        "max_single_respondent_reuse": max_reuse,
        "note": "orientation-only fallback projection (upper bound on re-use)",
    }


def save_group_respondent_pool(bundle: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
