"""Step 5 (Steps 7-8) — BNS seed finalization.

Each agent is given a *real* KGSS 2016 respondent's factor-score vector (direct
resampling from group_respondent_pool.pkl, fallback chain), then a primary +
secondary belief seed is selected from seed_templates.txt based on its strongest
*political* factors. 정치효능감 (political=False) is excluded from selection.
Agents with no strong political factor (all |score| < moderate_thresh) get a
single meta (중도) seed.
"""
from __future__ import annotations

import pickle
import re

import numpy as np
import pandas as pd

# region7 name → 광역 5 (pool primary keys use these values)
REGION7_TO_5WAY = {
    "서울": "수도권", "경기": "수도권", "강원": "강원·제주", "충청": "충청",
    "경상": "영남", "전라": "호남", "제주": "강원·제주",
}
# factor_meta name → seed_templates.txt section label
SEED_LABEL = {
    "경제·재분배(정부책임)": "F1",
    "대북·안보": "F2",
    "시장·민영화": "시장·민영화",
}
MODERATE_THRESH = 0.5   # all |political scores| below → 중도 (meta seed)
STRONG_THRESH = 1.0     # |primary score| above → strong, else moderate


def load_pool(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_seed_templates(path: str) -> tuple[dict, list[str]]:
    """Returns ({'<label> <Direction> <Strength>': [seeds]}, meta_seeds)."""
    templates: dict[str, list[str]] = {}
    cur = None
    for line in open(path, encoding="utf-8").read().splitlines():
        h = re.match(r"###\s+(.*)", line)
        if h:
            # normalize: drop a trailing "(...)" annotation on the header
            cur = re.sub(r"\s*\(.*\)\s*$", "", h.group(1).strip())
            templates.setdefault(cur, [])
            continue
        if cur:
            templates[cur].extend(re.findall(r"`([^`]+)`", line))
    metas = [s for hdr, seeds in templates.items()
             if hdr.startswith("Type") for s in seeds]
    return templates, metas


def resample_factor_scores(pool: dict, age_bucket, sex_label, region5, orientation,
                           rng) -> tuple[np.ndarray, str]:
    """Direct resampling with fallback: (age,sex,region5,orient) → (region5,orient) → orient."""
    minc = pool["min_cell_size"]
    key = (age_bucket, sex_label, region5, orientation)
    if key in pool["primary"] and len(pool["primary"][key]) >= minc:
        arr, lvl = pool["primary"][key], "primary"
    elif (region5, orientation) in pool["fallback_region_orient"]:
        arr, lvl = pool["fallback_region_orient"][(region5, orientation)], "region_orient"
    else:
        arr, lvl = pool["fallback_orient"][orientation], "orient"
    return arr[rng.integers(0, len(arr))], lvl


def select_seeds(scores: np.ndarray, factor_meta: list[dict], templates: dict,
                 metas: list[str], rng) -> dict:
    """Pick primary+secondary political seeds, or a meta seed for 중도 agents."""
    pol = [(i, m["name"]) for i, m in enumerate(factor_meta) if m["political"]]
    if all(abs(scores[i]) < MODERATE_THRESH for i, _ in pol):
        return {"seed_type": "meta", "bns_seed_1": str(rng.choice(metas)),
                "bns_seed_2": "", "bns_key": "meta"}

    order = sorted(pol, key=lambda x: -abs(scores[x[0]]))
    (pi, pname), (si, sname) = order[0], order[1]

    def direction(s):
        return "Conservative" if s > 0 else "Progressive"

    pstr = "Strong" if abs(scores[pi]) > STRONG_THRESH else "Moderate"
    key1 = f"{SEED_LABEL[pname]} {direction(scores[pi])} {pstr}"
    key2 = f"{SEED_LABEL[sname]} {direction(scores[si])} Moderate"
    return {
        "seed_type": "primary_secondary",
        "primary_factor": pname, "primary_dir": direction(scores[pi]),
        "primary_strength": pstr, "secondary_factor": sname,
        "secondary_dir": direction(scores[si]),
        "bns_seed_1": str(rng.choice(templates[key1])),
        "bns_seed_2": str(rng.choice(templates[key2])),
        "bns_key": f"{key1} | {key2}",
    }
