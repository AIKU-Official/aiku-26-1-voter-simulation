"""Steps 2a + 2b — demographic-conditional orientation + Gallup calibration.

2a (Kang-style): P(orientation | sex, age_bucket, region7, edu4) estimated from
KGSS PARTYLR, FINALWT-weighted with exponential recency decay toward the election
year, with a coarsening fallback chain for thin cells. Each agent samples its
5-way orientation from its cell distribution.

2b (Gallup): the 5-way marginal is calibrated to the 26/48/26 (진보/중도/보수)
3-way target by *borderline reassignment* (only P↔M and C↔M are moved; VP/VC are
left intact), then Kang 5-D belief_scores are initialised.

Default design choices:
  - recency: w = 0.5 ** ((election_year - YEAR) / half_life)   [half_life=8]
  - fallback chain: (sex,age,region,edu) → (sex,age,region) → (sex,age) → (region) → global
  - min raw cell n = orientation.min_cell_n (5)
  - calibration only relabels borderline categories (P/M/C); VP/VC preserved
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .demographics import ORIENT_5WAY, THREEWAY

# Kang Table 2 mean beliefs (5-D). VP/VC extrapolated toward extremes —
# Refer to docs/narrative_orientation.md swap_pair note for details.
KANG_MEAN_BELIEFS = {
    "VP": np.array([0.78, 0.75, 0.81, 0.42, 0.68]),
    "P":  np.array([0.75, 0.72, 0.78, 0.45, 0.65]),
    "M":  np.array([0.50, 0.50, 0.50, 0.55, 0.50]),
    "C":  np.array([0.30, 0.28, 0.25, 0.70, 0.35]),
    "VC": np.array([0.27, 0.25, 0.22, 0.73, 0.32]),
}
_FALLBACK_KEYS = [
    ["sex_label", "age_bucket", "region7", "edu4"],
    ["sex_label", "age_bucket", "region7"],
    ["sex_label", "age_bucket"],
    ["region7"],
    [],  # global marginal
]


class OrientationModel:
    """Weighted P(orientation | demographics) with a coarsening fallback chain."""

    def __init__(self, kgss: pd.DataFrame, election_year: int, half_life: float,
                 min_cell_n: int):
        self.min_cell_n = min_cell_n
        w_recency = 0.5 ** ((election_year - kgss["YEAR"].astype(float)) / half_life)
        kgss = kgss.assign(_w=kgss["FINALWT"].astype(float) * w_recency)
        # weighted orientation distribution + raw n at each fallback level
        self.tables = []
        for keys in _FALLBACK_KEYS:
            if keys:
                wsum = kgss.groupby(keys + ["orientation_5way"], observed=True)["_w"].sum()
                raw = kgss.groupby(keys, observed=True).size()
                dist = wsum.unstack("orientation_5way").reindex(columns=ORIENT_5WAY).fillna(0.0)
                dist = dist.div(dist.sum(axis=1), axis=0)
                self.tables.append((keys, dist, raw))
            else:
                wsum = kgss.groupby("orientation_5way", observed=True)["_w"].sum()
                g = wsum.reindex(ORIENT_5WAY).fillna(0.0)
                self.tables.append(([], (g / g.sum()), len(kgss)))

    def probs(self, row: pd.Series) -> tuple[np.ndarray, str]:
        for keys, dist, raw in self.tables:
            if not keys:
                return dist.values.astype(float), "global"
            key = tuple(row[k] for k in keys)
            key = key[0] if len(key) == 1 else key
            if key in dist.index and (raw.get(key, 0) >= self.min_cell_n):
                return dist.loc[key].values.astype(float), "+".join(keys)
        return self.tables[-1][1].values.astype(float), "global"


def assign_orientation(agents: pd.DataFrame, model: OrientationModel,
                       seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = agents.copy()
    orients, levels = [], []
    for _, row in out.iterrows():
        p, lvl = model.probs(row)
        orients.append(rng.choice(ORIENT_5WAY, p=p))
        levels.append(lvl)
    out["orientation_2a"] = orients
    out["orientation_cell_level"] = levels
    return out


def gallup_calibrate(agents: pd.DataFrame, target_3way: dict, seed: int = 42
                     ) -> tuple[pd.DataFrame, dict]:
    """Relabel borderline agents (P↔M, C↔M) so the 3-way marginal hits target."""
    rng = np.random.default_rng(seed + 1)
    out = agents.copy()
    orient = out["orientation_2a"].to_numpy().copy()
    n = len(orient)
    tgt = {k: int(round(v * n)) for k, v in target_3way.items()}

    def idx(cat):
        return np.where(orient == cat)[0]

    three = pd.Series(orient).map(THREEWAY)
    cur = {g: int((three == g).sum()) for g in ["진보", "중도", "보수"]}
    # net flow toward 중도: x from 진보, y from 보수
    x = cur["진보"] - tgt["진보"]
    y = cur["보수"] - tgt["보수"]

    def move(n_move, from_cat, to_cat):
        pool = idx(from_cat)
        k = min(n_move, len(pool))
        if k > 0:
            orient[rng.choice(pool, size=k, replace=False)] = to_cat

    # 진보 side (P↔M)
    if x > 0:
        move(x, "P", "M")
    elif x < 0:
        move(-x, "M", "P")
    # 보수 side (C↔M)
    if y > 0:
        move(y, "C", "M")
    elif y < 0:
        move(-y, "M", "C")

    out["orientation"] = orient
    three_after = pd.Series(orient).map(THREEWAY)
    log = {
        "target_3way_counts": tgt,
        "before_3way": cur,
        "after_3way": {g: int((three_after == g).sum()) for g in ["진보", "중도", "보수"]},
        "flow_P_to_M": int(x), "flow_C_to_M": int(y),
        "before_5way": pd.Series(agents["orientation_2a"]).value_counts().reindex(ORIENT_5WAY).to_dict(),
        "after_5way": pd.Series(orient).value_counts().reindex(ORIENT_5WAY).to_dict(),
    }
    return out, log


def init_belief_scores(agents: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    out = agents.copy()
    bs = np.vstack([
        np.clip(KANG_MEAN_BELIEFS[o] + rng.normal(0, 0.03, 5), 0, 1)
        for o in out["orientation"]
    ])
    for i in range(5):
        out[f"belief_{i}"] = bs[:, i]
    return out
