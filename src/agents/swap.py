"""Step 4 (B-2) — constrained pair-based orientation swap.

Resolves genuine narrative↔orientation conflicts by swapping orientation labels
between two agents in the same demographic cell. Design refinement:
the swap is constrained to (sido × sex × age) cells — not sido-only — so it
preserves the sex×orientation and age×orientation joints (a sido-only swap would
distort them), and a conflict requires opposite sides (dist>=2) AND a confident
narrative lean (conf>=min_confidence). M / ambiguous leans = no signal → skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .orientation import KANG_MEAN_BELIEFS

ORIENTATION_NUM = {"VP": 0, "P": 1, "M": 2, "C": 3, "VC": 4,
                   "vp": 0, "p": 1, "m": 2, "c": 3, "vc": 4}
BELIEF_COLS = [f"belief_{i}" for i in range(5)]
DEFAULT_GROUP_COLS = ["sido17", "sex_label", "age_bucket"]


def orientation_distance(o1: str, o2: str) -> int:
    if o1 not in ORIENTATION_NUM or o2 not in ORIENTATION_NUM:
        return 0  # 'ambiguous'/'m'(no-signal) etc.
    return abs(ORIENTATION_NUM[o1] - ORIENTATION_NUM[o2])


def run_swap(agents: pd.DataFrame, group_cols: list[str] | None = None,
             conflict_distance: int = 2, min_confidence: int = 4,
             signal_leans=("c", "p"), seed: int = 42):
    """Returns (updated_df, swap_log, summary). `agents` needs columns:
    agent_id, orientation, narrative_lean, confidence, belief_0..4, + group_cols."""
    group_cols = group_cols or DEFAULT_GROUP_COLS
    rng = np.random.default_rng(seed + 3)
    df = agents.copy().reset_index(drop=True)
    orient = df["orientation"].to_numpy().copy()
    orient_before = orient.copy()
    beliefs = df[BELIEF_COLS].to_numpy(float).copy()
    lean = df["narrative_lean"].to_numpy()
    conf = df["confidence"].fillna(0).to_numpy()
    unresolved = np.zeros(len(df), dtype=bool)

    def is_conflict(p):
        return (lean[p] in signal_leans and conf[p] >= min_confidence
                and orientation_distance(orient[p], lean[p]) >= conflict_distance)

    swap_log = []
    n_cells_with_conflict = n_swapped = n_unres = n_conflicts = 0

    for _cell, positions in df.groupby(group_cols, observed=True).indices.items():
        conflicts = [(int(p), orientation_distance(orient[p], lean[p]))
                     for p in positions if is_conflict(p)]
        if not conflicts:
            continue
        n_cells_with_conflict += 1
        n_conflicts += len(conflicts)
        conflicts.sort(key=lambda x: -x[1])
        used: set[int] = set()

        for a, dist_a in conflicts:
            if a in used:
                continue
            best_b, best_gain = None, 0
            for b, dist_b in conflicts:
                if b in used or b == a:
                    continue
                gain = (dist_a + dist_b) - (
                    orientation_distance(orient[b], lean[a])
                    + orientation_distance(orient[a], lean[b]))
                if gain > best_gain:
                    best_gain, best_b = gain, b
            if best_b is None:
                unresolved[a] = True
                n_unres += 1
                continue
            b = best_b
            orient[a], orient[b] = orient[b], orient[a]
            for p in (a, b):
                beliefs[p] = np.clip(
                    0.7 * beliefs[p] + 0.3 * KANG_MEAN_BELIEFS[orient[p]]
                    + rng.normal(0, 0.03, 5), 0, 1)
            used.update((a, b))
            n_swapped += 1
            swap_log.append({
                "cell": [df.at[a, c] for c in group_cols],
                "a_id": int(df.at[a, "agent_id"]), "a_old": orient_before[a],
                "a_new": orient[a], "a_lean": lean[a], "a_conf": int(conf[a]),
                "b_id": int(df.at[b, "agent_id"]), "b_old": orient_before[b],
                "b_new": orient[b], "b_lean": lean[b], "b_conf": int(conf[b]),
                "gain": int(best_gain),
            })

    df["orientation_before"] = orient_before
    df["orientation"] = orient
    df["has_unresolved_conflict"] = unresolved
    df[BELIEF_COLS] = beliefs

    # sanity: swapping within group_cols preserves orientation in EVERY such cell,
    # hence all coarser marginals (sido, sex, age, national) are preserved too.
    checks = {
        "national_orientation_preserved": (
            pd.Series(orient_before).value_counts().sort_index().to_dict()
            == pd.Series(orient).value_counts().sort_index().to_dict()),
        "cell_orientation_preserved":
            _cell_table(df, orient_before, group_cols) == _cell_table(df, orient, group_cols),
        "sex_x_orientation_preserved":
            _cell_table(df, orient_before, ["sex_label"]) == _cell_table(df, orient, ["sex_label"]),
        "age_x_orientation_preserved":
            _cell_table(df, orient_before, ["age_bucket"]) == _cell_table(df, orient, ["age_bucket"]),
    }
    nat = {
        "n_agents": len(df), "group_cols": group_cols,
        "conflict_distance": conflict_distance, "min_confidence": min_confidence,
        "n_conflicts": n_conflicts, "n_cells_with_conflict": n_cells_with_conflict,
        "n_swapped_pairs": n_swapped, "n_unresolved": int(unresolved.sum()),
        "unresolved_rate_pct": round(100 * unresolved.sum() / len(df), 2),
        "sanity": checks,
    }
    return df, swap_log, {"national": nat}


def _cell_table(df: pd.DataFrame, orient: np.ndarray, cols: list[str]) -> dict:
    t = pd.crosstab([df[c] for c in cols], pd.Series(orient, index=df.index))
    return {str(k): v for k, v in t.sort_index(axis=1).to_dict("index").items()}
