"""KGSS EFA pipeline — docs/factor_analysis.md Steps 1-8.

Pure functions over DataFrames; no I/O side effects except where noted. The
orchestration (and all file writes) live in scripts/kgss_factor_analysis.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyreadstat
from factor_analyzer import (
    FactorAnalyzer,
    calculate_bartlett_sphericity,
    calculate_kmo,
)

# Demographic / anchor columns pulled alongside the 42 attitude items.
# NOTE (vs efa_spec.md): the cumulative file has REGION (7-way) rather than
# PROVINCE/SIDO, matching configs/config.yaml's `region(7)` conditional design.
DEMO_COLS = ["YEAR", "AGE", "SEX", "EDUC", "REGION", "PARTYLR", "FINALWT"]

# Prior factor hypotheses (backbone 'factor' column) → proposed names.
EXPECTED_NAMES = {
    "F1": "경제·재분배",
    "F2": "대북·안보",
    "F3": "사회 가치관",
    "F4": "이민·다문화",
    "F5": "노동",
    "F6": "정치 제도",
}

SCALE_MAX = {"5-point": 5, "4-point": 4}


def load_backbone(path: str) -> pd.DataFrame:
    """Load the 42-item backbone metadata."""
    bb = pd.read_csv(path)
    assert "var_name" in bb and "factor" in bb and "reverse" in bb and "scale" in bb
    return bb


# ---------------------------------------------------------------------------
# Step 1 — load + missing handling
# ---------------------------------------------------------------------------
@dataclass
class CleanResult:
    df_clean: pd.DataFrame
    n_original: int
    n_after_drop: int
    per_item_missing: pd.Series
    wave_coverage: pd.DataFrame


def load_and_clean(
    sav_path: str,
    backbone: pd.DataFrame,
    na_thresh_frac: float = 0.30,
    wave: int | None = None,
) -> CleanResult:
    """Read the .sav, optionally filter to a single ``wave`` (YEAR), subset to
    items + demographics, map missing codes to NaN, and drop respondents missing
    more than ``na_thresh_frac`` of the items.

    Single-wave filtering is required here: the 42 backbone items are split
    across non-overlapping KGSS waves, so EFA runs on the jointly-measured 2016
    module only (efa_spec.md Option A). Missing handling is *scale-aware*: any
    value outside an item's valid range (1..5 for 5-point, 1..4 for 4-point)
    becomes NaN, subsuming every KGSS missing code (-8 DK, -1 IAP, 8/9, etc.).
    """
    items = backbone["var_name"].tolist()
    df, _meta = pyreadstat.read_sav(sav_path)
    if wave is not None:
        df = df[df["YEAR"].astype("Int64") == wave].copy()
    n_original = len(df)

    keep = items + [c for c in DEMO_COLS if c in df.columns]
    df_items = df[keep].copy()

    # Scale-aware valid-range masking.
    for _, row in backbone.iterrows():
        v = row["var_name"]
        hi = SCALE_MAX[row["scale"]]
        col = df_items[v]
        df_items[v] = col.where((col >= 1) & (col <= hi), np.nan)

    thresh = int(np.ceil(len(items) * (1 - na_thresh_frac)))  # min non-NA items
    df_clean = df_items.dropna(subset=items, thresh=thresh).copy()

    per_item_missing = df_clean[items].isna().sum().sort_values(ascending=False)
    wave_coverage = (
        df_items.assign(_n_items=df_items[items].notna().sum(axis=1))
        .groupby("YEAR")["_n_items"]
        .agg(n_respondents="size", mean_items_answered="mean")
        .round(1)
    )

    return CleanResult(
        df_clean=df_clean,
        n_original=n_original,
        n_after_drop=len(df_clean),
        per_item_missing=per_item_missing,
        wave_coverage=wave_coverage,
    )


# ---------------------------------------------------------------------------
# Step 2 — direction unification (reverse coding)
# ---------------------------------------------------------------------------
def apply_reverse_coding(df: pd.DataFrame, backbone: pd.DataFrame) -> pd.DataFrame:
    """Apply the backbone 'reverse' column. TBD-EFA and FALSE are left as-is
    (TBD items get resolved retroactively from EFA loading signs at the review checkpoint)."""
    out = df.copy()
    for _, row in backbone.iterrows():
        if row["reverse"] == "TRUE":
            hi = SCALE_MAX[row["scale"]]
            out[row["var_name"]] = (hi + 1) - out[row["var_name"]]
    return out


# ---------------------------------------------------------------------------
# Step 3 — standardize
# ---------------------------------------------------------------------------
def standardize(df: pd.DataFrame, items: list[str]) -> pd.DataFrame:
    """Z-score each item, then mean-impute remaining NaNs (factor_analyzer needs
    a complete matrix). Mean of a z-scored column is 0, so imputation is neutral."""
    z = df[items].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    return z.fillna(0.0)


# ---------------------------------------------------------------------------
# Step 4 — KMO + Bartlett
# ---------------------------------------------------------------------------
def kmo_bartlett(df_z: pd.DataFrame) -> dict:
    kmo_all, kmo_model = calculate_kmo(df_z)
    chi_sq, p_val = calculate_bartlett_sphericity(df_z)
    return {
        "kmo_model": float(kmo_model),
        "kmo_per_item": pd.Series(kmo_all, index=df_z.columns).sort_values(),
        "bartlett_chi_sq": float(chi_sq),
        "bartlett_p": float(p_val),
    }


# ---------------------------------------------------------------------------
# Step 5 — parallel analysis (Horn) for factor count
# ---------------------------------------------------------------------------
def _corr_eigenvalues(x: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(x, rowvar=False)
    ev = np.linalg.eigvalsh(corr)
    return np.sort(ev)[::-1]


def parallel_analysis(
    df_z: pd.DataFrame, n_iter: int = 100, percentile: int = 95, seed: int = 42
) -> dict:
    """Horn's parallel analysis on the correlation matrix. Returns real vs
    random (percentile) eigenvalues and the retained factor count."""
    rng = np.random.default_rng(seed)
    x = df_z.values
    n, p = x.shape
    real_ev = _corr_eigenvalues(x)
    rand = np.empty((n_iter, p))
    for i in range(n_iter):
        rand[i] = _corr_eigenvalues(rng.standard_normal((n, p)))
    rand_ev = np.percentile(rand, percentile, axis=0)
    n_factors = int(np.sum(real_ev > rand_ev))
    return {
        "real_eigenvalues": real_ev,
        "random_eigenvalues": rand_ev,
        "n_factors": n_factors,
        "kaiser_n": int(np.sum(real_ev > 1.0)),
    }


# ---------------------------------------------------------------------------
# Step 6 — EFA
# ---------------------------------------------------------------------------
def run_efa(df_z: pd.DataFrame, n_factors: int, rotation: str = "varimax"):
    """Fit FactorAnalyzer and return (model, loadings_df, variance_df)."""
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    fa.fit(df_z)
    cols = [f"F{i + 1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=df_z.columns, columns=cols)
    ssl, prop, cum = fa.get_factor_variance()
    variance = pd.DataFrame(
        {"ss_loadings": ssl, "prop_var": prop, "cum_var": cum}, index=cols
    )
    return fa, loadings, variance


# ---------------------------------------------------------------------------
# Step 7 — loading interpretation + naming proposal
# ---------------------------------------------------------------------------
def assign_items_to_factors(loadings: pd.DataFrame, cutoff: float = 0.40):
    """Assign each item to its highest-|loading| factor; flag cross-loaders
    (a 2nd factor also above cutoff) and items that load nowhere."""
    assignments, cross, none_items = {}, [], []
    for item in loadings.index:
        absl = loadings.loc[item].abs().sort_values(ascending=False)
        top_factor, top_val = absl.index[0], absl.iloc[0]
        if top_val < cutoff:
            assignments[item] = "NONE"
            none_items.append(item)
            continue
        assignments[item] = top_factor
        if absl.iloc[1] > cutoff:
            cross.append(
                {
                    "item": item,
                    "primary": top_factor,
                    "primary_loading": round(float(loadings.loc[item, top_factor]), 2),
                    "secondary": absl.index[1],
                    "secondary_loading": round(
                        float(loadings.loc[item, absl.index[1]]), 2
                    ),
                }
            )
    return assignments, pd.DataFrame(cross), none_items


def propose_factor_names(
    assignments: dict, backbone: pd.DataFrame, loadings: pd.DataFrame
) -> pd.DataFrame:
    """Map each empirical factor (F1..Fk columns) to a proposed name by majority
    vote of the prior hypothesis (backbone 'factor') among items assigned to it.
    Direction is decided at the EFA review checkpoint; we only report dominant-item signs.
    """
    prior = dict(zip(backbone["var_name"], backbone["factor"]))
    rows = []
    for fac in loadings.columns:
        members = [it for it, f in assignments.items() if f == fac]
        prior_votes = pd.Series([prior[m] for m in members]).value_counts()
        dominant_prior = prior_votes.index[0] if len(prior_votes) else "—"
        # mean loading sign of the dominant-prior members (direction hint)
        dom_members = [m for m in members if prior[m] == dominant_prior]
        mean_sign = (
            float(np.mean([loadings.loc[m, fac] for m in dom_members]))
            if dom_members
            else 0.0
        )
        rows.append(
            {
                "empirical_factor": fac,
                "n_items": len(members),
                "dominant_prior": dominant_prior,
                "proposed_name": EXPECTED_NAMES.get(dominant_prior, "?"),
                "prior_vote_breakdown": prior_votes.to_dict(),
                "dominant_mean_loading": round(mean_sign, 2),
                "members": members,
            }
        )
    return pd.DataFrame(rows)


def resolve_tbd_reverse(
    loadings: pd.DataFrame, assignments: dict, backbone: pd.DataFrame
) -> pd.DataFrame:
    """For each TBD-EFA item, report its assigned factor and whether its loading
    sign agrees with the majority sign of that factor's non-TBD members — i.e.
    whether it should be reverse-coded."""
    tbd = backbone[backbone["reverse"] == "TBD-EFA"]["var_name"].tolist()
    rev = {v: r for v, r in zip(backbone["var_name"], backbone["reverse"])}
    rows = []
    for item in tbd:
        fac = assignments.get(item, "NONE")
        if fac == "NONE":
            rows.append(
                {"item": item, "assigned_factor": "NONE", "recommendation": "drop?"}
            )
            continue
        peers = [
            it
            for it, f in assignments.items()
            if f == fac and it != item and rev.get(it) != "TBD-EFA"
        ]
        item_sign = np.sign(loadings.loc[item, fac])
        peer_signs = [np.sign(loadings.loc[p, fac]) for p in peers]
        majority = np.sign(sum(peer_signs)) if peer_signs else item_sign
        agrees = bool(item_sign == majority) or not peer_signs
        rows.append(
            {
                "item": item,
                "assigned_factor": fac,
                "loading": round(float(loadings.loc[item, fac]), 2),
                "sign_agrees_with_peers": agrees,
                "recommendation": "keep as-is" if agrees else "reverse-code",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7b — map empirical factors to confirmed identities + orient sign
# ---------------------------------------------------------------------------
def map_factors_to_identities(loadings: pd.DataFrame, identities: list[dict]) -> dict:
    """Greedily 1-1 match each empirical factor column to the confirmed identity
    whose signature items load most strongly on it (robust to factor ordering).

    Returns {empirical_col: identity_dict} plus a match-quality table.
    """
    facs = list(loadings.columns)
    score = pd.DataFrame(index=facs, columns=range(len(identities)), dtype=float)
    for fi, fac in enumerate(facs):
        for ii, idn in enumerate(identities):
            sig = [s for s in idn["signature"] if s in loadings.index]
            score.loc[fac, ii] = loadings.loc[sig, fac].abs().mean() if sig else 0.0

    mapping, used_id, used_fac = {}, set(), set()
    pairs = sorted(
        ((score.loc[f, i], f, i) for f in facs for i in range(len(identities))),
        reverse=True,
    )
    for _val, f, i in pairs:
        if f in used_fac or i in used_id:
            continue
        mapping[f] = identities[i]
        used_fac.add(f)
        used_id.add(i)
    return {"by_factor": mapping, "match_score": score.round(3)}


def orient_factors(
    loadings: pd.DataFrame, scores: pd.DataFrame, mapping: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flip each factor so its signature items load negatively → the POSITIVE
    score pole becomes 보수 (정치) / 신뢰 (효능감), since every substantive item
    recodes so HIGH=진보. Returns (oriented_loadings, oriented_scores, report)
    with columns renamed to the confirmed identity names.
    """
    L, S = loadings.copy(), scores.copy()
    rows, rename_L, rename_S = [], {}, {}
    score_cols = list(scores.columns)
    for ci, fac in enumerate(loadings.columns):
        idn = mapping[fac]
        sig = [s for s in idn["signature"] if s in L.index]
        mean_sig = float(L.loc[sig, fac].mean()) if sig else 0.0
        flip = mean_sig > 0
        if flip:
            L[fac] = -L[fac]
            S[score_cols[ci]] = -S[score_cols[ci]]
        rows.append(
            {
                "empirical_factor": fac,
                "name": idn["name"],
                "political": idn["political"],
                "positive_label": idn["positive_label"],
                "mean_signature_loading_raw": round(mean_sig, 3),
                "flipped": flip,
            }
        )
        rename_L[fac] = idn["name"]
        rename_S[score_cols[ci]] = idn["name"] + "_score"
    L = L.rename(columns=rename_L)
    S = S.rename(columns=rename_S)
    return L, S, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 8 — factor scores
# ---------------------------------------------------------------------------
def factor_scores(fa, df_z: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    scores = fa.transform(df_z)
    cols = [f"F{i + 1}_score" for i in range(n_factors)]
    return pd.DataFrame(scores, index=df_z.index, columns=cols)


def score_distribution_report(scores: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean": scores.mean().round(3),
            "std": scores.std().round(3),
            "skew": scores.skew().round(3),
            "kurtosis": scores.kurtosis().round(3),
        }
    )
