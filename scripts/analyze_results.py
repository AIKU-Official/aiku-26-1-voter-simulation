#!/usr/bin/env python
"""Main K=1 experiment analysis — generates Markdown summary + per-table CSVs.

Run from package root:
    HF_HOME=... uv run python scripts/analyze_results.py

Outputs under results/analysis/:
    summary.md                — comprehensive human-readable report
    nation_by_cell.csv        — Nation-level distribution + MAE
    orientation_by_cell.csv   — Orientation × cell vote counts + shares
    sido_by_cell.csv          — Sido × cell shares + per-cell MAE vs NEC
    age_by_cell.csv           — Age × cell shares + per-cell MAE vs KEP
    sex_by_cell.csv           — Sex × cell shares + per-cell MAE vs KEP
    marginal_contributions.csv — Ablation: BNS / PV / BNS+PV deltas
    case_studies/{agent_id}.md — Per-agent thinking traces across 5 cells
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import resolve  # noqa: E402

CELLS = ["C_L3", "C_base", "C_bns", "C_pv", "C_full"]
ORIENTS = ["VP", "P", "M", "C", "VC"]
CANDS = ["이재명", "김문수", "이준석"]

SIDO_MAP = {
    "강원특별자치도": "강원", "경기도": "경기", "경상남도": "경상남",
    "경상북도": "경상북", "광주광역시": "광주", "대구광역시": "대구",
    "대전광역시": "대전", "부산광역시": "부산", "서울특별시": "서울",
    "세종특별자치시": "세종", "울산광역시": "울산", "인천광역시": "인천",
    "전라남도": "전라남", "전북특별자치도": "전북", "제주특별자치도": "제주",
    "충청남도": "충청남", "충청북도": "충청북",
}
AGE_MAP = {"19-29": "18-29", "30-39": "30-39", "40-49": "40-49",
           "50-59": "50-59", "60-69": "60-69", "70+": "70+"}

CASE_AGENT_IDS = [1805, 4205]


def load_cell(cell: str) -> list[dict]:
    p = resolve(f"results/raw_outputs/{cell}/outputs.jsonl")
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def attach(cell: str, agents: pd.DataFrame) -> pd.DataFrame:
    rows = load_cell(cell)
    for r in rows:
        a = agents.loc[r["agent_id"]]
        r["sido17"] = a.sido17
        r["age_bucket"] = a.age_bucket
        r["sex_label"] = a.sex_label
    df = pd.DataFrame(rows)
    return df


def mae3(p: list[float], n: list[float]) -> float:
    return sum(abs(p[i] - n[i]) for i in range(3)) / 3


def main():
    out_root = resolve("results/analysis")
    out_root.mkdir(parents=True, exist_ok=True)
    case_dir = out_root / "case_studies"
    case_dir.mkdir(exist_ok=True)

    agents = pd.read_parquet(resolve("data/agents/final_agents.parquet")).set_index("agent_id")
    nec = pd.read_csv(resolve("data/ground_truth/nec_official.csv"))
    kep = pd.read_csv(resolve("data/ground_truth/kep_exit_poll.csv"))

    nec_national = nec[nec.region_level == "national"].set_index("candidate_name")["vote_share_pct"]
    nec_national_vec = [float(nec_national.get(c, 0)) for c in CANDS]

    nec_sido = nec[nec.region_level == "sido"].copy()
    nec_sido["sido_short"] = nec_sido.region_sido.map(SIDO_MAP)
    nec_sido_pivot = nec_sido.pivot(index="sido_short", columns="candidate_name",
                                    values="vote_share_pct").fillna(0)
    kep_age = kep[kep.breakdown_type == "age"].pivot(
        index="breakdown_value", columns="candidate_name", values="share_pct").fillna(0)
    kep_sex = kep[kep.breakdown_type == "sex"].pivot(
        index="breakdown_value", columns="candidate_name", values="share_pct").fillna(0)

    # Voting-only rows attached with subgroup labels, keyed by cell.
    voted = {c: attach(c, agents).query("will_vote == True and candidate != ''")
             for c in CELLS}
    # All rows (for abstain/non-voted counts), keyed by cell.
    all_rows = {c: attach(c, agents) for c in CELLS}

    md = ["# Main Experiment (K=1) — Analysis Report",
          "",
          "**Setup**: 5,000 agents (natural orientation distribution 26/48/26), "
          "5 cells × K=1, named candidates + policy text, soft mitigation prompts.",
          "**Output**: `results/raw_outputs/{cell}/outputs.jsonl`",
          "",
          "## 1. Nation-level distribution",
          ""]

    # --- 1. Nation table ---
    nation_rows = []
    for c in CELLS:
        rows = all_rows[c]
        n = len(rows)
        voted_rows = voted[c]
        cn = voted_rows["candidate"].value_counts()
        abstain = (~rows["will_vote"].astype(bool)).sum()
        unmatched = (rows["will_vote"].astype(bool) & (rows["candidate"] == "")).sum()
        p = [cn.get(cand, 0) / n * 100 for cand in CANDS]
        m = mae3(p, nec_national_vec)
        nation_rows.append({
            "cell": c, "n_total": n,
            "이재명_pct": round(p[0], 2), "김문수_pct": round(p[1], 2),
            "이준석_pct": round(p[2], 2),
            "abstain_pct": round(abstain / n * 100, 2),
            "unmatched_pct": round(unmatched / n * 100, 2),
            "mae_vs_nec_pp": round(m, 2),
        })
    nec_row = {
        "cell": "NEC_actual", "n_total": "",
        "이재명_pct": round(nec_national_vec[0], 2),
        "김문수_pct": round(nec_national_vec[1], 2),
        "이준석_pct": round(nec_national_vec[2], 2),
        "abstain_pct": "", "unmatched_pct": "", "mae_vs_nec_pp": 0.0,
    }
    nation_df = pd.DataFrame(nation_rows + [nec_row])
    nation_df.to_csv(out_root / "nation_by_cell.csv", index=False, encoding="utf-8-sig")

    md += ["| Cell | n | 이재명 % | 김문수 % | 이준석 % | abstain % | MAE vs NEC |",
           "|---|---|---|---|---|---|---|"]
    for r in nation_rows:
        md.append(f"| **{r['cell']}** | {r['n_total']} | {r['이재명_pct']:.1f} | "
                  f"{r['김문수_pct']:.1f} | {r['이준석_pct']:.1f} | {r['abstain_pct']:.1f} | "
                  f"**{r['mae_vs_nec_pp']:.2f}pp** |")
    md.append(f"| _NEC actual_ | _-_ | _{nec_national_vec[0]:.1f}_ | "
              f"_{nec_national_vec[1]:.1f}_ | _{nec_national_vec[2]:.1f}_ | _-_ | _0.00pp_ |")
    md.append("")

    # --- 2. Orientation breakdown ---
    md += ["## 2. Orientation × cell — counts and shares",
           "",
           "Agents per orientation (natural distribution): "
           f"VP={sum(agents.orientation == 'VP')}, P={sum(agents.orientation == 'P')}, "
           f"M={sum(agents.orientation == 'M')}, C={sum(agents.orientation == 'C')}, "
           f"VC={sum(agents.orientation == 'VC')}.",
           ""]
    orient_records = []
    for c in CELLS:
        rows = all_rows[c]
        for o in ORIENTS:
            sub = rows[rows.orientation == o]
            n_total = len(sub)
            cn = Counter(sub[sub.will_vote.astype(bool) & (sub.candidate != "")].candidate)
            abstain = (~sub.will_vote.astype(bool)).sum()
            orient_records.append({
                "cell": c, "orient": o, "n_total": n_total,
                "이재명_n": cn.get("이재명", 0), "김문수_n": cn.get("김문수", 0),
                "이준석_n": cn.get("이준석", 0), "abstain_n": int(abstain),
                "이재명_pct": round(cn.get("이재명", 0) / n_total * 100, 2),
                "김문수_pct": round(cn.get("김문수", 0) / n_total * 100, 2),
                "이준석_pct": round(cn.get("이준석", 0) / n_total * 100, 2),
                "abstain_pct": round(abstain / n_total * 100, 2),
            })
    pd.DataFrame(orient_records).to_csv(
        out_root / "orientation_by_cell.csv", index=False, encoding="utf-8-sig")

    # --- 2b. Orientation × age × cell ---
    # Long-format: one row per (orient, age, cell). Filter-friendly for Excel.
    orient_age_records = []
    for c in CELLS:
        rows = all_rows[c].copy()
        rows["age_kep"] = rows.age_bucket.map(AGE_MAP)
        for o in ORIENTS:
            for ag in sorted(rows["age_kep"].dropna().unique()):
                sub = rows[(rows.orientation == o) & (rows.age_kep == ag)]
                n_total = len(sub)
                if n_total == 0:
                    continue
                voted_sub = sub[sub.will_vote.astype(bool) & (sub.candidate != "")]
                cn = Counter(voted_sub.candidate)
                abstain = int((~sub.will_vote.astype(bool)).sum())
                orient_age_records.append({
                    "orient": o, "age": ag, "cell": c, "n_total": n_total,
                    "이재명_n": cn.get("이재명", 0), "김문수_n": cn.get("김문수", 0),
                    "이준석_n": cn.get("이준석", 0), "abstain_n": abstain,
                    "이재명_pct": round(cn.get("이재명", 0) / n_total * 100, 2),
                    "김문수_pct": round(cn.get("김문수", 0) / n_total * 100, 2),
                    "이준석_pct": round(cn.get("이준석", 0) / n_total * 100, 2),
                    "abstain_pct": round(abstain / n_total * 100, 2),
                })
    # Order: orient (VP→VC), then age, then cell (C_L3→C_full)
    orient_age_records.sort(key=lambda r: (ORIENTS.index(r["orient"]),
                                            r["age"],
                                            CELLS.index(r["cell"])))
    pd.DataFrame(orient_age_records).to_csv(
        out_root / "orientation_age_by_cell.csv", index=False, encoding="utf-8-sig")

    # --- 2c. Orientation × age × cell (wide format pivot for quick scanning) ---
    # Row = (orient, age); columns = cell × {이재명_pct, 김문수_pct, 이준석_pct}.
    wide_rows = []
    by_oa = {}
    for r in orient_age_records:
        by_oa.setdefault((r["orient"], r["age"]), {})[r["cell"]] = r
    for (o, ag), per_cell in sorted(by_oa.items(),
                                    key=lambda kv: (ORIENTS.index(kv[0][0]), kv[0][1])):
        n_total = per_cell[CELLS[0]]["n_total"] if CELLS[0] in per_cell else 0
        row = {"orient": o, "age": ag, "n_total": n_total}
        for c in CELLS:
            if c in per_cell:
                rec = per_cell[c]
                row[f"{c}_이재명_pct"] = rec["이재명_pct"]
                row[f"{c}_김문수_pct"] = rec["김문수_pct"]
                row[f"{c}_이준석_pct"] = rec["이준석_pct"]
        wide_rows.append(row)
    pd.DataFrame(wide_rows).to_csv(
        out_root / "orientation_age_by_cell_wide.csv", index=False, encoding="utf-8-sig")

    for c in CELLS:
        md.append(f"### {c}")
        md.append("")
        md.append("| Orient | n | 이재명 | 김문수 | 이준석 | 미투표 |")
        md.append("|---|---|---|---|---|---|")
        for o in ORIENTS:
            rec = next(r for r in orient_records if r["cell"] == c and r["orient"] == o)
            md.append(f"| {o} | {rec['n_total']} | "
                      f"{rec['이재명_n']} ({rec['이재명_pct']:.1f}%) | "
                      f"{rec['김문수_n']} ({rec['김문수_pct']:.1f}%) | "
                      f"{rec['이준석_n']} ({rec['이준석_pct']:.1f}%) | "
                      f"{rec['abstain_n']} ({rec['abstain_pct']:.1f}%) |")
        md.append("")

    # --- 3. Sido × cell with MAE ---
    md += ["## 3. Sido (17 region) × cell — shares + MAE vs NEC", ""]
    sido_records = []
    agent_sido_counts = agents.groupby("sido17").size()
    cell_sido = {c: voted[c].groupby(["sido17", "candidate"]).size().unstack(fill_value=0)
                 for c in CELLS}
    sido_avg_mae = {c: [] for c in CELLS}
    for sido in sorted(agent_sido_counts.index):
        n = int(agent_sido_counts[sido])
        nec_vals = ([float(nec_sido_pivot.loc[sido].get(cand, 0)) for cand in CANDS]
                    if sido in nec_sido_pivot.index else None)
        rec = {"sido": sido, "n": n}
        if nec_vals:
            rec["nec_이재명"] = round(nec_vals[0], 2)
            rec["nec_김문수"] = round(nec_vals[1], 2)
            rec["nec_이준석"] = round(nec_vals[2], 2)
        for c in CELLS:
            cd = cell_sido[c]
            if sido in cd.index:
                row = cd.loc[sido]; tot = row.sum()
                p = [row.get(cand, 0) / tot * 100 if tot > 0 else 0 for cand in CANDS]
                rec[f"{c}_이재명"] = round(p[0], 2)
                rec[f"{c}_김문수"] = round(p[1], 2)
                rec[f"{c}_이준석"] = round(p[2], 2)
                if nec_vals:
                    m = mae3(p, nec_vals)
                    rec[f"{c}_mae"] = round(m, 2)
                    sido_avg_mae[c].append(m)
        sido_records.append(rec)
    pd.DataFrame(sido_records).to_csv(
        out_root / "sido_by_cell.csv", index=False, encoding="utf-8-sig")

    md.append("| Sido | n | C_L3 share / MAE | C_base share / MAE | C_bns share / MAE | "
              "C_pv share / MAE | C_full share / MAE | NEC actual |")
    md.append("|---|---|---|---|---|---|---|---|")
    for rec in sido_records:
        cells_str = []
        for c in CELLS:
            if f"{c}_이재명" in rec:
                cells_str.append(
                    f"{rec[f'{c}_이재명']:.1f}/{rec[f'{c}_김문수']:.1f}/"
                    f"{rec[f'{c}_이준석']:.1f} / **{rec.get(f'{c}_mae', 0):.2f}**")
            else:
                cells_str.append("—")
        nec_str = (f"{rec['nec_이재명']:.1f}/{rec['nec_김문수']:.1f}/{rec['nec_이준석']:.1f}"
                   if "nec_이재명" in rec else "—")
        md.append(f"| {rec['sido']} | {rec['n']} | {cells_str[0]} | {cells_str[1]} | "
                  f"{cells_str[2]} | {cells_str[3]} | {cells_str[4]} | {nec_str} |")
    md.append("| **avg MAE** | | **"
              + "** | **".join(f"{sum(sido_avg_mae[c])/len(sido_avg_mae[c]):.2f}**"
                               for c in CELLS) + " | — |")
    md.append("")

    # --- 4. Age × cell with MAE ---
    md += ["## 4. Age × cell — shares + MAE vs KEP exit poll", ""]
    age_records = []
    age_counts = agents.age_bucket.map(AGE_MAP).value_counts()
    cell_age = {}
    for c in CELLS:
        d = voted[c].copy()
        d["age_kep"] = d.age_bucket.map(AGE_MAP)
        cell_age[c] = d.groupby(["age_kep", "candidate"]).size().unstack(fill_value=0)
    age_avg_mae = {c: [] for c in CELLS}
    for ag in sorted(age_counts.index):
        n = int(age_counts[ag])
        kep_vals = ([float(kep_age.loc[ag].get(cand, 0)) for cand in CANDS]
                    if ag in kep_age.index else None)
        rec = {"age": ag, "n": n}
        if kep_vals:
            rec["kep_이재명"] = round(kep_vals[0], 2)
            rec["kep_김문수"] = round(kep_vals[1], 2)
            rec["kep_이준석"] = round(kep_vals[2], 2)
        for c in CELLS:
            cd = cell_age[c]
            if ag in cd.index:
                row = cd.loc[ag]; tot = row.sum()
                p = [row.get(cand, 0) / tot * 100 if tot > 0 else 0 for cand in CANDS]
                rec[f"{c}_이재명"] = round(p[0], 2)
                rec[f"{c}_김문수"] = round(p[1], 2)
                rec[f"{c}_이준석"] = round(p[2], 2)
                if kep_vals:
                    m = mae3(p, kep_vals)
                    rec[f"{c}_mae"] = round(m, 2)
                    age_avg_mae[c].append(m)
        age_records.append(rec)
    pd.DataFrame(age_records).to_csv(
        out_root / "age_by_cell.csv", index=False, encoding="utf-8-sig")

    md.append("| Age | n | C_L3 share / MAE | C_base share / MAE | C_bns share / MAE | "
              "C_pv share / MAE | C_full share / MAE | KEP actual |")
    md.append("|---|---|---|---|---|---|---|---|")
    for rec in age_records:
        cs = [f"{rec[f'{c}_이재명']:.1f}/{rec[f'{c}_김문수']:.1f}/{rec[f'{c}_이준석']:.1f}"
              f" / **{rec.get(f'{c}_mae', 0):.2f}**" if f"{c}_이재명" in rec else "—"
              for c in CELLS]
        kep_str = (f"{rec['kep_이재명']:.1f}/{rec['kep_김문수']:.1f}/{rec['kep_이준석']:.1f}"
                   if "kep_이재명" in rec else "—")
        md.append(f"| {rec['age']} | {rec['n']} | "
                  + " | ".join(cs) + f" | {kep_str} |")
    md.append("| **avg MAE** | | **"
              + "** | **".join(f"{sum(age_avg_mae[c])/len(age_avg_mae[c]):.2f}**"
                               for c in CELLS) + " | — |")
    md.append("")

    # --- 5. Sex × cell with MAE ---
    md += ["## 5. Sex × cell — shares + MAE vs KEP exit poll", ""]
    sex_records = []
    sex_counts = agents.sex_label.value_counts()
    cell_sex = {c: voted[c].groupby(["sex_label", "candidate"]).size().unstack(fill_value=0)
                for c in CELLS}
    sex_avg_mae = {c: [] for c in CELLS}
    for sx in sex_counts.index:
        n = int(sex_counts[sx])
        kep_vals = ([float(kep_sex.loc[sx].get(cand, 0)) for cand in CANDS]
                    if sx in kep_sex.index else None)
        rec = {"sex": sx, "n": n}
        if kep_vals:
            rec["kep_이재명"] = round(kep_vals[0], 2)
            rec["kep_김문수"] = round(kep_vals[1], 2)
            rec["kep_이준석"] = round(kep_vals[2], 2)
        for c in CELLS:
            cd = cell_sex[c]
            if sx in cd.index:
                row = cd.loc[sx]; tot = row.sum()
                p = [row.get(cand, 0) / tot * 100 if tot > 0 else 0 for cand in CANDS]
                rec[f"{c}_이재명"] = round(p[0], 2)
                rec[f"{c}_김문수"] = round(p[1], 2)
                rec[f"{c}_이준석"] = round(p[2], 2)
                if kep_vals:
                    m = mae3(p, kep_vals)
                    rec[f"{c}_mae"] = round(m, 2)
                    sex_avg_mae[c].append(m)
        sex_records.append(rec)
    pd.DataFrame(sex_records).to_csv(
        out_root / "sex_by_cell.csv", index=False, encoding="utf-8-sig")

    md.append("| Sex | n | C_L3 / MAE | C_base / MAE | C_bns / MAE | C_pv / MAE | C_full / MAE | KEP |")
    md.append("|---|---|---|---|---|---|---|---|")
    for rec in sex_records:
        cs = [f"{rec[f'{c}_이재명']:.1f}/{rec[f'{c}_김문수']:.1f}/{rec[f'{c}_이준석']:.1f}"
              f" / **{rec.get(f'{c}_mae', 0):.2f}**" if f"{c}_이재명" in rec else "—"
              for c in CELLS]
        kep_str = (f"{rec['kep_이재명']:.1f}/{rec['kep_김문수']:.1f}/{rec['kep_이준석']:.1f}"
                   if "kep_이재명" in rec else "—")
        md.append(f"| {rec['sex']} | {rec['n']} | "
                  + " | ".join(cs) + f" | {kep_str} |")
    md.append("| **avg MAE** | | **"
              + "** | **".join(f"{sum(sex_avg_mae[c])/len(sex_avg_mae[c]):.2f}**"
                               for c in CELLS) + " | — |")
    md.append("")

    # --- 6. Marginal contributions ---
    md += ["## 6. Marginal contributions of each framework component",
           "",
           "Reduction in MAE relative to C_L3 (baseline) and to C_base "
           "(orientation-only baseline).",
           ""]
    bases = {"Nation": (nation_rows[0]["mae_vs_nec_pp"], nation_rows[1]["mae_vs_nec_pp"]),
             "Sido": (sum(sido_avg_mae["C_L3"]) / len(sido_avg_mae["C_L3"]),
                      sum(sido_avg_mae["C_base"]) / len(sido_avg_mae["C_base"])),
             "Age": (sum(age_avg_mae["C_L3"]) / len(age_avg_mae["C_L3"]),
                     sum(age_avg_mae["C_base"]) / len(age_avg_mae["C_base"])),
             "Sex": (sum(sex_avg_mae["C_L3"]) / len(sex_avg_mae["C_L3"]),
                     sum(sex_avg_mae["C_base"]) / len(sex_avg_mae["C_base"]))}
    cell_maes = {
        "Nation": {r["cell"]: r["mae_vs_nec_pp"] for r in nation_rows},
        "Sido": {c: sum(sido_avg_mae[c]) / len(sido_avg_mae[c]) for c in CELLS},
        "Age": {c: sum(age_avg_mae[c]) / len(age_avg_mae[c]) for c in CELLS},
        "Sex": {c: sum(sex_avg_mae[c]) / len(sex_avg_mae[c]) for c in CELLS},
    }
    marg_records = []
    for metric, (l3, base) in bases.items():
        rec = {
            "metric": metric, "C_L3_mae": round(l3, 2), "C_base_mae": round(base, 2),
            "C_bns_mae": round(cell_maes[metric]["C_bns"], 2),
            "C_pv_mae": round(cell_maes[metric]["C_pv"], 2),
            "C_full_mae": round(cell_maes[metric]["C_full"], 2),
            "orient_contrib_pp": round(l3 - base, 2),
            "bns_contrib_pp": round(base - cell_maes[metric]["C_bns"], 2),
            "pv_contrib_pp": round(base - cell_maes[metric]["C_pv"], 2),
            "bns_pv_contrib_pp": round(base - cell_maes[metric]["C_full"], 2),
        }
        marg_records.append(rec)
    pd.DataFrame(marg_records).to_csv(
        out_root / "marginal_contributions.csv", index=False, encoding="utf-8-sig")

    md.append("| Metric | C_L3 | C_base | C_bns | C_pv | C_full | Orient (C_L3→C_base) | "
              "BNS (C_base→C_bns) | PV (C_base→C_pv) | BNS+PV (C_base→C_full) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in marg_records:
        md.append(f"| {r['metric']} | {r['C_L3_mae']:.2f} | {r['C_base_mae']:.2f} | "
                  f"{r['C_bns_mae']:.2f} | {r['C_pv_mae']:.2f} | {r['C_full_mae']:.2f} | "
                  f"**-{r['orient_contrib_pp']:.2f}** | **-{r['bns_contrib_pp']:.2f}** | "
                  f"-{r['pv_contrib_pp']:.2f} | **-{r['bns_pv_contrib_pp']:.2f}** |")
    md.append("")

    # --- 7. Case studies ---
    md += ["## 7. Case studies",
           "",
           f"See `case_studies/{{agent_id}}.md` for full reasoning traces "
           f"({', '.join(str(a) for a in CASE_AGENT_IDS)}).",
           ""]
    for aid in CASE_AGENT_IDS:
        a = agents.loc[aid]
        lines = [f"# Agent {aid} — full reasoning trace across 5 cells",
                 "",
                 "## Persona",
                 f"- **Demographics**: {a.age_bucket} {a.sex_label}, "
                 f"{a.sido17}, 학력={a.edu4}, 직업={a.occupation}",
                 f"- **Orientation label**: {a.orientation}",
                 "- **KGSS factor scores**:",
                 f"    - 경제·재분배(정부책임): {a['경제·재분배(정부책임)_score']:+.3f}",
                 f"    - 대북·안보: {a['대북·안보_score']:+.3f}",
                 f"    - 시장·민영화: {a['시장·민영화_score']:+.3f}",
                 f"    - 정치효능감(신뢰/냉소): {a['정치효능감(신뢰/냉소)_score']:+.3f}",
                 f"- **BNS seed_1**: {a.bns_seed_1}"]
        if a.bns_seed_2:
            lines.append(f"- **BNS seed_2**: {a.bns_seed_2}")
        lines += ["", "## Per-cell outputs"]
        for cell in CELLS:
            for d in load_cell(cell):
                if d["agent_id"] == aid:
                    txt = d.get("text") or d.get("raw") or ""
                    lines.append(f"### {cell} → **{d['candidate']}**")
                    lines.append("")
                    if "<thinking>" in txt:
                        th = txt.split("<thinking>")[1].split("</thinking>")[0].strip()
                        after = txt.split("</thinking>")[1].strip()
                        lines.append("**thinking**:")
                        lines.append("")
                        lines.append("> " + th.replace("\n", "\n> "))
                        if after:
                            lines.append("")
                            lines.append("**output**:")
                            lines.append("")
                            lines.append("```json")
                            lines.append(after)
                            lines.append("```")
                    else:
                        lines.append("```json")
                        lines.append(txt)
                        lines.append("```")
                    lines.append("")
                    break
        (case_dir / f"agent_{aid}.md").write_text("\n".join(lines), encoding="utf-8")

    md += ["",
           "## 8. Files generated",
           "",
           "- `summary.md` (this file)",
           "- `nation_by_cell.csv`",
           "- `orientation_by_cell.csv`",
           "- `orientation_age_by_cell.csv` (long format)",
           "- `orientation_age_by_cell_wide.csv` (wide pivot)",
           "- `sido_by_cell.csv`",
           "- `age_by_cell.csv`",
           "- `sex_by_cell.csv`",
           "- `marginal_contributions.csv`",
           f"- `case_studies/agent_{{aid}}.md` for aid in {CASE_AGENT_IDS}",
           ""]

    (out_root / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote: {out_root}/summary.md")
    print(f"       {out_root}/*.csv (6 tables)")
    print(f"       {case_dir}/agent_*.md ({len(CASE_AGENT_IDS)} agents)")


if __name__ == "__main__":
    main()
