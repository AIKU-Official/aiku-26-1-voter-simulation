#!/usr/bin/env python
"""Generate full prompt + response transcripts for selected case study agents.

For each agent and cell, this script renders the actual SYSTEM and USER prompts
that were sent to the model (reconstructed via build_cell with the current
candidate_info / orientation_descriptions / instructions_main files) plus the
model's thinking and JSON output (loaded from results/raw_outputs/).

Run from package root:
    uv run python scripts/generate_case_studies.py

Outputs:
    results/analysis/case_studies_full/agent_<aid>.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.prompt_builder import (build_cell,  # noqa: E402
                                          load_candidate_block,
                                          load_instructions_main,
                                          load_orientation_descriptions)
from src.utils.config import resolve  # noqa: E402

CELLS = ["C_L3", "C_base", "C_bns", "C_pv", "C_full"]
CASE_AGENT_IDS = [1805, 4205]


def main():
    out_root = resolve("results/analysis/case_studies_full")
    out_root.mkdir(parents=True, exist_ok=True)

    agents = pd.read_parquet(resolve("data/agents/final_agents.parquet")).set_index("agent_id")
    # Build prompt resources for the main-experiment setup (named + policy + soft).
    res = {
        "orient": load_orientation_descriptions(
            str(resolve("prompts/system_prompts/orientation_descriptions.txt"))),
        "candidate": load_candidate_block(
            str(resolve("prompts/user_prompts/candidate_info.txt")),
            include_policy=True, candidate_mode="named"),
        "instructions_main": load_instructions_main(
            str(resolve("prompts/user_prompts/instructions_main.txt"))),
    }

    for aid in CASE_AGENT_IDS:
        a = agents.loc[aid]
        out_lines = [
            f"# Agent {aid} — full prompts + responses across 5 cells",
            "",
            "**Setup**: main experiment (`results/raw_outputs/`)",
            "- `candidate_mode=named` (1번 이재명 / 2번 김문수 / 4번 이준석)",
            "- WITH policy text",
            "- Soft mitigation prompts (defection clauses in orientation/BLOCK5/instructions)",
            "",
            "## Persona",
            "",
            f"- **agent_id**: {aid}",
            f"- **Demographics**: {a.age_bucket} {a.sex_label}, {a.sido17}, "
            f"학력={a.edu4}, 직업={a.occupation}",
            f"- **Orientation label**: {a.orientation}  (label assigned by the agent construction pipeline)",
            f"- **Narrative lean** (LLM judge): `{a.narrative_lean}` (confidence {a.confidence:.2f})",
            f"- **Resample level**: {a.resample_level}",
            "- **KGSS factor scores**:",
            f"    - 경제·재분배(정부책임): {a['경제·재분배(정부책임)_score']:+.3f}",
            f"    - 대북·안보: {a['대북·안보_score']:+.3f}",
            f"    - 시장·민영화: {a['시장·민영화_score']:+.3f}",
            f"    - 정치효능감(신뢰/냉소): {a['정치효능감(신뢰/냉소)_score']:+.3f}",
            f"- **BNS seed type**: {a.seed_type} "
            f"(primary={a.primary_factor}/{a.primary_dir}/{a.primary_strength}, "
            f"secondary={a.secondary_factor}/{a.secondary_dir})",
            f"- **BNS seed_1**: {a.bns_seed_1}",
        ]
        if a.bns_seed_2:
            out_lines.append(f"- **BNS seed_2**: {a.bns_seed_2}")
        out_lines += ["", "---", ""]

        for cell in CELLS:
            # Reconstruct prompts (this mirrors what phase4 sent at inference time).
            system, user = build_cell(a, cell, res, include_policy=True)

            # Look up actual response.
            response = None
            for d in (json.loads(l) for l in
                      open(resolve(f"results/raw_outputs/{cell}/outputs.jsonl"),
                           encoding="utf-8")):
                if d["agent_id"] == aid:
                    response = d
                    break

            chosen = response["candidate"] if response else "(missing)"
            out_lines += [f"## Cell `{cell}` → **{chosen}**", ""]

            out_lines += [
                f"- system prompt: {len(system):,} chars",
                f"- user prompt: {len(user):,} chars",
                "",
                "### SYSTEM prompt", "",
                "```text", system, "```", "",
                "### USER prompt", "",
                "```text", user, "```", "",
                "### Model response", "",
            ]

            if response is None:
                out_lines += ["_(no response found)_", "", "---", ""]
                continue

            txt = response.get("text") or response.get("raw") or ""
            if "<thinking>" in txt:
                th = txt.split("<thinking>")[1].split("</thinking>")[0].strip()
                after = txt.split("</thinking>")[1].strip()
                out_lines += ["**thinking**:", "", "```text", th, "```", ""]
                if after:
                    out_lines += ["**output JSON**:", "", "```json", after, "```", ""]
            else:
                out_lines += ["**raw output**:", "", "```json", txt, "```", ""]
            out_lines += [
                f"- parsed: `will_vote={response['will_vote']}`, "
                f"`candidate={response['candidate']!r}`",
                "", "---", "",
            ]

        path = out_root / f"agent_{aid}.md"
        path.write_text("\n".join(out_lines), encoding="utf-8")
        print(f"wrote: {path}  ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
