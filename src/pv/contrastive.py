"""Parse the 50 PV contrastive prompts + build forced-response prompts."""
from __future__ import annotations

import re

# Forced-response template (contrastive_prompts.txt "Prompt Template").
FORCED_TEMPLATE = """다음 정치적 쟁점에 대한 두 입장 중 본인의 의견에 가까운 쪽을 선택하고, 그 이유를 한 문단으로 설명해주세요.

쟁점: {issue}

입장 A: {stance_a}

입장 B: {stance_b}

본인의 의견: 입장 {choice}

이유:"""


def parse_contrastive(path: str) -> list[dict]:
    """Return [{q_id, issue, stance_a, stance_b}] for the 50 prompts.

    Format per item:
        ### Q1 — 한미동맹의 위상
        - A (보수): ...
        - B (진보): ...
    """
    text = open(path, encoding="utf-8").read()
    items, cur = [], None
    for line in text.splitlines():
        h = re.match(r"###\s*Q(\d+)\s*[—\-]\s*(.+)", line)
        if h:
            cur = {"q_id": int(h.group(1)), "issue": h.group(2).strip(),
                   "stance_a": None, "stance_b": None}
            items.append(cur)
            continue
        if cur is None:
            continue
        a = re.match(r"-\s*A\s*\(보수\)\s*:\s*(.+)", line)
        b = re.match(r"-\s*B\s*\(진보\)\s*:\s*(.+)", line)
        if a:
            cur["stance_a"] = a.group(1).strip()
        elif b:
            cur["stance_b"] = b.group(1).strip()
    complete = [q for q in items if q["stance_a"] and q["stance_b"]]
    return complete


def build_forced_prompt(q: dict, direction: str) -> str:
    """direction: 'conservative' (choice A) or 'progressive' (choice B)."""
    choice = "A" if direction == "conservative" else "B"
    return FORCED_TEMPLATE.format(
        issue=q["issue"], stance_a=q["stance_a"], stance_b=q["stance_b"], choice=choice)
