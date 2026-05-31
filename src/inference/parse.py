"""Parse a model response into a structured vote (evaluation_pipeline.md D3, lite)."""
from __future__ import annotations

import json
import re

CANDIDATES = ["이재명", "김문수", "이준석"]
# 3-way orientation → the SET of same-side candidates (diagnostic only; consistency
# is NOT the success metric — see memory consistency-not-the-target). Both 김문수 and
# 이준석 are conservative-side, so a 보수 agent voting either is orientation-consistent.
ALIGNED = {"보수": {"김문수", "이준석"}, "진보": {"이재명"}}  # 중도: no aligned set
PARTY_TO_CAND = {"더불어민주당": "이재명", "국민의힘": "김문수", "개혁신당": "이준석"}
# Anonymized-ballot mapping (v3 setup: model sees "1번/2번/4번 후보" instead of names).
# Numbers match the original NEC ballot order (3번 absent — V1 quirk preserved).
NUMBER_TO_CAND = {"1": "이재명", "2": "김문수", "4": "이준석"}


def parse_vote(text: str) -> dict:
    """Returns {ok, will_vote, candidate, raw}. Handles a leading <thinking>…</thinking>
    block (manual CoT) by parsing the JSON after it; takes the last valid flat JSON object."""
    rec = {"ok": False, "will_vote": None, "candidate": "", "raw": text[:200]}
    body = text.split("</thinking>")[-1] if "</thinking>" in text else text
    objs = re.findall(r"\{[^{}]*\}", body, re.DOTALL)
    j = None
    for o in reversed(objs):              # last parseable flat object
        try:
            j = json.loads(o)
            break
        except Exception:
            continue
    if j is None:
        return rec
    rec["ok"] = True
    rec["will_vote"] = bool(j.get("will_vote", False))
    rec["candidate"] = _match_candidate(str(j.get("candidate", "")).strip())
    return rec


def _match_candidate(cand: str) -> str:
    if not cand:
        return ""
    for c in CANDIDATES:
        if c in cand:
            return c
    for party, c in PARTY_TO_CAND.items():
        if party in cand:
            return c
    # Anonymized ballot: "1번 후보" / "2번" / "후보 4" / bare "4" etc.
    m = re.search(r"(?<!\d)([124])\s*번", cand)
    if m:
        return NUMBER_TO_CAND[m.group(1)]
    m = re.fullmatch(r"\s*([124])\s*", cand)
    if m:
        return NUMBER_TO_CAND[m.group(1)]
    return ""  # unmatched (e.g. 권영국 or junk) → treated as no clear vote
