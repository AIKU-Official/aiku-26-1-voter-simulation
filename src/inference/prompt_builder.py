"""Build per-cell system/user prompts (prompts/system_prompts/cell_templates/all_cells.md).

For the α sweep we use the C_L3 base (narrative + candidates, no orientation text)
so PV injection is the only steering signal.
"""
from __future__ import annotations

import re

import pandas as pd

# Candidate anonymization (3-candidate ballot) is selectable per run, since the
# model's real-name and party priors are very strong (esp. 이재명 / 더불어민주당)
# and override persona conditioning. The caller picks a `candidate_mode`:
#   - "named"       → "1번 이재명 (더불어민주당 · 진보)"          (v1 baseline)
#   - "name_anon"   → "1번 후보 (더불어민주당 · 진보)"            (default; v3/v4)
#   - "party_anon"  → "1번 후보 (진보)"                           (strip party name)
#   - "full_anon"   → "1번 후보"                                  (strip all cues)
# load_candidate_block applies the same transform whether the policy block is
# included or not, so reasoning patterns stay comparable across --no-policy.
CANDIDATE_MODES = ("named", "name_anon", "party_anon", "full_anon")
_CANDIDATE_INFO = {
    "1": ("이재명", "더불어민주당", "진보"),
    "2": ("김문수", "국민의힘", "보수"),
    "4": ("이준석", "개혁신당", "청년 보수, 실용·효율 노선"),
}


def candidate_block_short(candidate_mode: str = "name_anon") -> str:
    """Name+party (or anonymized) only candidate block — no policy text.
    Used when include_policy=False."""
    if candidate_mode not in CANDIDATE_MODES:
        raise ValueError(f"unknown candidate_mode={candidate_mode!r}, "
                         f"expected one of {CANDIDATE_MODES}")
    lines = []
    for n in ("1", "2", "4"):
        name, party, ideology = _CANDIDATE_INFO[n]
        if candidate_mode == "named":
            lines.append(f"{n}번 {name} ({party} · {ideology})")
        elif candidate_mode == "name_anon":
            lines.append(f"{n}번 후보 ({party} · {ideology})")
        elif candidate_mode == "party_anon":
            lines.append(f"{n}번 후보 ({ideology})")
        else:  # full_anon
            lines.append(f"{n}번 후보")
    return "\n".join(lines)


# Back-compat alias: legacy callers (e.g. older alpha_sweep code paths) imported
# CANDIDATE_BLOCK_NAMES expecting the v3/v4 default (name anonymized, party kept).
CANDIDATE_BLOCK_NAMES = candidate_block_short("name_anon")

ELECTION_HEADER = ("오늘은 2025년 6월 3일, 제21대 대한민국 대통령 선거 투표일입니다.\n"
                   "이번 선거는 윤석열 전 대통령 탄핵 이후 치러지는 조기 대선입니다.")

INSTRUCTIONS_L3 = """다음 두 질문에 답해주세요.
1. 투표하시겠습니까? (예 / 아니오)
2. 투표한다면 어느 후보를 선택하시겠습니까?

응답은 다음 JSON 형식으로만 작성하세요.

{"will_vote": true, "candidate": "1번 후보", "reasoning_brief": "한 문장 이유"}

- will_vote: 투표 의향이 있으면 true, 없으면 false.
- candidate: 위 후보 정보의 "1번 후보" / "2번 후보" / "4번 후보" 중 하나. 투표하지 않으면 빈 문자열 "".
- reasoning_brief: 본인이 그렇게 결정한 핵심 이유를 한 문장으로."""

VOTE_REINFORCE = "- 당신의 가치관과 정치 성향에 가장 부합하는 후보를 선택하세요."

_JOSA = {"는": "은", "가": "이", "를": "을", "의": "의", "와": "과", "에게": "에게",
         "도": "도", "만": "만", "께서": "은", "에게서": "에게서", "랑": "이랑"}


def convert_voice_to_2nd_person(narrative: str) -> str:
    """Generic 3rd→2nd person: any '<이름> 씨<josa>' → '당신<josa>'; 1st-person residuals."""
    def repl(m):
        josa = m.group(1) or ""
        return "당신" + _JOSA.get(josa, josa)

    out = re.sub(r"[가-힣]{2,4}\s*씨(께서|에게서|에게|와|과|는|가|를|의|도|만|랑)?", repl, narrative)
    for p, r in [(r"(^|\n)\s*나는\s", r"\1당신은 "), (r"(^|\n)\s*나의\s", r"\1당신의 "),
                 (r"(^|\n)\s*내가\s", r"\1당신이 ")]:
        out = re.sub(p, r, out)
    return out


def _demographic_intro(a) -> str:
    occ = getattr(a, "occupation", None)
    occ_txt = f", 직업은 {occ}" if isinstance(occ, str) and occ and occ != "nan" else ""
    return (f"당신은 {a.sido17}에 거주하는 {a.age_bucket} {a.sex_label} 유권자입니다. "
            f"최종학력은 {a.edu4}{occ_txt}입니다.")


BLOCK1 = ("당신은 대한민국의 유권자입니다.\n"
          "당신이 아래의 인물이라고 생각하고, 투표권 행사에 대한 질문에 답하세요.")
BLOCK5_GUIDE = """## 중요 지침
- 위 프로필의 정치성향을 일반적 경향으로 참고하되, 본인의 직업·생활환경·후보 자질을 종합 고려해 결정하세요.
- 진보 성향 유권자도 일부는 후보 자질이나 정책에 따라 보수 후보를 지지할 수 있으며, 그 반대도 가능합니다. 정치성향은 결정의 한 요소일 뿐입니다.
- 같은 진영(진보/보수) 안에서도 후보마다 정책과 강조점이 다르므로, 본인의 연령·직업·지역·생활환경에 따라 어느 후보가 더 적합한지 판단하세요.
- 중도 성향이면 진보(1번 후보)와 보수(2번·4번 후보) 셋 모두를 진영 균등하게 검토하고, "사회 안전망 vs 자유 시장" 같은 거시 프레임이 아닌 본인의 직업·소득·지역 현실에 직결되는 정책을 우선 보세요. 진영 default가 아니라 본인 생활환경 기준으로 결정하세요.
- AI의 관점이 아닌, 이 프로필을 가진 실제 한국 유권자의 관점에서 판단하세요."""
BLOCK6 = "당신의 입장에서 진솔하게 답해주세요. 응답은 반드시 지정된 JSON 형식으로만 작성하세요."

_ORIENT_HEADER = {"강한 진보 (VP)": "VP", "진보 (P)": "P", "중도 (M)": "M",
                  "보수 (C)": "C", "강한 보수 (VC)": "VC"}


def _fenced(block_text: str) -> str:
    m = re.search(r"```\s*\n(.*?)```", block_text, re.DOTALL)
    return m.group(1).strip() if m else block_text.strip()


def load_orientation_descriptions(path: str) -> dict:
    """{VP/P/M/C/VC: description text} from orientation_descriptions.txt."""
    text = open(path, encoding="utf-8").read()
    out = {}
    for hdr, key in _ORIENT_HEADER.items():
        m = re.search(rf"###\s*{re.escape(hdr)}\s*\n(.*?)(?=\n###|\Z)", text, re.DOTALL)
        if m:
            out[key] = _fenced(m.group(1))
    return out


def _apply_anonymization(block: str, candidate_mode: str) -> str:
    """Anonymize a policy-rich candidate block according to candidate_mode.
    Source file is the named version ("## 1번 이재명 (더불어민주당 · 진보)"
    + "정당: 더불어민주당 (진보)" line); we rewrite headers + 정당 line as
    needed so the same source can be replayed at any anonymization level."""
    if candidate_mode == "named":
        return block
    for n, (name, party, ideology) in _CANDIDATE_INFO.items():
        if candidate_mode == "name_anon":
            block = re.sub(rf"##\s*{n}번\s+{re.escape(name)}\b",
                           f"## {n}번 후보", block)
            # 정당 line kept as-is (party + ideology preserved)
        elif candidate_mode == "party_anon":
            # Header: drop party token, keep ideology
            block = re.sub(rf"##\s*{n}번\s+{re.escape(name)}\s*\([^)]*\)",
                           f"## {n}번 후보 ({ideology})", block)
            # 정당 line: drop party name, keep ideology
            block = re.sub(rf"정당:\s*{re.escape(party)}\s*\(([^)]*)\)",
                           r"정당: 미공개 (\1)", block)
        else:  # full_anon
            block = re.sub(rf"##\s*{n}번\s+{re.escape(name)}\s*\([^)]*\)",
                           f"## {n}번 후보", block)
            block = re.sub(rf"정당:\s*{re.escape(party)}\s*\([^)]*\)\n?",
                           "", block)
    return block


def load_candidate_block(path: str, include_policy: bool,
                         candidate_mode: str = "name_anon") -> str:
    """Return the 3-candidate block in the requested anonymization mode.
    If include_policy=False, returns just the name+ideology line per candidate.
    If include_policy=True, returns the full policy section, with names/party
    rewritten in place per candidate_mode."""
    if candidate_mode not in CANDIDATE_MODES:
        raise ValueError(f"unknown candidate_mode={candidate_mode!r}, "
                         f"expected one of {CANDIDATE_MODES}")
    if not include_policy:
        return candidate_block_short(candidate_mode)
    text = open(path, encoding="utf-8").read()
    # Header may say "1번 이재명" (source-of-truth named version).
    m = re.search(r"(##\s*1번\s+\S+.*?)(?=\n##\s*Symmetric|\Z)", text, re.DOTALL)
    block = m.group(1).strip() if m else candidate_block_short(candidate_mode)
    return _apply_anonymization(block, candidate_mode)


def load_instructions_main(path: str) -> str:
    """The thinking + mitigation #4 instruction block (## Instructions Text fence)."""
    text = open(path, encoding="utf-8").read()
    m = re.search(r"##\s*Instructions Text\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    return _fenced(m.group(1)) if m else text


def build_cell(agent, cell: str, res: dict, include_policy: bool = True) -> tuple[str, str]:
    """Full per-cell (system, user) prompt (all_cells.md). res must hold
    'orient' (dict), 'candidate' (str), 'instructions_main' (str)."""
    blocks = [BLOCK1, _demographic_intro(agent),
              convert_voice_to_2nd_person(str(agent.narrative))]
    if cell != "C_L3":
        blocks.append(f"## 정치성향 및 가치관\n{res['orient'][agent.orientation]}")
    if cell in ("C_bns", "C_full"):
        s2 = getattr(agent, "bns_seed_2", "") or ""
        seeds = str(agent.bns_seed_1) + (("\n" + str(s2)) if s2 else "")
        blocks.append(f"## 명확한 정치적 의견\n{seeds}")
    if cell != "C_L3":
        blocks.append(BLOCK5_GUIDE)
    blocks.append(BLOCK6)
    system = "\n\n".join(blocks)

    # Always use res["candidate"] — the caller picks include_policy + candidate_mode
    # when constructing res via load_candidate_block(). The legacy fallback to
    # CANDIDATE_BLOCK_NAMES silently masked --candidate-mode in v5.
    cand = res["candidate"]
    instr = INSTRUCTIONS_L3 if cell == "C_L3" else res["instructions_main"]
    user = (f"{ELECTION_HEADER}\n\n후보자 정보:\n{cand}\n\n"
            f"당신은 {agent.sido17}에 거주하는 유권자입니다.\n\n{instr}")
    return system, user


def build_l3(agent, vote_instruction: bool = True) -> tuple[str, str]:
    """C_L3-style (system, user). `agent` is a namedtuple/Series with sido17,
    age_bucket, sex_label, edu4, occupation, narrative."""
    system = "\n\n".join([
        "당신은 대한민국의 유권자입니다.\n당신이 아래의 인물이라고 생각하고, 투표권 행사에 대한 질문에 답하세요.",
        _demographic_intro(agent),
        convert_voice_to_2nd_person(str(agent.narrative)),
        "당신의 입장에서 진솔하게 답해주세요. 응답은 반드시 지정된 JSON 형식으로만 작성하세요.",
    ])
    instr = INSTRUCTIONS_L3 + (("\n" + VOTE_REINFORCE) if vote_instruction else "")
    user = (f"{ELECTION_HEADER}\n\n후보자 정보:\n{CANDIDATE_BLOCK_NAMES}\n\n"
            f"당신은 {agent.sido17}에 거주하는 유권자입니다.\n\n{instr}")
    return system, user
