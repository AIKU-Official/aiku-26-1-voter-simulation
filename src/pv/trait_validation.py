"""EXAONE cross-model trait validation for the 50 contrastive prompts.

EXAONE scores each Qwen forced response on BOTH axes (보수성 1-5, 진보성 1-5).
A prompt PASSES (relative criterion, configs/config.yaml) if forced-conservative
responses out-score forced-progressive on the 보수 axis by >= sep_margin AND
forced-progressive out-score forced-conservative on the 진보 axis by >= sep_margin.
The absolute rubric (consistency/strength >= 3.5) is also reported.
"""
from __future__ import annotations

import json
import re

import pandas as pd

JUDGE_SYSTEM = """당신은 한국 정치 담론 분석 전문가입니다. 주어진 응답이 한국 정치의 보수 trait과 진보 trait을 각각 얼마나 강하게 표현하는지 두 축으로 독립 평가합니다.

평가 맥락: 응답은 아래 쟁점의 두 입장(A=보수, B=진보) 중 하나를 옹호하도록 작성된 것입니다.

## 보수성 (1-5): 응답이 한국 정치의 보수 입장(안보 우선·한미동맹·시장경제·전통·강경대북 등)을 표현하는 정도
- 5: 강한 보수 가치를 명확히 옹호   3: 중간/약함   1: 보수 표현 없음 또는 반대
## 진보성 (1-5): 응답이 한국 정치의 진보 입장(분배·복지·평화·인권·대화·자주외교 등)을 표현하는 정도
- 5: 강한 진보 가치를 명확히 옹호   3: 중간/약함   1: 진보 표현 없음 또는 반대

두 축은 독립적입니다(한 응답이 둘 다 낮거나 둘 다 중간일 수 있음). 응답은 다음 JSON만 출력하세요:
{"보수": <1-5>, "진보": <1-5>}"""


def _user_msg(issue, stance_a, stance_b, response) -> str:
    return (f"쟁점: {issue}\n입장 A(보수): {stance_a}\n입장 B(진보): {stance_b}\n\n"
            f"[응답]\n{response}")


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
        return {"보수": float(j["보수"]), "진보": float(j["진보"])}
    except Exception:
        return None


def judge_responses(responses: pd.DataFrame, qs_by_id: dict, model_id: str,
                    batch_size: int = 16, log=None, _loaded=None) -> pd.DataFrame:
    """EXAONE scores each response on (보수, 진보). `responses` cols: item, q_id,
    direction, rollout, text. Returns same + 보수/진보 score columns."""
    import torch

    from src.agents.narrative_judge import load_local

    tok, model = _loaded if _loaded else load_local(model_id)
    rows = responses.reset_index(drop=True)
    out: dict[int, dict] = {}
    for i in range(0, len(rows), batch_size):
        batch = rows.iloc[i:i + batch_size]
        prompts = []
        for _, r in batch.iterrows():
            q = qs_by_id[r["q_id"]]
            msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": _user_msg(
                        q["issue"], q["stance_a"], q["stance_b"], r["text"])}]
            try:
                prompts.append(tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                prompts.append(tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (_, r), t in zip(batch.iterrows(), texts):
            out[int(r["item"])] = _parse(t) or {"보수": None, "진보": None}
        if log:
            log.info("  judged %d/%d", min(i + batch_size, len(rows)), len(rows))
    sc = pd.DataFrame([{"item": k, **v} for k, v in out.items()])
    return rows.merge(sc, on="item")


def aggregate(judged: pd.DataFrame, sep_margin: float = 0.5,
              min_pass_per_set: int = 35) -> dict:
    """Per-prompt relative separation + set pass rates + cross-set asymmetry."""
    g = (judged.dropna(subset=["보수", "진보"])
         .groupby(["q_id", "direction"])[["보수", "진보"]].mean().unstack("direction"))
    per_prompt, passed = {}, []
    for q_id in g.index:
        sc = g.loc[q_id]
        cons_on_cons = sc[("보수", "conservative")]
        cons_on_prog = sc[("보수", "progressive")]
        prog_on_prog = sc[("진보", "progressive")]
        prog_on_cons = sc[("진보", "conservative")]
        sep_cons = cons_on_cons - cons_on_prog   # forced-보수 more 보수 than forced-진보
        sep_prog = prog_on_prog - prog_on_cons   # forced-진보 more 진보 than forced-보수
        ok = bool(sep_cons >= sep_margin and sep_prog >= sep_margin)
        per_prompt[int(q_id)] = {
            "sep_conservative_axis": round(float(sep_cons), 2),
            "sep_progressive_axis": round(float(sep_prog), 2),
            "passes": ok,
            "s_보수_A": round(float(cons_on_cons), 2), "s_보수_B": round(float(cons_on_prog), 2),
            "s_진보_A": round(float(prog_on_cons), 2), "s_진보_B": round(float(prog_on_prog), 2),
        }
        if ok:
            passed.append(int(q_id))
    return {
        "n_prompts": len(per_prompt), "n_pass": len(passed),
        "set_passes": len(passed) >= min_pass_per_set,
        "min_pass_per_set": min_pass_per_set, "sep_margin": sep_margin,
        "filtered_prompts": sorted(passed),
        "failed_prompts": sorted(set(per_prompt) - set(passed)),
        "mean_sep_conservative_axis": round(
            float(pd.DataFrame(per_prompt).T["sep_conservative_axis"].mean()), 3),
        "mean_sep_progressive_axis": round(
            float(pd.DataFrame(per_prompt).T["sep_progressive_axis"].mean()), 3),
        "per_prompt": per_prompt,
    }
