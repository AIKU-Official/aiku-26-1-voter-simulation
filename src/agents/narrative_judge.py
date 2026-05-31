"""Step 3 (B-1) — narrative_lean inference.

Cross-model judge: each Nemotron narrative → {VP,P,M,C,VC,ambiguous}. Two backends
share the rubric (docs/narrative_orientation.md Part B-1) + JSON parsing +
conservative post-processing:
  - LOCAL (default): EXAONE 4.0 32B via HuggingFace, on GPU, no API / no cost.
  - Claude API: kept as an alternative (run_judge, needs ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

DEFAULT_MODEL = "claude-opus-4-7"          # Claude backend
FALLBACK_MODEL = "claude-sonnet-4-6"
LOCAL_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"  # local backend

SYSTEM_PROMPT = """당신은 한국 정치 분석 전문가입니다. 주어진 한국인 페르소나 narrative만 기반으로 그 사람의 정치 성향을 추론합니다.

## 추론할 정치 성향 (6가지 중 하나)
- VP (Very Progressive, 강한 진보): 명백히 진보적 가치(사회적 평등, 분배, 약자 보호, 환경, 다양성)에 강하게 끌리는 라이프스타일·직업·관심사. 예: 인권 활동가, 환경 운동, 페미니스트 잡지 구독.
- P (Progressive, 진보): 진보 가치에 호의적이나 강도가 moderate. 예: 인디 음악, 예술 감수성, 평등 관련 가치 언급.
- M (Moderate, 중도): 양쪽 가치 모두 포함하거나 정치적 색채가 옅음. 실용주의·일상 중심.
- C (Conservative, 보수): 전통·안정·시장경제·가족 가치에 호의적. 예: 골프, 종교 활동, 자영업 운영, 자녀 교육 강조.
- VC (Very Conservative, 강한 보수): 명백히 보수 가치(전통, 권위, 안보, 종교)에 강하게 끌리는 라이프스타일. 예: 군 경력 강조, 보수 미디어 정기 구독, 종교 보수.
- AMBIGUOUS: 정치 성향을 추론할 신호가 부족하거나 양쪽이 섞여 판단 불가.

## 평가 원칙
- narrative 텍스트만 보고 판단. demographic(나이·성별·지역) 통계적 prior는 무시.
- 직업·취미·관심사·라이프스타일·가치관 등 모든 단서를 종합 고려.
- 단순 한두 단서로 강한 분류 금지. confidence 낮으면 AMBIGUOUS 또는 P/C(moderate).
- 정치적 단어(보수·진보·민주·자유)가 명시적으로 나오는 경우만 의존하지 말 것.

## Confidence (1-5)
- 5: 매우 명확한 정치 신호  4: 분명하나 약간 모호  3: 중간, 합리적 추론  2: 약한 신호  1: 거의 없음 → AMBIGUOUS

응답은 다음 JSON 형식으로만 작성하세요. 다른 텍스트 금지:
{"lean": "<VP|P|M|C|VC|AMBIGUOUS>", "confidence": <1-5>, "key_evidence": "주된 단서 2-3개를 콤마로", "reason": "한 문장 추론"}"""

_VALID = {"VP", "P", "M", "C", "VC", "AMBIGUOUS"}


def post_process_lean(j: dict) -> str:
    """Conservative labelling: low confidence → ambiguous; conf 3 demotes VP/VC."""
    lean, conf = j.get("lean", "AMBIGUOUS"), int(j.get("confidence", 1))
    if lean not in _VALID:
        return "ambiguous"
    if conf <= 2:
        return "ambiguous"
    if conf == 3:
        if lean == "VP":
            return "p"
        if lean == "VC":
            return "c"
    return lean.lower()


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in response: {text[:120]}")
    return json.loads(m.group(0))


def _client():
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot run the narrative judge.")
    return anthropic.Anthropic()


def judge_one(client, narrative: str, model: str, max_retries: int = 4) -> dict:
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                system=[{"type": "text", "text": SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"[Narrative]\n{narrative}"}],
            )
            j = _parse(resp.content[0].text)
            return {
                "raw_lean": j.get("lean"),
                "confidence": j.get("confidence"),
                "key_evidence": j.get("key_evidence", ""),
                "reason": j.get("reason", ""),
                "lean": post_process_lean(j),
            }
        except Exception as e:  # noqa: BLE001 — network/parse, retry then surface
            last = e
            time.sleep(2 ** attempt)
    return {"raw_lean": None, "confidence": None, "key_evidence": "",
            "reason": f"ERROR: {last}", "lean": "ambiguous"}


def run_judge(agents: pd.DataFrame, model: str = DEFAULT_MODEL, workers: int = 8,
              limit: int | None = None, log=None) -> pd.DataFrame:
    """Judge each agent's narrative (concurrent). Returns narrative_lean rows."""
    client = _client()
    rows = agents if limit is None else agents.head(limit)
    results: dict[int, dict] = {}

    def work(rec):
        return rec["agent_id"], judge_one(client, rec["narrative"], model)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for _, r in rows.iterrows()]
        for f in as_completed(futs):
            aid, res = f.result()
            results[aid] = res
            done += 1
            if log and done % 250 == 0:
                log.info("  judged %d/%d", done, len(rows))

    out = pd.DataFrame(
        [{"agent_id": aid, **results[aid]} for aid in rows["agent_id"]]
    )
    return out


# ---------------------------------------------------------------------------
# Local backend — EXAONE 4.0 32B via HuggingFace (no API, no cost)
# ---------------------------------------------------------------------------
def _result_from_text(text: str) -> dict:
    try:
        j = _parse(text)
        return {
            "raw_lean": j.get("lean"),
            "confidence": j.get("confidence"),
            "key_evidence": j.get("key_evidence", ""),
            "reason": j.get("reason", ""),
            "lean": post_process_lean(j),
        }
    except Exception as e:  # noqa: BLE001
        return {"raw_lean": None, "confidence": None, "key_evidence": "",
                "reason": f"PARSE_ERR: {str(e)[:60]} | {text[:80]}", "lean": "ambiguous"}


def load_local(model_id: str = LOCAL_MODEL):
    """Load EXAONE (bf16, sharded across GPUs). 32B≈64GB → ≥2×A6000."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tok, model


def _apply_template(tok, narrative: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Narrative]\n{narrative}"},
    ]
    try:  # EXAONE 4.0 hybrid-reasoning: force non-reasoning for direct JSON
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_judge_local(agents: pd.DataFrame, model_id: str = LOCAL_MODEL,
                    batch_size: int = 16, max_new_tokens: int = 256,
                    limit: int | None = None, log=None, _loaded=None) -> pd.DataFrame:
    """Judge narratives with a local EXAONE model (greedy, batched)."""
    import torch

    tok, model = _loaded if _loaded else load_local(model_id)
    rows = agents if limit is None else agents.head(limit)
    items = list(zip(rows["agent_id"].tolist(), rows["narrative"].tolist()))
    results: dict = {}

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [_apply_template(tok, narr) for _, narr in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for (aid, _narr), text in zip(batch, texts):
            results[aid] = _result_from_text(text)
        if log:
            log.info("  judged %d/%d", min(i + batch_size, len(items)), len(items))

    return pd.DataFrame([{"agent_id": aid, **results[aid]} for aid in rows["agent_id"]])
