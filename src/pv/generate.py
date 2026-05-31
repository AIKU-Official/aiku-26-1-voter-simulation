"""Qwen3-30B forced-response generation + residual-stream activation capture.

One pass yields both (a) the response text (for EXAONE trait validation) and
(b) per-probe-layer, response-token-averaged residual activations (for PV
extraction). Activations are captured via forward hooks on model.model.layers[L-1]
(whose output == hidden_states[L]) — memory-efficient vs output_hidden_states.
"""
from __future__ import annotations

import numpy as np


def load_qwen(model_id: str, dtype="bfloat16", max_memory_per_gpu: str | None = None):
    """Load Qwen sharded across visible GPUs. `max_memory_per_gpu` (e.g. '32GiB')
    forces a balanced split + headroom — important on a shared box where another
    process already holds ~11GB/GPU (else device_map='auto' can overload one GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = {"torch_dtype": getattr(torch, dtype), "device_map": "auto"}
    if max_memory_per_gpu:
        n = torch.cuda.device_count()
        kwargs["max_memory"] = {i: max_memory_per_gpu for i in range(n)}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tok, model


def _wrap_prompt(tok, prompt: str, enable_thinking: bool = False) -> str:
    """Apply chat template (non-reasoning) so the model answers as a chat turn."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=enable_thinking)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate_with_activations(tok, model, prompts: list[str], probe_layers: list[int],
                              max_new_tokens: int = 100, temperature: float = 0.7,
                              enable_thinking: bool = False):
    """Greedy/sampled generation for a batch of raw prompts. Returns
    (texts, {layer: (batch, hidden) activations})."""
    import torch

    captured: dict[int, list] = {L: [] for L in probe_layers}
    handles = []
    for L in probe_layers:
        layer = model.model.layers[L - 1]  # output == hidden_states[L]

        def mk(LL):
            def hook(_m, _i, o):
                h = o[0] if isinstance(o, tuple) else o
                captured[LL].append(h[:, -1, :].detach().float().cpu())
            return hook
        handles.append(layer.register_forward_hook(mk(L)))

    wrapped = [_wrap_prompt(tok, p, enable_thinking) for p in prompts]
    enc = tok(wrapped, return_tensors="pt", padding=True, truncation=True,
              max_length=2048).to(model.device)
    try:
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=temperature or None,
                pad_token_id=tok.pad_token_id, return_dict_in_generate=True)
    finally:
        for h in handles:
            h.remove()

    seqs = out.sequences[:, enc["input_ids"].shape[1]:]      # (batch, gen_len)
    texts = tok.batch_decode(seqs, skip_special_tokens=True)
    gen_len = (seqs != tok.pad_token_id).sum(1).clamp(min=1)  # real response length/seq

    acts = {}
    for L in probe_layers:
        steps = torch.stack(captured[L][1:], dim=1)          # (batch, n_decode, hidden); [0]=prefill
        vecs = [steps[b, :int(gen_len[b])].mean(0) for b in range(steps.shape[0])]
        acts[L] = torch.stack(vecs).numpy()
    return texts, acts


def build_items(qs: list[dict], rollouts: int) -> list[dict]:
    """Flatten (prompt × {conservative, progressive} × rollout) work items."""
    from .contrastive import build_forced_prompt
    items = []
    for q in qs:
        for direction in ("conservative", "progressive"):
            prompt = build_forced_prompt(q, direction)
            for r in range(rollouts):
                items.append({"q_id": q["q_id"], "direction": direction,
                              "rollout": r, "prompt": prompt})
    return items
