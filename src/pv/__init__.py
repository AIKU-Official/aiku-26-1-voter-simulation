"""Persona Vector (PV) extraction + injection.

contrastive.py   — parse the 50 contrastive prompts + forced-response prompt
generate.py      — Qwen3-30B forced-response generation (text + per-layer activations)
trait_validation.py — EXAONE cross-model trait judging + prompt filtering
layer_probing.py — v_보수/v_진보 extraction + separation/causal/mvm metrics
injection.py     — forward-hook PV injection

See docs/persona_vectors.md and configs/config.yaml (persona_vector block).
"""
