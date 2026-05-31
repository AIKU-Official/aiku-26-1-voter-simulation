"""Agent construction pipeline.

Step 1   sampling.py      — Nemotron 5,000 persona sampling + narrative assembly
Step 2a  orientation.py   — KGSS demographic-conditional 5-way orientation
Step 2b  orientation.py   — Gallup 26/48/26 calibration + Kang belief_scores
Step 3   narrative_judge.py — Claude API narrative_lean inference (needs API key)
Step 4   swap.py          — within-sido pair-based orientation swap
Steps 7-8 bns.py          — direct-resampling factor scores + BNS seed selection

See docs/narrative_orientation.md and configs/config.yaml (orientation block).
"""
