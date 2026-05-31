# Persona-Conditioned LLM Vote Simulation

A research framework that tests whether a persona-conditioned large language
model can reproduce the vote-share distribution of Korean voter subpopulations
for the 21st Korean Presidential Election (2025-06-03), using the Qwen3-30B-A3B
model as the backbone.

The framework introduces two complementary regularizers on top of a vanilla
persona prompt:

- Belief-Network Seeding (BNS): per-agent policy-belief sentences derived from
  KGSS exploratory factor analysis (4 factors: economic redistribution, North
  Korea / security, market / privatization, political efficacy).
- Persona Vector (PV) injection: a contrastive activation difference vector
  (`v_conservative`) added to the residual stream at a selected decoder layer,
  with the magnitude varying per orientation strength.

Five inference conditions form an ablation ladder
(C_L3 -> C_base -> C_bns / C_pv -> C_full) that measures the marginal
contribution of orientation labels, BNS seeds, and PV injection.

## Repository layout

```
.
├── README.md
├── pyproject.toml, uv.lock         (dependency lock)
├── configs/
│   └── config.yaml                 (model, EFA, persona-vector, paths)
├── prompts/                        (runtime prompts)
│   ├── system_prompts/
│   │   └── orientation_descriptions.txt
│   ├── user_prompts/
│   │   ├── candidate_info.txt
│   │   ├── instructions_main.txt
│   │   └── instructions_L3.txt
│   ├── pv_contrastive/contrastive_prompts.txt
│   ├── bns_seeds/seed_templates.txt
│   └── trait_validation_rubric.txt
├── src/
│   ├── agents/                     (agent construction pipeline)
│   ├── kgss/                       (KGSS EFA + factor sampling)
│   ├── pv/                         (persona vector extraction + injection)
│   ├── inference/                  (prompt builder + vote parser)
│   └── utils/
├── scripts/                        (entry points; one per pipeline stage)
│   ├── kgss_factor_analysis.py
│   ├── construct_agents.py
│   ├── narrative_swap.py
│   ├── assign_bns_seeds.py
│   ├── extract_persona_vectors.py
│   ├── layer_probing.py
│   ├── trait_validation.py
│   ├── alpha_sweep.py
│   ├── run_inference.py
│   ├── analyze_results.py
│   └── generate_case_studies.py
├── data/
│   ├── ground_truth/               (NEC official + KEP exit poll)
│   ├── candidates/                 (candidate roster + policy text)
│   ├── persona_vectors/            (selected layer / vector / alpha)
│   ├── trait_validation/           (filtered prompts + summary)
│   └── kgss/                       (EFA outputs only; raw .sav not committed)
└── results/
    ├── raw_outputs/                (per-cell JSONL outputs from inference)
    └── analysis/                   (CSV tables + Markdown summary + case studies)
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
uv sync
```

GPU + model cache:

- The main inference run requires 4 x NVIDIA RTX A6000 (48 GB each) under
  tensor-parallel size 4 with the vLLM backend, plus a separate environment
  with vLLM installed.
- The HuggingFace cache directory must point to a volume with at least 70 GB
  free (Qwen3-30B-A3B is approximately 61 GB in bf16):
  ```bash
  export HF_HOME=/path/to/cache
  ```

The Qwen3 model is downloaded on first use through the HuggingFace Hub.

## Pipeline

Each stage writes intermediate artifacts under `data/` or `results/`. Re-running
a later stage does not require re-running earlier stages once their outputs are
present.

### 1. KGSS factor analysis

```bash
uv run python scripts/kgss_factor_analysis.py --config configs/config.yaml
```

Loads the KGSS cumulative dataset, performs exploratory factor analysis on the
2016 wave, and writes factor loadings and per-agent factor scores under
`data/kgss/`.

### 2. Agent construction

```bash
uv run python scripts/construct_agents.py --config configs/config.yaml
```

Builds the 5,000-agent pool by sampling personas from the
Nemotron-Personas-Korea dataset, assigning demographic-conditional orientation
labels, calibrating to the target distribution, and running an LLM-based
narrative judge (Claude API; requires `ANTHROPIC_API_KEY`) followed by a
within-region swap step that resolves narrative-orientation conflicts.

Outputs: `data/agents/final_agents.parquet`.

### 3. Persona vector extraction

```bash
HF_HOME=$HF_HOME uv run python scripts/extract_persona_vectors.py \
    --config configs/config.yaml
HF_HOME=$HF_HOME uv run python scripts/layer_probing.py \
    --config configs/config.yaml
HF_HOME=$HF_HOME uv run python scripts/trait_validation.py \
    --config configs/config.yaml
HF_HOME=$HF_HOME uv run python scripts/alpha_sweep.py --cell C_full
```

These four steps together compute the contrastive vector
`v_conservative = mean(conservative activations) - mean(progressive
activations)` at each candidate layer, evaluate each layer with three metrics
(separation, causal effect, mention shift), validate the resulting trait via a
cross-model judge, and sweep candidate alpha values. The selected layer and
alpha are recorded in `data/persona_vectors/SELECTED.json`.

### 4. Main inference

```bash
HF_HOME=$HF_HOME uv run python scripts/run_inference.py \
    --cells C_L3,C_base,C_bns,C_pv,C_full \
    --k-rollouts 1 --tp 4 --max-num-seqs 64 \
    --candidate-mode named \
    --out results/raw_outputs
```

Runs the 5,000-agent x 5-cell inference on Qwen3-30B-A3B under vLLM with the PV
forward-hook patch applied to the Qwen3-MoE decoder layer. Each cell writes a
JSON-lines file under `results/raw_outputs/<cell>/outputs.jsonl`.

### 5. Analysis

```bash
uv run python scripts/analyze_results.py
uv run python scripts/generate_case_studies.py
```

Generates the report under `results/analysis/`:

- `summary.md`: human-readable Markdown summary across all tables.
- `nation_by_cell.csv`, `orientation_by_cell.csv`,
  `orientation_age_by_cell.csv` (long + wide), `sido_by_cell.csv`,
  `age_by_cell.csv`, `sex_by_cell.csv`, `marginal_contributions.csv`.
- `case_studies/agent_<id>.md`: per-cell vote + reasoning trace for two
  illustrative agents (1805 and 4205).
- `case_studies_full/agent_<id>.md`: same agents with the complete prompt
  shown alongside each response.

## Key results (main inference)

Mean Absolute Error (MAE) of predicted vote shares against the ground truth,
in percentage points. Ground truth: NEC official results for nation and sido,
KEP exit poll for age and sex.

| Metric | C_L3 | C_base | C_bns | C_pv | C_full |
|---|---|---|---|---|---|
| Nation | 33.32 | 12.26 |  9.43 | 11.87 |  9.26 |
| Sido (17 region) | 32.01 | 12.87 | 11.12 | 12.94 | 11.34 |
| Age (6 buckets) | 31.32 | 12.11 | 11.05 | 11.77 | 10.33 |
| Sex (2 groups) | 31.73 | 10.58 |  7.77 | 10.18 |  7.60 |

Detailed analysis with per-region, per-age, per-sex breakdowns is in
`results/analysis/summary.md`.

## Citation

This repository accompanies a coursework submission. Please contact the author
for citation details if this work is referenced in subsequent research.
