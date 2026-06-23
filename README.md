# Persona-Conditioned Korean Voter Simulation

📢 2026년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다.

## 소개

Large Language Model(LLM)을 persona-conditioned synthetic agent로 활용하여 실제 인간 집단의 행동을 모사하려는 연구가 활발히 이루어지고 있습니다. 그러나 demographic information이나 단순한 persona description만으로는 각 agent에게 부여된 정치적 성향이 일관되게 반영되지 않을 수 있으며, 모델이 사전학습 과정에서 획득한 정치적 편향이 persona condition을 압도할 가능성이 있습니다.

본 프로젝트에서는 **제21대 대한민국 대통령선거**를 대상으로, persona-conditioned LLM agent가 한국 유권자 집단의 실제 득표율 분포를 얼마나 잘 재현할 수 있는지 분석합니다.

총 5,000명의 synthetic Korean voter agent를 구축하고, **Qwen3-30B-A3B**를 backbone model로 사용하여 각 agent의 투표 행동을 생성하였습니다. 생성된 투표 분포는 다음 실제 통계와 비교하여 평가합니다.

* 중앙선거관리위원회가 공개한 전국 및 17개 시도별 공식 득표율
* 방송 3사 공동 출구조사에서 보고된 성별 및 연령대별 득표율 추정치

Agent에게 부여된 정치 성향을 보다 구체적이고 일관되게 반영하기 위해 다음 두 가지 intervention을 적용하였습니다.

* **Belief-Network Seeding(BNS)**: KGSS factor analysis를 기반으로 생성한 agent별 정책 신념 문장을 prompt에 추가
* **Persona Vector(PV) injection**: 보수–진보 방향을 나타내는 activation vector를 inference 과정에서 hidden representation에 주입

각 요소의 효과를 비교하기 위해 다음 다섯 가지 condition을 구성하였습니다.

| Condition | 설명                                         |
| --------- | ------------------------------------------ |
| `C_narr`    | Narrative-only persona baseline            |
| `C_base`  | 명시적인 정치 성향 label을 포함한 persona              |
| `C_bns`   | `C_base`에 Belief-Network Seeding을 추가한 조건   |
| `C_pv`    | `C_base`에 Persona Vector injection을 추가한 조건 |
| `C_full`  | BNS와 Persona Vector를 모두 적용한 조건             |

본 프로젝트에 대한 더 자세한 설명은 [Project Paper](docs/korean_voter_simulation.pdf) 및 [AIKU 노션](https://www.notion.so/Korean-Voter-Simulation-387a7930e09c80379a84db7f777f6a19?source=copy_link)에서 확인하실 수 있습니다.

## 방법론

### 1. Synthetic voter agent 구축

Nemotron-Personas-Korea에서 persona narrative를 sampling하여 총 5,000명의 synthetic Korean voter agent를 구축합니다. 각 agent에는 demographic attribute와 5단계 political orientation을 부여하고, LLM judge를 통해 narrative와 orientation 간의 불일치를 완화합니다.

### 2. Belief Network Seeding (BNS)

BNS는 KGSS의 정치 태도 문항에서 추출한 issue-level belief를 agent persona에 추가하는 prompt-level intervention입니다. 이를 통해 단순한 orientation label보다 구체적인 정책 신념과 정치적 태도를 반영합니다.

### 3. Persona Vector Steering (PV)

PV는 진보 및 보수 성향 prompt의 hidden activation 차이로부터 steering vector를 추출하는 activation-level intervention입니다.

```math
v_C^{\ell}
=
\mu_C^{\ell}
-
\mu_P^{\ell}
```

Inference 시 선택된 decoder layer의 hidden state에 agent의 orientation에 따른 vector를 주입합니다.

```math
h'_{\ell,t}
=
h_{\ell,t}
+
\alpha_i v_{o_i}^{\ell}
```

여기서 $\alpha_i$는 agent의 orientation strength에 따라 결정되며, Moderate agent에는 PV를 적용하지 않습니다.


### 4. Ablation 실험

동일한 5,000명의 agent를 대상으로 `C_narr`, `C_base`, `C_bns`, `C_pv`, `C_full`의 다섯 조건을 비교합니다. 이를 통해 explicit orientation, BNS, PV가 투표 분포에 미치는 효과를 각각 분석합니다.

### 5. 투표 생성 및 평가

Qwen3-30B-A3B를 사용하여 각 agent당 5개의 vote sample을 생성하고, 총 125,000개의 결과를 분석합니다. 생성된 vote-share distribution은 전국, 지역, 성별, 연령별 ground truth와 비교하여 MAE로 평가합니다.

```math
\mathrm{MAE}^{(m)}_G
=
\frac{1}{|G||C|}
\sum_{g \in G}
\sum_{c \in C}
\left|
\hat{p}^{(m)}_{g,c}
-
p_{g,c}
\right|
```

여기서 $`G`$는 평가 집단의 집합, $`C`$는 후보자 집합이며, $`p_{g,c}`$와 $`\hat{p}^{(m)}_{g,c}`$는 각각 실제 득표율과 condition $`m`$에서 생성된 득표율을 의미합니다.

### 6. Candidate-information prior 분석

후보자 이름, 정책 설명, 정당 정보를 단계적으로 제거하는 diagnostic anonymization experiment를 수행합니다. 이를 통해 candidate-related information이 model output과 persona-following behavior에 미치는 영향을 분석합니다.


## 환경 설정

### Repository 구조

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   └── config.yaml
├── prompts/
│   ├── system_prompts/
│   │   └── orientation_descriptions.txt
│   ├── user_prompts/
│   │   ├── candidate_info.txt
│   │   ├── instructions_main.txt
│   │   └── instructions_L3.txt
│   ├── pv_contrastive/
│   │   └── contrastive_prompts.txt
│   ├── bns_seeds/
│   │   └── seed_templates.txt
│   └── trait_validation_rubric.txt
├── src/
│   ├── agents/
│   ├── kgss/
│   ├── pv/
│   ├── inference/
│   └── utils/
├── scripts/
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
│   ├── ground_truth/
│   ├── candidates/
│   ├── persona_vectors/
│   ├── trait_validation/
│   └── kgss/
└── results/
    ├── raw_outputs/
    └── analysis/
```

## 환경 설정

본 프로젝트는 environment 관리를 위해 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
uv sync
```

GPU 및 model cache 설정은 다음과 같습니다.

* Main inference는 vLLM backend에서 tensor parallel size 4로 실행되며, NVIDIA RTX A6000 48GB GPU 4장이 필요합니다.
* vLLM이 설치된 별도의 environment가 필요합니다.
* Qwen3-30B-A3B model은 bf16 기준 약 61GB이므로, Hugging Face cache directory에는 최소 70GB 이상의 여유 공간이 필요합니다.

```bash
export HF_HOME=/path/to/cache
```

Qwen3 model은 최초 실행 시 Hugging Face Hub를 통해 자동으로 다운로드됩니다.

## 사용 방법

각 단계는 intermediate artifact를 `data/` 또는 `results/` 아래에 저장합니다. 이전 단계의 output이 이미 존재하는 경우, 이후 단계만 다시 실행할 수 있습니다.

### 1. KGSS factor analysis

```bash
uv run python scripts/kgss_factor_analysis.py --config configs/config.yaml
```

KGSS cumulative dataset을 불러와 2016년 wave에 Exploratory Factor Analysis를 수행하고, factor loading과 agent별 factor score를 `data/kgss/` 아래에 저장합니다.

### 2. Agent construction

```bash
uv run python scripts/construct_agents.py --config configs/config.yaml
```

Nemotron-Personas-Korea dataset에서 persona를 sampling하고, demographic-conditional political orientation을 할당하여 5,000명의 agent pool을 구축합니다.

이후 LLM-based narrative judge와 동일 demographic cell 내 swap 과정을 통해 narrative와 orientation 간의 충돌을 완화합니다.

Output:

```text
data/agents/final_agents.parquet
```

### 3. Persona Vector extraction

```bash
HF_HOME=$HF_HOME uv run python scripts/extract_persona_vectors.py \
    --config configs/config.yaml

HF_HOME=$HF_HOME uv run python scripts/layer_probing.py \
    --config configs/config.yaml

HF_HOME=$HF_HOME uv run python scripts/trait_validation.py \
    --config configs/config.yaml

HF_HOME=$HF_HOME uv run python scripts/alpha_sweep.py --cell C_full
```

위 네 단계를 통해 contrastive vector를 계산하고, candidate layer를 평가하며, cross-model judge를 통해 trait를 검증하고, candidate alpha value를 탐색합니다.

최종적으로 선택된 layer와 alpha는 다음 파일에 저장됩니다.

```text
data/persona_vectors/SELECTED.json
```

### 4. Main inference

```bash
HF_HOME=$HF_HOME uv run python scripts/run_inference.py \
    --cells C_narr,C_base,C_bns,C_pv,C_full \
    --k-rollouts 5 \
    --tp 4 \
    --max-num-seqs 64 \
    --candidate-mode named \
    --out results/raw_outputs
```

Qwen3-30B-A3B를 사용하여 5,000명의 agent에 대해 다섯 ablation condition을 실행합니다.

PV가 적용되는 condition에서는 Qwen3-MoE decoder layer에 forward-hook patch를 적용합니다. 각 condition의 결과는 다음 경로에 JSONL 형식으로 저장됩니다.

```text
results/raw_outputs/<cell>/outputs.jsonl
```

### 5. Analysis

```bash
uv run python scripts/analyze_results.py
uv run python scripts/generate_case_studies.py
```

분석 결과는 `results/analysis/` 아래에 저장됩니다.

주요 output은 다음과 같습니다.

* `summary.md`: 전체 결과를 정리한 Markdown summary
* `nation_by_cell.csv`: 전국 단위 결과
* `orientation_by_cell.csv`: 정치 성향별 결과
* `orientation_age_by_cell.csv`: 정치 성향 및 연령별 결과
* `sido_by_cell.csv`: 지역별 결과
* `age_by_cell.csv`: 연령별 결과
* `sex_by_cell.csv`: 성별 결과
* `marginal_contributions.csv`: condition별 marginal contribution
* `case_studies/agent_<id>.md`: representative agent의 condition별 vote 및 reasoning
* `case_studies_full/agent_<id>.md`: 전체 prompt와 response를 포함한 case study

## 예시 결과

<img width="844" height="730" alt="image" src="https://github.com/user-attachments/assets/cd0d6020-c974-4f7e-9845-bebccccaa025" />

<img width="866" height="303" alt="image" src="https://github.com/user-attachments/assets/8f85155d-81d2-4034-bbf5-ea132799f29d" />




## 팀원

* [강동혁](https://github.com/cucumber5252): 연구 목표 수립, 방법론 설계, Paper 작성
* [마현우](https://github.com/ruaqktk): 연구 목표 수립, 방법론 설계, 관련 연구 조사, 실험 수행
* [이성은](https://github.com/lse072222): 연구 목표 수립, 방법론 설계, 결과 분석
