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
| `C_L3`    | Narrative-only persona baseline            |
| `C_base`  | 명시적인 정치 성향 label을 포함한 persona              |
| `C_bns`   | `C_base`에 Belief-Network Seeding을 추가한 조건   |
| `C_pv`    | `C_base`에 Persona Vector injection을 추가한 조건 |
| `C_full`  | BNS와 Persona Vector를 모두 적용한 조건             |

## 방법론

### 1. Synthetic voter agent 구축

Nemotron-Personas-Korea에서 persona narrative를 sampling하여 총 5,000명의 synthetic Korean voter agent를 구축합니다. 각 agent에는 demographic attribute와 5단계 political orientation을 부여하고, LLM judge를 통해 narrative와 orientation 간의 불일치를 완화합니다.

### 2. Belief Network Seeding

BNS는 KGSS의 정치 태도 문항에서 추출한 issue-level belief를 agent persona에 추가하는 prompt-level intervention입니다. 이를 통해 단순한 orientation label보다 구체적인 정책 신념과 정치적 태도를 반영합니다.

### 3. Persona Vector Steering

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

여기서 (G)는 평가 집단, (C)는 후보자 집합, (p_{g,c})와 (\hat{p}^{(m)}_{g,c})는 각각 실제 득표율과 condition (m)에서 생성된 득표율을 의미합니다.

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

### Requirements

본 프로젝트는 dependency 및 environment 관리를 위해 [uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
uv sync
```

Agent construction 과정에서는 Anthropic API를 사용하므로 환경 변수에 API key를 설정해야 합니다.

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
```

Main inference experiment의 실행 환경은 다음과 같습니다.

* Qwen3-30B-A3B
* vLLM backend
* NVIDIA RTX A6000 48GB GPU 4장
* Tensor parallel size 4
* Hugging Face model cache를 위한 최소 70GB 이상의 저장 공간

Model cache 경로는 다음과 같이 설정합니다.

```bash
export HF_HOME=/path/to/cache
```

Qwen3 model은 최초 실행 시 Hugging Face Hub에서 자동으로 다운로드됩니다.

Raw KGSS dataset, 전체 Nemotron persona dataset, 대용량 생성 결과 및 재배포가 제한된 파일은 repository에 포함하지 않았습니다.

## 사용 방법

각 pipeline stage의 결과는 `data/` 또는 `results/` 디렉터리에 저장됩니다. 이전 단계의 output이 존재하는 경우 이후 stage만 독립적으로 다시 실행할 수 있습니다.

### 1. KGSS factor analysis

```bash
uv run python scripts/kgss_factor_analysis.py \
    --config configs/config.yaml
```

KGSS 2016년 wave를 대상으로 Exploratory Factor Analysis를 수행하고, factor loading 및 factor score를 `data/kgss/`에 저장합니다.

### 2. Agent construction

```bash
uv run python scripts/construct_agents.py \
    --config configs/config.yaml
```

다음 과정을 통해 synthetic voter agent를 구축합니다.

1. Korean persona narrative sampling
2. Demographic-conditional political orientation 할당
3. Orientation distribution calibration
4. Narrative–orientation consistency 평가
5. 동일 지역 내 orientation swap

최종 agent pool은 다음 위치에 저장됩니다.

```text
data/agents/final_agents.parquet
```

### 3. BNS seed 할당

```bash
uv run python scripts/assign_bns_seeds.py \
    --config configs/config.yaml
```

각 agent에 KGSS 기반 factor score를 sampling하고, 이를 issue-level belief statement로 변환합니다.

### 4. Persona Vector 추출 및 검증

```bash
HF_HOME=$HF_HOME uv run python scripts/extract_persona_vectors.py \
    --config configs/config.yaml

HF_HOME=$HF_HOME uv run python scripts/layer_probing.py \
    --config configs/config.yaml

HF_HOME=$HF_HOME uv run python scripts/trait_validation.py \
    --config configs/config.yaml

HF_HOME=$HF_HOME uv run python scripts/alpha_sweep.py \
    --cell C_full
```

선택된 decoder layer, steering vector 및 intervention strength는 다음 파일에 저장됩니다.

```text
data/persona_vectors/SELECTED.json
```

### 5. Main inference

```bash
HF_HOME=$HF_HOME uv run python scripts/run_inference.py \
    --cells C_L3,C_base,C_bns,C_pv,C_full \
    --k-rollouts 1 \
    --tp 4 \
    --max-num-seqs 64 \
    --candidate-mode named \
    --out results/raw_outputs
```

각 condition의 생성 결과는 다음 경로에 JSONL 형식으로 저장됩니다.

```text
results/raw_outputs/<condition>/outputs.jsonl
```

### 6. 결과 분석

```bash
uv run python scripts/analyze_results.py
uv run python scripts/generate_case_studies.py
```

다음 분석 결과를 생성합니다.

* 전국 및 subgroup별 득표율
* 지역별, 연령별, 성별 결과
* Condition별 marginal contribution
* 전체 결과를 정리한 Markdown summary
* Representative agent case study

분석 결과는 다음 경로에 저장됩니다.

```text
results/analysis/
```

## 예시 결과

아래 표는 실제 득표율과 simulation으로 생성된 득표율 사이의 MAE를 percentage point 단위로 나타낸 결과입니다.

| 평가 수준    | `C_L3` | `C_base` |   `C_bns` | `C_pv` |  `C_full` |
| -------- | -----: | -------: | --------: | -----: | --------: |
| 전국       |  33.32 |    12.26 |      9.43 |  11.87 |  **9.26** |
| 지역 17개   |  32.01 |    12.87 | **11.12** |  12.94 |     11.34 |
| 연령 6개 집단 |  31.32 |    12.11 |     11.05 |  11.77 | **10.33** |
| 성별 2개 집단 |  31.73 |    10.58 |      7.77 |  10.18 |  **7.60** |

Narrative-only condition에서는 실제 득표율과 큰 차이가 나타났습니다. 반면 명시적인 political orientation을 prompt에 포함한 `C_base`에서는 모든 평가 수준에서 오차가 크게 감소하였습니다.

BNS는 전국, 지역 및 subgroup 평가에서 추가적인 성능 향상을 보였습니다. PV 단독 적용의 효과는 상대적으로 작고 일관되지 않았으며, BNS와 PV를 함께 적용한 `C_full`은 연령 및 성별 subgroup 평가에서 가장 낮은 MAE를 기록했습니다.

전체 결과와 세부 분석은 다음 파일에서 확인할 수 있습니다.

```text
results/analysis/summary.md
```

Representative agent의 condition별 투표 결과와 reasoning은 다음 경로에 저장되어 있습니다.

```text
results/analysis/case_studies/
```

## 팀원

* [이성은](https://github.com/lse072222): 실험 설계, 평가 지표 구성, 결과 분석 및 프로젝트 문서화
* [팀원 이름](https://github.com/ruaqktk): Agent construction pipeline 구현, model inference 및 실험 수행
* [팀원 이름](TEAM_MEMBER_GITHUB_LINK): 수행한 역할

## 프로젝트 자료

* [Project Notion](NOTION_LINK)
* [발표 자료](PRESENTATION_LINK)
