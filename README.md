# CGEV: Confidence-Gated Ensemble Verification for Actor-Critic Multi-Agent Systems

An extension of the SiriuS Actor-Critic framework (Zhao et al., 2025) that replaces
the single Judgment Agent with three specialized Verifiers and a Confidence Gate,
achieving **70.0% Overall Accuracy** on PubMedQA versus SiriuS's 50.6% (+19.4 pp).

## Architecture Overview

CGEV introduces three key innovations over SiriuS:
1. **Ensemble Judgment** — 3 specialized Verifiers (Evidence, Logic, Conclusion) replace the single Judgment Agent
2. **Confidence Gate** — a probabilistic routing mechanism with 3 paths: ACCEPT / REJECT / UNCERTAIN
3. **Step-Level Critic** — fine-grained step attribution replaces the holistic Janusian feedback prompt

## Pipeline

| Phase | Script | Description |
|---|---|---|
| 1 | `phase1_actor_solve.py` | Actor generates initial solutions on PubMedQA training set |
| 2 | `phase2_ensemble_judge.py` | 3 Verifiers evaluate each response; Confidence Gate routes to ACCEPT/REJECT/UNCERTAIN |
| 2.5 | `merge_cgev.py` | ACCEPT + UNCERTAIN → library.jsonl (bypass Critic); REJECT → reject.jsonl |
| 3 | `phase3_stepwise_feedback.py` | Step-Level Critic generates per-step feedback on REJECT items only |
| 4 | `phase4_actor_regenerate.py` | Actor regenerates answers using step-level feedback |
| 5A | `phase5a_build_finetune.py` | Builds 5 SFT training files (actor, verifier_a/b/c, critic) |
| 5B | `phase5b_submit_finetune.py` | Submits fine-tuning jobs to OpenAI one at a time |
| Eval | `evaluate.py` | End-to-end evaluation on PubMedQA test set |

## Results (PubMedQA, GPT-3.5-Turbo, 500 test questions)

| Method | TP Accuracy | Overall Accuracy | PF Rate ↓ | UT Rescue Rate |
|---|---|---|---|---|
| Self-Correct† | 11.80% | 16.40% | N/A | N/A |
| Prompt† | 18.40% | 47.60% | N/A | N/A |
| SiriuS† | 35.00% | 50.60% | N/A | N/A |
| CGEV base (Ours) | 47.60% | 53.80% | 0.83% | 37.74% |
| **CGEV fine-tuned (Ours)** | **31.80%** | **70.00%** | 22.06% | **86.15%** |

† Results from Zhao et al. (2025) Table 5

**Metrics:**
- **TP Accuracy** = PT / total — fraction correctly answered AND accepted by the gate
- **Overall Accuracy** = final_correct / total — after all regenerations
- **PF Rate** = PF / (PT+PF) — false rejection rate (lower is better)
- **UT Rescue Rate** = UT / (UT+UF) — correct answers preserved via UNCERTAIN path

## Setup

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate sirius

# 2. Set your OpenAI API key
cp .env.example .env
# Open .env and replace sk-... with your real OpenAI API key
```

## Running the Full Pipeline

```bash
# Load your API key from .env
export OPENAI_API_KEY=<paste-your-openai-key-here>
export MODEL=gpt-3.5-turbo-0125

# Set path variables
export BASE="logs/solve_PubMedQA_${MODEL}"
export EJ="${BASE}/sol/ensemble_judgement-${MODEL}"
export SF="${EJ}/stepwise_feedback-${MODEL}"
export FT="logs/actor_critic/generate/${MODEL}_${MODEL}"

# Phase 1
PYTHONPATH=. python phase1_actor_solve.py --model $MODEL --mode generate --subject PubMedQA
PYTHONPATH=. python merge.py --dir ${BASE}/sol

# Phase 2
PYTHONPATH=. python phase2_ensemble_judge.py --model $MODEL --input_file ${BASE}/sol/sol.jsonl

# Phase 2.5
PYTHONPATH=. python merge_cgev.py --base_dir ${EJ}

# Phase 3
PYTHONPATH=. python phase3_stepwise_feedback.py --model $MODEL --input_file ${EJ}/reject.jsonl
PYTHONPATH=. python merge.py --dir ${SF}/feedback

# Phase 4
PYTHONPATH=. python phase4_actor_regenerate.py --model $MODEL --input_file ${SF}/feedback/feedback.jsonl

# Phase 5A
PYTHONPATH=. python phase5a_build_finetune.py \
  --model $MODEL \
  --library_file      ${EJ}/library.jsonl \
  --correct_re_dir    ${SF}/feedback/regenerate-${MODEL}/correct_re \
  --all_judgement_dir ${EJ}/ALL \
  --feedback_dir      ${SF}/feedback

# Phase 5B (one agent at a time)
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents actor
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents verifier_a
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents verifier_b
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents verifier_c
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents critic

# Evaluate fine-tuned
PYTHONPATH=. python evaluate.py \
  --ft_ids_file ${FT}/finetuning_ids.jsonl \
  --input_file  dataset/PubMedQA_test.jsonl \
  --num_processes 4

# Evaluate base model
PYTHONPATH=. python evaluate.py \
  --base_model $MODEL \
  --input_file dataset/PubMedQA_test.jsonl \
  --num_processes 4
```

## Repo Structure

```
CGEV/
├── agent.py                    Agent class (LLM calls + fine-tuning upload)
├── args.py                     CLI argument parser
├── selective_gate.py           Confidence Gate logic + 6-label system
├── prompt.py                   Actor and Rephrase prompts
├── prompt_ensemble.py          3 Verifier prompts + Step-Level Critic prompts
├── merge.py                    Merge per-item JSONL files into one
├── merge_cgev.py               Route ACCEPT+UNCERTAIN → library, REJECT → critic
├── phase1_actor_solve.py       Phase 1: Actor initial solve
├── phase2_ensemble_judge.py    Phase 2: Ensemble judgement
├── phase3_stepwise_feedback.py Phase 3: Step-level critic feedback
├── phase4_actor_regenerate.py  Phase 4: Actor regeneration
├── phase5a_build_finetune.py   Phase 5A: Build SFT training files
├── phase5b_submit_finetune.py  Phase 5B: Submit fine-tuning jobs
├── evaluate.py                 Evaluation script
├── libs/
│   ├── data_loader.py
│   └── utils.py
├── dataset/
│   ├── PubMedQA_train.jsonl
│   └── PubMedQA_test.jsonl
└── logs/                       (gitignored — generated at runtime)
```

## Citation

This work extends:
```
@article{zhao2025sirius,
  title={SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning},
  author={Zhao, Wanjia and Yuksekgonul, Mert and Wu, Shirley and Zou, James},
  journal={arXiv preprint arXiv:2502.04780},
  year={2025}
}
```
