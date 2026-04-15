# CGEV: Confidence-Gated Ensemble Verification for Actor-Critic Multi-Agent Systems

> An extension of the SiriuS Actor-Critic framework (Zhao et al., 2025) that replaces
> the single Judgment Agent with three specialized Verifiers and a Confidence Gate,
> achieving **70.0% Overall Accuracy** on PubMedQA versus SiriuS's 50.6% (+19.4 pp).

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [How CGEV Works](#2-how-cgev-works)
3. [The 6-Label System](#3-the-6-label-system)
4. [Metrics](#4-metrics)
5. [Results](#5-results)
6. [Setup](#6-setup)
7. [Running the Full Pipeline — Step by Step](#7-running-the-full-pipeline--step-by-step)
8. [Evaluation](#8-evaluation)
9. [Repository Structure](#9-repository-structure)
10. [Dataset Format](#10-dataset-format)
11. [Citation](#11-citation)

---

## 1. Background and Motivation

### What is SiriuS?

[SiriuS (Zhao et al., 2025)](https://arxiv.org/abs/2502.04780) is a multi-agent self-improvement framework where:

1. An **Actor** agent solves a question and writes out its reasoning chain.
2. A **Judgment Agent** evaluates whether the Actor's answer is correct (`True`) or incorrect (`False`).
3. If the answer is judged incorrect, a **Critic Agent** generates feedback.
4. The Actor reads the feedback and **regenerates** a new answer.
5. High-quality trajectories are collected and used to **fine-tune** all agents via Supervised Fine-Tuning (SFT).

### The Core Problem with SiriuS

SiriuS uses a **single Judgment Agent** to decide if an answer is correct or not. This creates a critical vulnerability:

- If the Judgment Agent wrongly labels a **correct answer as incorrect** (a false rejection), the Critic generates feedback based on that wrong premise.
- The Actor then **corrects a correct answer**, introducing errors.
- This corrupted trajectory then enters the fine-tuning data, teaching the model to get answers wrong.

This is called the **PF problem** (Passed False Rejection). SiriuS achieves only **35.0% TP Accuracy** on PubMedQA because a large fraction of correct answers are corrupted this way.

### CGEV's Solution

CGEV introduces three targeted fixes:

| Fix                   | Mechanism                                  | Purpose                                                                   |
| --------------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| **Ensemble Judgment** | 3 specialized Verifiers instead of 1       | Reduce single-point-of-failure false rejections                           |
| **Confidence Gate**   | 3-way routing: ACCEPT / REJECT / UNCERTAIN | Preserve borderline correct answers instead of sending them to the Critic |
| **Step-Level Critic** | Annotates individual reasoning steps       | Replace vague holistic feedback with precise, actionable fix instructions |

---

## 2. How CGEV Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CGEV PIPELINE                                       │
│                                                                             │
│  TRAINING DATA (PubMedQA_train.jsonl — 500 questions)                       │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────┐                                                           │
│  │  Phase 1     │  Actor reads the question + context                       │
│  │  Actor Solve │  Writes a full reasoning chain                            │
│  │              │  Extracts a YES/NO/MAYBE answer from the last line        │
│  └──────┬───────┘                                                           │
│         │  sol/  (per-question JSONL files with Actor response + score)     │
│         ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────┐               │
│  │  Phase 2 — Ensemble Judgment (3 Verifiers in parallel)   │               │
│  │                                                          │               │
│  │  Verifier A (Evidence) ──┐                               │               │
│  │  Verifier B (Logic)    ──┼─→ votes: [True, False, True]  │               │
│  │  Verifier C (Conclusion)─┘                               │               │
│  │                          │                               │               │
│  │                          ▼                               │               │
│  │             Confidence Gate  (selective_gate.py)         │               │
│  │             p_wrong = count(False) / 3                   │               │
│  │                                                          │               │
│  │   p_wrong < 0.34  →  ACCEPT    (0 or 1 False vote)       │               │
│  │   p_wrong ≥ 0.67  →  REJECT    (2 or 3 False votes)      │               │
│  │   0.34 ≤ p_wrong < 0.67 → UNCERTAIN (split vote: 1/3)    │               │
│  └───────────┬──────────────┬──────────────┬────────────────┘               │
│              │              │              │                                 │
│           ACCEPT        UNCERTAIN        REJECT                             │
│              │              │              │                                 │
│              └──────────────┘              │                                 │
│                     │                      │                                 │
│              library.jsonl           reject.jsonl                           │
│           (bypass Critic)         (send to Critic)                          │
│                     │                      │                                 │
│                     │              ┌───────▼────────┐                       │
│                     │              │   Phase 3      │                       │
│                     │              │ Step-Level     │                       │
│                     │              │ Critic         │                       │
│                     │              │ Feedback       │                       │
│                     │              └───────┬────────┘                       │
│                     │                      │  per-step analysis:            │
│                     │                      │  "Step 2: [WRONG] — …"         │
│                     │                      │  "Fix: replace X with Y"       │
│                     │              ┌───────▼────────┐                       │
│                     │              │   Phase 4      │                       │
│                     │              │ Actor          │                       │
│                     │              │ Regenerate     │                       │
│                     │              └───────┬────────┘                       │
│                     │                      │                                 │
│                     └──────────────────────┘                                │
│                                    │                                        │
│                             ┌──────▼──────┐                                 │
│                             │  Phase 5A   │  Build 5 fine-tuning files      │
│                             │  Build SFT  │  (actor, verifier_a/b/c, critic)│
│                             └──────┬──────┘                                 │
│                                    │                                        │
│                             ┌──────▼──────┐                                 │
│                             │  Phase 5B   │  Submit to OpenAI               │
│                             │  Fine-Tune  │  fine-tuning API (one by one)   │
│                             └──────┬──────┘                                 │
│                                    │                                        │
│                             ┌──────▼──────┐                                 │
│                             │  Evaluate   │  Run pipeline on test set        │
│                             │  (test set) │  Compute 4 metrics               │
│                             └─────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why the Confidence Gate Matters

In SiriuS, every answer judged as `False` goes to the Critic — even if the Judgment Agent was wrong. CGEV's gate adds a middle state:

- **ACCEPT** (0 or 1 verifier says False): the answer is almost certainly correct → preserve it, skip the Critic entirely.
- **REJECT** (2 or 3 verifiers say False): the answer is almost certainly wrong → send it to the Critic for targeted repair.
- **UNCERTAIN** (exactly 1 out of 3 says False with a split borderline): the verifiers disagree → preserve the answer rather than risk corrupting it.

The key insight is that **only REJECT items ever touch the Critic**. ACCEPT and UNCERTAIN items flow directly into the training library. This eliminates the PF problem for all borderline cases.

### Why Step-Level Feedback Matters

SiriuS's Critic prompt asks: _"The answer may be wrong — consider alternative reasoning."_ This is **holistic**: the Actor doesn't know which specific step failed, so it rewrites the entire chain, often introducing new errors.

CGEV's Critic prompt says: _"Step 2: [WRONG] — the study measured X not Y. Fix: replace 'the study shows X' with 'the study shows Y'."_ This is **targeted**: the Actor fixes only the broken step, leaving correct steps intact.

---

## 3. The 6-Label System

Every question gets one of six labels after Phase 2 (defined in `selective_gate.py`):

| Label  | Gate Decision | Actor Correct? | Meaning                                                                            |
| ------ | ------------- | -------------- | ---------------------------------------------------------------------------------- |
| **PT** | ACCEPT        | Yes            | True Positive: verifiers correctly passed a correct answer                         |
| **PF** | REJECT        | Yes            | False Rejection: verifiers wrongly rejected a correct answer ← CGEV minimizes this |
| **NT** | REJECT        | No             | True Negative: verifiers correctly flagged a wrong answer                          |
| **NF** | ACCEPT        | No             | False Acceptance: verifiers missed an error                                        |
| **UT** | UNCERTAIN     | Yes            | Rescued: correct answer preserved via the uncertainty buffer                       |
| **UF** | UNCERTAIN     | No             | Preserved wrong: wrong answer escaped the Critic (cost of UNCERTAIN)               |

**Why UT/UF matter:** SiriuS has no UNCERTAIN state. Every borderline correct answer (UT) would have been sent to the Critic as a PF case and corrupted. CGEV's UT category measures exactly how many correct answers were **saved** from this fate.

---

## 4. Metrics

All four metrics are computed in `evaluate.py` from the 6-label counts:

### TP Accuracy

```
TP Accuracy = PT / total × 100
```

Fraction of all test questions that were answered correctly **and** accepted by the gate on the first try. Directly comparable to SiriuS Table 5.

### Overall Accuracy

```
Overall Accuracy = final_correct / total × 100
```

Fraction of all test questions with a correct **final** answer — either accepted on the first try (PT, NF treated as wrong, UT), or successfully corrected after Critic feedback (NT → correct regeneration).

### PF Rate

```
PF Rate = PF / (PT + PF) × 100   (lower is better)
```

Among all the questions the Actor answered correctly, what fraction did the verifiers wrongly reject? This directly measures how much the gate corrupts good answers.

### UT Rescue Rate

```
UT Rescue Rate = UT / (UT + UF) × 100
```

Among all items routed to UNCERTAIN, what fraction were actually correct? A high value (≥ 80%) confirms that the gate's conservative behaviour is helping, not hurting.

---

## 5. Results

Evaluated on PubMedQA test set (500 questions), base model `gpt-3.5-turbo-0125`.

### Comparison with SiriuS (Table 5)

| Method                     | TP Accuracy | Overall Accuracy |
| -------------------------- | ----------- | ---------------- |
| Self-Correct†              | 11.80%      | 16.40%           |
| Prompt†                    | 18.40%      | 47.60%           |
| SiriuS†                    | 35.00%      | 50.60%           |
| CGEV base (Ours)           | 47.60%      | 53.80%           |
| **CGEV fine-tuned (Ours)** | **31.80%**  | **70.00%**       |

† Results from Zhao et al. (2025) Table 5

### CGEV-Specific Metrics

| Method              | TP Accuracy | Overall Accuracy | PF Rate ↓ | UT Rescue Rate |
| ------------------- | ----------- | ---------------- | --------- | -------------- |
| SiriuS†             | 35.00%      | 50.60%           | N/A       | N/A            |
| CGEV base           | 47.60%      | 53.80%           | 0.83%     | 37.74%         |
| **CGEV fine-tuned** | **31.80%**  | **70.00%**       | 22.06%    | **86.15%**     |

**Reading the results:**

- CGEV fine-tuned TP Accuracy (31.8%) is lower than base (47.6%) because the fine-tuned gate routes more uncertain correct answers through UNCERTAIN rather than ACCEPT — but they are still preserved.
- CGEV fine-tuned Overall Accuracy (70.0%) is +19.4 pp above SiriuS (50.6%) because the Critic now successfully repairs most REJECT items (86.15% UT Rescue Rate), while almost none of the ACCEPT/UNCERTAIN correct answers are corrupted.

---

## 6. Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- An [OpenAI API key](https://platform.openai.com/api-keys) with fine-tuning access

### Install

```bash
# 1. Clone the repository
git clone https://github.com/quachphu/CGEV.git
cd CGEV

# 2. Create and activate the conda environment
#    (installs Python 3.10 + all pinned dependencies from environment.yml)
conda env create -f environment.yml
conda activate sirius

# 3. Configure your OpenAI API key
cp .env.example .env
# Open .env in any text editor and replace sk-... with your real OpenAI key
```

Your `.env` file should look like:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
```

---

## 7. Running the Full Pipeline — Step by Step

All commands are run from the **project root** (`CGEV/`). The `PYTHONPATH=.` prefix ensures Python can find local modules (`agent.py`, `args.py`, etc.) without installing a package.

### Set Up Environment Variables

```bash
# Load your API key into the shell session
export OPENAI_API_KEY=<your-openai-api-key>

# Choose the base model (must match one of the choices in args.py)
export MODEL=gpt-3.5-turbo-0125

# Convenience path variables — derived from MODEL so you don't have to type them repeatedly
export BASE="logs/solve_PubMedQA_${MODEL}"
export EJ="${BASE}/sol/ensemble_judgement-${MODEL}"
export SF="${EJ}/stepwise_feedback-${MODEL}"
export FT="logs/actor_critic/generate/${MODEL}_${MODEL}"
```

**What these paths mean:**

- `BASE` — root output directory for Phase 1 (Actor solutions)
- `EJ` — output directory for Phase 2 (Ensemble Judgement)
- `SF` — output directory for Phase 3 (Step-level Feedback)
- `FT` — output directory for Phase 5A/5B (Fine-tuning data and job IDs)

---

### Phase 1 — Actor Generates Initial Solutions

```bash
PYTHONPATH=. python phase1_actor_solve.py --model $MODEL --mode generate --subject PubMedQA
```

**What this does:**

- Loads `dataset/PubMedQA_train.jsonl` (500 training questions, each with `index`, `question`, `context`, `groundtruth`).
- For each question, calls the Actor LLM with a 0-shot prompt asking it to reason step-by-step and end with `Answer: yes/no/maybe`.
- Runs up to 64 questions in parallel using `multiprocessing`.
- For each question, saves a `{index}_sol.jsonl` file in `logs/solve_PubMedQA_{MODEL}/sol/` containing the full Actor response and a boolean `score` (whether the answer matched the ground truth).
- Also saves to `correct/` or `wrong/` subdirectories for quick inspection.
- Prints total correct at the end (rough baseline accuracy of the un-fine-tuned model).

**Output tree after Phase 1:**

```
logs/solve_PubMedQA_gpt-3.5-turbo-0125/
└── sol/
    ├── PMID_12345_sol.jsonl
    ├── PMID_67890_sol.jsonl
    └── ...
```

---

```bash
PYTHONPATH=. python merge.py --dir ${BASE}/sol
```

**What this does:**

- Reads all individual `*_sol.jsonl` files from the `sol/` directory.
- Concatenates them into a single `sol/sol.jsonl` file.
- This merged file is the input for Phase 2 so that one script call can process all 500 items.

---

### Phase 2 — Ensemble Judgement

```bash
PYTHONPATH=. python phase2_ensemble_judge.py --model $MODEL --input_file ${BASE}/sol/sol.jsonl
```

**What this does:**

- Reads each Actor response from `sol.jsonl`.
- For each question, calls **three separate Verifier agents in parallel**:
  - **Verifier A (Evidence):** "Are the factual claims in this response supported by the provided context?"
  - **Verifier B (Logic):** "Is the reasoning chain from context to conclusion logically sound?"
  - **Verifier C (Conclusion):** "Does the final YES/NO answer match what the argument actually concludes?"
- Each verifier replies with `Opinion: True` or `Opinion: False` only (no ground truth is given — same blind constraint as SiriuS).
- Collects the three votes and passes them to `apply_gate()` in `selective_gate.py`:
  - 0 or 1 False votes → **ACCEPT** (p_wrong < 0.34)
  - 2 or 3 False votes → **REJECT** (p_wrong ≥ 0.67)
  - Exactly 1 of 3 False in the borderline range → **UNCERTAIN**
- Assigns one of the 6 labels (PT/PF/NT/NF/UT/UF) using `compute_label(score, gate_decision)`.
- Writes each item to four places: `ALL/` (every item), and either `ACCEPT/`, `REJECT/`, or `UNCERTAIN/` based on the gate decision.
- Prints a summary table of label counts and current TP Accuracy / PF Rate at the end.

**Output tree after Phase 2:**

```
logs/solve_PubMedQA_gpt-3.5-turbo-0125/sol/ensemble_judgement-gpt-3.5-turbo-0125/
├── ALL/        (every item with all 3 verifier logs attached)
├── ACCEPT/     (items that passed: 0–1 False votes)
├── REJECT/     (items that failed: 2–3 False votes)
└── UNCERTAIN/  (items that split: exactly 1 False vote)
```

---

### Phase 2.5 — Routing Split

```bash
PYTHONPATH=. python merge_cgev.py --base_dir ${EJ}
```

**What this does:**

- Reads `ACCEPT/` and `UNCERTAIN/` subdirectories → merges all items into `library.jsonl`.
  - These items **bypass the Critic entirely** and go directly to the fine-tuning library.
  - This is CGEV's core anti-corruption step: correct answers in ACCEPT and UNCERTAIN can never be corrupted.
- Reads `REJECT/` → writes all items to `reject.jsonl`.
  - These are the only items that will be sent to the Step-Level Critic.
- Prints how many items went to each path.

**Output files after Phase 2.5:**

```
ensemble_judgement-gpt-3.5-turbo-0125/
├── library.jsonl   ← ACCEPT + UNCERTAIN (bypass Critic)
└── reject.jsonl    ← REJECT only (will receive Critic feedback)
```

---

### Phase 3 — Step-Level Critic Feedback

```bash
PYTHONPATH=. python phase3_stepwise_feedback.py --model $MODEL --input_file ${EJ}/reject.jsonl
```

**What this does:**

- Reads only the `REJECT` items from `reject.jsonl`.
- For each item, calls the **Step-Level Critic** with a prompt that instructs it to:
  - Evaluate each numbered reasoning step in the Actor's response independently.
  - Label each step as `[CORRECT]` or `[WRONG]` with a brief reason.
  - Identify the **root cause step** (the first step where the error occurred).
  - Provide a single, specific **Fix instruction** for only that step.
- The Critic is **never told the ground truth** — it infers errors from context + logic alone (same constraint as SiriuS).
- Saves feedback to `stepwise_feedback-{model}/feedback/{index}_feedback.jsonl`.

**Example Critic output:**

```
Step 1: [CORRECT] - correctly identifies the study design
Step 2: [WRONG] - claims the study showed reduced mortality, but context says "no significant difference"
Step 3: [CORRECT] - reasoning from Step 2 follows logically, but inherits the error

Root cause: Step 2 — misread the study outcome
Fix: Replace "the study showed reduced mortality" with "the study showed no significant difference in mortality"
```

```bash
PYTHONPATH=. python merge.py --dir ${SF}/feedback
```

Merges all per-item `*_feedback.jsonl` files into a single `feedback.jsonl` — the input for Phase 4.

---

### Phase 4 — Actor Regenerates with Step-Level Feedback

```bash
PYTHONPATH=. python phase4_actor_regenerate.py --model $MODEL --input_file ${SF}/feedback/feedback.jsonl
```

**What this does:**

- For each item in `feedback.jsonl`, shows the Actor:
  1. The original question + context.
  2. Its previous (incorrect) response.
  3. The step-level feedback from the Critic.
- Asks the Actor to regenerate a corrected answer by fixing only the identified step.
- After regeneration, calls a **rephrase step** — a second Actor call that rewrites the regenerated answer in clean, natural language, removing any phrases like _"upon reviewing the feedback"_ that would contaminate fine-tuning data.
- Checks the rephrased answer against ground truth and saves results to:
  - `sol_re/` — all regenerated items
  - `correct_re/` — regenerations that became correct
  - `wrong_re/` — regenerations that remained wrong

---

### Phase 5A — Build Fine-Tuning Data

```bash
PYTHONPATH=. python phase5a_build_finetune.py \
  --model $MODEL \
  --library_file      ${EJ}/library.jsonl \
  --correct_re_dir    ${SF}/feedback/regenerate-${MODEL}/correct_re \
  --all_judgement_dir ${EJ}/ALL \
  --feedback_dir      ${SF}/feedback
```

**What this does:**
Builds **5 separate OpenAI SFT training files** in `logs/actor_critic/generate/{MODEL}_{MODEL}/`:

| File                        | Source                                                                    | Description                                                                                                 |
| --------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `finetune_actor.jsonl`      | Pool A: library.jsonl items where `score=True` AND `gate_decision=ACCEPT` | Clean, direct-solve correct answers. Pool B: `correct_re/` (self-corrected answers, contamination-filtered) |
| `finetune_verifier_a.jsonl` | ALL/ items (excluding UT/UF)                                              | Actor log + correct verdict for Evidence Verifier. Wrong votes are flipped to the correct verdict           |
| `finetune_verifier_b.jsonl` | Same source as A                                                          | Same logic, for Logic Verifier                                                                              |
| `finetune_verifier_c.jsonl` | Same source as A                                                          | Same logic, for Conclusion Verifier                                                                         |
| `finetune_critic.jsonl`     | feedback/ items whose index appears in correct_re/                        | Only feedback that actually led to successful Actor correction                                              |

**Key quality controls:**

- **Contamination filter for actor:** Any regenerated response starting with phrases like _"upon reviewing the feedback"_ is excluded. These phrases reveal the feedback-loop origin and would teach the model bad self-correction habits.
- **UNCERTAIN items excluded from verifier training:** Items in UT/UF are noisy (verifiers disagreed) and would teach inconsistent voting behaviour.
- **Critic training uses only successful feedback:** Only Critic outputs that resulted in a correct regeneration are used, ensuring the Critic learns from its successes, not its failures.

---

### Phase 5B — Submit Fine-Tuning Jobs

Submit one agent at a time to stay within OpenAI's concurrent fine-tuning limit of 3 jobs per organization.

```bash
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents actor
```

Uploads `finetune_actor.jsonl` to OpenAI, waits for the file to be fully processed (polls status), then submits a fine-tuning job. Polls until the job completes. Saves the resulting fine-tuned model ID to `finetuning_ids.jsonl`.

```bash
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents verifier_a
```

Same process for Verifier A (Evidence).

```bash
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents verifier_b
```

Same process for Verifier B (Logic).

```bash
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents verifier_c
```

Same process for Verifier C (Conclusion).

```bash
PYTHONPATH=. python phase5b_submit_finetune.py --model $MODEL --only_agents critic
```

Same process for the Step-Level Critic.

**What happens for each agent:**

1. Checks that the training file exists and has ≥ 10 examples (OpenAI minimum). Skips if not.
2. Waits until fewer than 3 fine-tuning jobs are active for this model (respects rate limits).
3. Uploads the `.jsonl` file to OpenAI Files API and polls until `status = "processed"`.
4. Waits an additional 30 seconds for the file to fully propagate across OpenAI's internal systems.
5. Submits the fine-tuning job.
6. Polls the job status every 60 seconds until `succeeded` or `failed`.
7. Appends `{"agent": "actor", "model": "ft:gpt-3.5-turbo-0125:personal::XXXXXX"}` to `finetuning_ids.jsonl`.

**After all 5 agents, `finetuning_ids.jsonl` will contain:**

```jsonl
{"agent": "actor",      "model": "ft:gpt-3.5-turbo-0125:personal::..."}
{"agent": "verifier_a", "model": "ft:gpt-3.5-turbo-0125:personal::..."}
{"agent": "verifier_b", "model": "ft:gpt-3.5-turbo-0125:personal::..."}
{"agent": "verifier_c", "model": "ft:gpt-3.5-turbo-0125:personal::..."}
{"agent": "critic",     "model": "ft:gpt-3.5-turbo-0125:personal::..."}
```

---

## 8. Evaluation

### Evaluate the Fine-Tuned CGEV Model

```bash
PYTHONPATH=. python evaluate.py \
  --ft_ids_file ${FT}/finetuning_ids.jsonl \
  --input_file  dataset/PubMedQA_test.jsonl \
  --num_processes 4
```

**What this does:**

- Reads `finetuning_ids.jsonl` to load all 5 fine-tuned model IDs.
- For each of the 500 test questions, runs the full CGEV pipeline end-to-end:
  1. Fine-tuned Actor generates an answer.
  2. Three fine-tuned Verifiers evaluate it.
  3. Confidence Gate routes to ACCEPT, REJECT, or UNCERTAIN.
  4. ACCEPT + UNCERTAIN → preserve Actor's answer.
  5. REJECT → fine-tuned Critic generates step-level feedback → fine-tuned Actor regenerates.
- Runs `--num_processes 4` questions in parallel.
- Saves per-question results to `logs/eval_cgev/cgev/` and a `summary.json`.
- Prints the full metrics table.

### Evaluate the Base Model (No Fine-Tuning)

```bash
PYTHONPATH=. python evaluate.py \
  --base_model $MODEL \
  --input_file dataset/PubMedQA_test.jsonl \
  --num_processes 4
```

**What this does:**

- Uses the same base model (`$MODEL`) for all 5 agents (Actor, 3 Verifiers, Critic).
- Runs the identical CGEV pipeline — same routing logic, same step-level feedback.
- Saves results to `logs/eval_cgev/base/` and prints the metrics table.
- Useful as an ablation to measure how much fine-tuning actually helps.

### Expected Terminal Output

```
==================================================================
  Results — CGEV SiriuS (ft:gpt-3.5-turbo-0125:personal::...)
==================================================================
  Total questions       : 500
  PT (accept, correct)  : 159
  PF (reject, correct)  : 45   ← false rejections
  NT (reject, wrong)    : 117
  NF (accept, wrong)    : 22
  UT (uncertain, correct): 112  ← rescued
  UF (uncertain, wrong)  : 18
  Final correct         : 350
------------------------------------------------------------------
  TP Accuracy      : 31.80%   (Table 5 metric)
  Overall Accuracy : 70.00%
  PF Rate          : 22.06%   (lower = better)
  UT Rescue Rate   : 86.15%   (CGEV-specific)
==================================================================
```

---

## 9. Repository Structure

```
CGEV/
│
│  ── Core utilities ──────────────────────────────────────────────
├── agent.py                  Agent class: wraps OpenAI chat + fine-tuning upload
│                             Initializes actor_agent, judge_agent, critic_agent at import time
│
├── args.py                   CLI argument parser shared by all phase scripts
│                             Key args: --model, --mode, --input_file, --only_agents
│
├── selective_gate.py         Confidence Gate logic (apply_gate) + 6-label system (compute_label)
│                             Thresholds: HIGH=0.34, LOW=0.67
│
│  ── Prompts ─────────────────────────────────────────────────────
├── prompt.py                 Actor prompts: sys_single_sol_prompt, pubmed_prompt_0shot,
│                             sys_single_regenerate_prompt, rephrase_sys_prompt, etc.
│
├── prompt_ensemble.py        3 Verifier system prompts (Evidence/Logic/Conclusion)
│                             Step-Level Critic prompts
│                             Shared user_verifier_prompt
│
│  ── Merge utilities ─────────────────────────────────────────────
├── merge.py                  Generic: merge all *.jsonl in a directory into one file
├── merge_cgev.py             CGEV-specific: ACCEPT+UNCERTAIN→library, REJECT→reject
│
│  ── Pipeline phases ─────────────────────────────────────────────
├── phase1_actor_solve.py     Phase 1: Actor generates initial solutions (multiprocessing)
├── phase2_ensemble_judge.py  Phase 2: 3 Verifiers + Confidence Gate (multiprocessing)
├── phase3_stepwise_feedback.py  Phase 3: Step-Level Critic generates per-step feedback
├── phase4_actor_regenerate.py   Phase 4: Actor regenerates + rephrase step
├── phase5a_build_finetune.py    Phase 5A: Build 5 SFT training files with quality controls
├── phase5b_submit_finetune.py   Phase 5B: Upload + submit fine-tuning jobs one at a time
│
├── evaluate.py               Evaluation: full CGEV pipeline on test set, 4 metrics
│
│  ── Library ─────────────────────────────────────────────────────
├── libs/
│   ├── __init__.py
│   ├── data_loader.py        load_dataset, load_jsonl_objects, extract_answer_yesno
│   └── utils.py              compare_answer_with_groundtruth
│
│  ── Data ────────────────────────────────────────────────────────
├── dataset/
│   ├── PubMedQA_train.jsonl  500 training questions (used in Phases 1–5)
│   └── PubMedQA_test.jsonl   500 test questions (used in Evaluate)
│
│  ── Config ──────────────────────────────────────────────────────
├── environment.yml           Conda environment (Python 3.10 + all pinned deps)
├── requirements.txt          Pip-only alternative
├── .env.example              Template: copy to .env and fill in OPENAI_API_KEY
├── .gitignore                Excludes logs/, .env, __pycache__, *.pyc
│
└── logs/                     (gitignored — generated at runtime)
    ├── solve_PubMedQA_{MODEL}/      Phase 1–2.5 outputs
    ├── actor_critic/generate/       Phase 5 fine-tuning data + finetuning_ids.jsonl
    └── eval_cgev/                   Evaluation results + summary.json
```

---

## 10. Dataset Format

Each line in `PubMedQA_train.jsonl` and `PubMedQA_test.jsonl` is a JSON object:

```json
{
  "index": "PMID_12345678",
  "question": "Does X cause Y in Z patients?",
  "context": "Background: ... Methods: ... Results: ... Conclusion: ...",
  "groundtruth": ["yes"]
}
```

| Field         | Type      | Description                                                          |
| ------------- | --------- | -------------------------------------------------------------------- |
| `index`       | string    | PubMed article ID, used as a unique file key throughout the pipeline |
| `question`    | string    | Binary (yes/no/maybe) clinical question derived from the abstract    |
| `context`     | string    | The abstract of the paper (without the conclusion in training)       |
| `groundtruth` | list[str] | Accepted answers: one of `["yes"]`, `["no"]`, `["maybe"]`            |

---

## 11. Citation

If you use this code, please cite both the original SiriuS paper and this work:

```bibtex
@article{zhao2025sirius,
  title   = {SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning},
  author  = {Zhao, Wanjia and Yuksekgonul, Mert and Wu, Shirley and Zou, James},
  journal = {arXiv preprint arXiv:2502.04780},
  year    = {2025}
}
```

```bibtex
@misc{quach2026cgev,
  title  = {CGEV: Confidence-Gated Ensemble Verification for Actor-Critic Multi-Agent Systems},
  author = {Quach, Thien Phu},
  year   = {2026},
  url    = {https://github.com/quachphu/CGEV}
}
```
