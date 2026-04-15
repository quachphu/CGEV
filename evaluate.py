"""
CGEV Evaluation Script — reproduces Table 5 metrics with CGEV extensions.

Runs the full CGEV pipeline on PubMedQA_test.jsonl:
  Actor → Ensemble (3 verifiers + gate) → (Critic + Regenerate if REJECT)

Reports:
  TP Accuracy     = PT / total × 100   (same definition as SiriuS Table 5)
  Overall Accuracy = final_correct / total × 100
  PF Rate         = PF / (PT+PF) × 100  (false rejection rate — CGEV minimizes)
  UT Rescue Rate  = UT / (UT+UF) × 100  (CGEV-specific: uncertain items that were correct)

Usage (run from project root with PYTHONPATH=.):

  # Evaluate fine-tuned CGEV models:
  PYTHONPATH=. python evaluate.py \\
    --ft_ids_file logs/actor_critic/generate/gpt-3.5-turbo-0125_gpt-3.5-turbo-0125/finetuning_ids.jsonl \\
    --input_file  dataset/PubMedQA_test.jsonl

  # Evaluate base model as baseline:
  PYTHONPATH=. python evaluate.py \\
    --base_model gpt-3.5-turbo-0125 \\
    --input_file dataset/PubMedQA_test.jsonl
"""

import os
import re
import json
import copy
import argparse
import datetime
import multiprocessing
import types
from tqdm import tqdm
from openai import OpenAI

from libs.data_loader import load_dataset, extract_answer_yesno
from libs.utils import compare_answer_with_groundtruth
from prompt import (
    sys_single_sol_prompt,
    pubmed_prompt_0shot,
    format_prompt_yesno,
    sys_single_regenerate_prompt,
    user_single_regenerate_prompt,
    rephrase_sys_prompt,
    rephrase_user_prompt,
)
from prompt_ensemble import (
    sys_verifier_A,
    sys_verifier_B,
    sys_verifier_C,
    user_verifier_prompt,
    stepwise_feedback_sys,
    stepwise_feedback_user,
)
from selective_gate import apply_gate, compute_label

DATA_BATCH_SIZE = 1
client = OpenAI()


# Helpers

def call_llm(model: str, sys_prompt: str, user_prompt: str) -> str | None:
    """Single LLM call with retry, returns assistant text or None."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    for attempt in range(10):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=4096
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  LLM error (attempt {attempt+1}): {e}")
    return None


def extract_verdict(text: str) -> str:
    pattern = r"(?i)(Decision|Opinion)\s*:\s*(True|False)"
    m = re.search(pattern, text)
    return m.group(2).capitalize() if m else "False"


def extract_response(text: str):
    return extract_answer_yesno(text[-15:]) if text else None


#Per-question evaluation

def evaluate_one(item, actor_model, verifier_a_model, verifier_b_model, verifier_c_model,
                 critic_model, log_dir, results):
    index       = item["index"]
    question    = item["question"]
    context     = item["context"]
    groundtruth = item["groundtruth"]
    if isinstance(groundtruth, str):
        groundtruth = [groundtruth]

    # Phase 1: Actor initial solve
    actor_user = pubmed_prompt_0shot.format(
        context=context, question=question, format_prompt=format_prompt_yesno
    )
    actor_text   = call_llm(actor_model, sys_single_sol_prompt, actor_user)
    if actor_text is None:
        return {"index": index, "label": "error", "final_correct": False}

    actor_answer  = extract_response(actor_text)
    actor_correct = compare_answer_with_groundtruth(actor_answer or "", *groundtruth)

    # Phase 2: Ensemble judgement (3 verifiers)
    verifier_user = user_verifier_prompt.format(
        context=context, question=question, original_response=actor_text
    )
    vote_A_text = call_llm(verifier_a_model, sys_verifier_A, verifier_user)
    vote_B_text = call_llm(verifier_b_model, sys_verifier_B, verifier_user)
    vote_C_text = call_llm(verifier_c_model, sys_verifier_C, verifier_user)

    vote_A = extract_verdict(vote_A_text or "")
    vote_B = extract_verdict(vote_B_text or "")
    vote_C = extract_verdict(vote_C_text or "")

    votes         = [vote_A, vote_B, vote_C]
    gate_decision = apply_gate(votes)
    label         = compute_label(actor_correct, gate_decision)

    # Phase 3+4: Critic + Regenerate (only when gate says REJECT)
    final_correct = actor_correct  # default: preserve Actor's answer

    if gate_decision == "REJECT":
        # Step-level Critic feedback
        feedback_user = stepwise_feedback_user.format(
            context=context, question=question, original_response=actor_text
        )
        feedback_text = call_llm(critic_model, stepwise_feedback_sys, feedback_user)

        if feedback_text:
            # Actor regenerates using step-level feedback
            regen_user = user_single_regenerate_prompt.format(
                question=question,
                context=context,
                original_response=actor_text,
                feedback=feedback_text,
                format_prompt=format_prompt_yesno,
            )
            regen_text = call_llm(actor_model, sys_single_regenerate_prompt, regen_user)

            if regen_text:
                # Rephrase to remove feedback traces
                rephrase_user = rephrase_user_prompt.format(
                    question=question, original_response=regen_text
                )
                rephrased  = call_llm(actor_model, rephrase_sys_prompt, rephrase_user)
                final_text = rephrased if rephrased else regen_text
                final_ans  = extract_response(final_text)
                final_correct = compare_answer_with_groundtruth(final_ans or "", *groundtruth)

    # ACCEPT + UNCERTAIN: preserve Actor's original answer (no Critic corruption)

    result = {
        "index":         index,
        "actor_correct": actor_correct,
        "votes":         {"A": vote_A, "B": vote_B, "C": vote_C},
        "gate_decision": gate_decision,
        "label":         label,
        "final_correct": final_correct,
    }

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{index}_eval.jsonl"), "w") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    results[index] = result
    return result


# Worker (multiprocessing)

def worker(batched_input_data, rank, actor_model, verifier_a_model, verifier_b_model,
           verifier_c_model, critic_model, log_dir, shared_results):
    for batch in tqdm(batched_input_data, desc=f"rank{rank}", position=rank):
        item = batch[0]
        res  = evaluate_one(
            item, actor_model, verifier_a_model, verifier_b_model,
            verifier_c_model, critic_model, log_dir, shared_results
        )
        print(f"  [{res['index']}] gate={res['gate_decision']} label={res['label']} "
              f"final={res['final_correct']}")


# Metrics

def compute_metrics(results: list, total: int) -> dict:
    counts = {k: 0 for k in ("PT", "PF", "NT", "NF", "UT", "UF")}
    for r in results:
        label = r.get("label", "")
        if label in counts:
            counts[label] += 1
    final_correct = sum(1 for r in results if r["final_correct"])

    tp_accuracy      = counts["PT"] / total * 100 if total else 0
    overall_accuracy = final_correct / total * 100 if total else 0
    pf_denominator   = counts["PT"] + counts["PF"]
    pf_rate          = counts["PF"] / pf_denominator * 100 if pf_denominator else 0
    ut_denominator   = counts["UT"] + counts["UF"]
    ut_rescue        = counts["UT"] / ut_denominator * 100 if ut_denominator else 0

    return {
        "total":            total,
        **counts,
        "final_correct":    final_correct,
        "TP_Accuracy":      round(tp_accuracy, 2),
        "Overall_Accuracy": round(overall_accuracy, 2),
        "PF_Rate":          round(pf_rate, 2),
        "UT_Rescue_Rate":   round(ut_rescue, 2),
    }


def print_table(label: str, metrics: dict):
    print("\n" + "=" * 65)
    print(f"  Results — {label}")
    print("=" * 65)
    print(f"  Total questions       : {metrics['total']}")
    print(f"  PT (accept, correct)  : {metrics['PT']}")
    print(f"  PF (reject, correct)  : {metrics['PF']}  ← false rejections")
    print(f"  NT (reject, wrong)    : {metrics['NT']}")
    print(f"  NF (accept, wrong)    : {metrics['NF']}")
    print(f"  UT (uncertain, correct): {metrics['UT']}  ← rescued")
    print(f"  UF (uncertain, wrong)  : {metrics['UF']}")
    print(f"  Final correct         : {metrics['final_correct']}")
    print("-" * 65)
    print(f"  TP Accuracy      : {metrics['TP_Accuracy']}%   (Table 5 metric)")
    print(f"  Overall Accuracy : {metrics['Overall_Accuracy']}%")
    print(f"  PF Rate          : {metrics['PF_Rate']}%  (lower = better)")
    print(f"  UT Rescue Rate   : {metrics['UT_Rescue_Rate']}%  (CGEV-specific)")
    print("=" * 65 + "\n")


# Main

def load_ft_models(ft_ids_file: str) -> dict:
    models = {}
    with open(ft_ids_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "model" in entry:
                models[entry["agent"]] = entry["model"]
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CGEV pipeline (Table 5 + CGEV metrics)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ft_ids_file", type=str,
        help="Path to finetuning_ids.jsonl (evaluate fine-tuned CGEV models)"
    )
    group.add_argument(
        "--base_model", type=str,
        help="Base model name for all agents (unfine-tuned baseline)"
    )
    parser.add_argument(
        "--input_file", type=str, default="dataset/PubMedQA_test.jsonl",
        help="Test JSONL file (default: dataset/PubMedQA_test.jsonl)"
    )
    parser.add_argument("--num_processes", type=int, default=16)
    args = parser.parse_args()

    # Resolve models
    if args.ft_ids_file:
        ft_models       = load_ft_models(args.ft_ids_file)
        actor_model     = ft_models.get("actor")
        verifier_a_model = ft_models.get("verifier_a")
        verifier_b_model = ft_models.get("verifier_b")
        verifier_c_model = ft_models.get("verifier_c")
        critic_model    = ft_models.get("critic")
        run_label       = f"CGEV SiriuS ({actor_model})"
        log_subdir      = "cgev"
    else:
        actor_model = verifier_a_model = verifier_b_model = verifier_c_model = critic_model = args.base_model
        run_label   = f"Base ({args.base_model})"
        log_subdir  = "base"

    print(f"\nActor      model : {actor_model}")
    print(f"Verifier A model : {verifier_a_model}")
    print(f"Verifier B model : {verifier_b_model}")
    print(f"Verifier C model : {verifier_c_model}")
    print(f"Critic     model : {critic_model}")

    # Load test data
    load_args = types.SimpleNamespace(
        input_file=args.input_file,
        subject="PubMedQA",
        mode="eval",
        ft_round=0, sol_round=0, fd_round=0, re_round=0,
    )
    input_data = load_dataset(load_args)
    batched    = [input_data[i:i+DATA_BATCH_SIZE] for i in range(0, len(input_data), DATA_BATCH_SIZE)]
    total      = len(batched)
    print(f"Test questions : {total}")

    log_dir = f"logs/eval_cgev/{log_subdir}"
    os.makedirs(log_dir, exist_ok=True)

    # Run in parallel
    manager        = multiprocessing.Manager()
    shared_results = manager.dict()
    n_proc         = min(args.num_processes, total)

    start = datetime.datetime.now()
    processes = []
    for i in range(n_proc):
        p = multiprocessing.Process(
            target=worker,
            args=(
                batched[i::n_proc], i,
                actor_model, verifier_a_model, verifier_b_model,
                verifier_c_model, critic_model,
                log_dir, shared_results,
            ),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    elapsed = round((datetime.datetime.now() - start).total_seconds() / 60, 2)

    # Compute and display metrics
    results_list = list(shared_results.values())
    metrics      = compute_metrics(results_list, total)
    print_table(run_label, metrics)
    print(f"Time: {elapsed} mins")

    summary_path = os.path.join(log_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"label": run_label, "metrics": metrics, "elapsed_mins": elapsed}, f, indent=2)
    print(f"Summary saved to: {summary_path}")
