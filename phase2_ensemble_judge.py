"""
Phase 2 — Ensemble Judgement (CGEV).

Replaces get_critic_judegement.py from original SiriuS.

Runs 3 specialized verifiers (Evidence / Logic / Conclusion) on each Actor
response, then applies the confidence gate from selective_gate.py to route
items into ACCEPT / REJECT / UNCERTAIN buckets.

Usage (run from project root with PYTHONPATH=.):
  PYTHONPATH=. python Actor_Critic_CGEV/get_ensemble_judgement.py \\
    --model gpt-4.1-nano-2025-04-14 \\
    --input_file Actor_Critic_CGEV/logs/solve_PubMedQA_gpt-4.1-nano-2025-04-14/sol/sol.jsonl
"""

import os
import re
import json
import datetime
import multiprocessing
from tqdm import tqdm

from args import parse_args
from libs.data_loader import load_jsonl_objects
from Actor_Critic_CGEV.agent import Agent
from prompt_ensemble import (
    sys_verifier_A,
    sys_verifier_B,
    sys_verifier_C,
    user_verifier_prompt,
)
from selective_gate import apply_gate, compute_label
from openai import OpenAI

DATA_BATCH_SIZE = 1


def extract_verdict(text: str) -> str:
    """Extract True/False from verifier response. Defaults to False on parse failure."""
    pattern = r"(?i)(Decision|Opinion)\s*:\s*(True|False)"
    match = re.search(pattern, text)
    return match.group(2).capitalize() if match else "False"


def get_ensemble_judgement(
    batched_input_data,
    rank,
    log_dir,
    model,
    count_dict,
):
    counts = {"PT": 0, "PF": 0, "NT": 0, "NF": 0, "UT": 0, "UF": 0}

    os.makedirs(f"{log_dir}/ALL/", exist_ok=True)
    os.makedirs(f"{log_dir}/ACCEPT/", exist_ok=True)
    os.makedirs(f"{log_dir}/REJECT/", exist_ok=True)
    os.makedirs(f"{log_dir}/UNCERTAIN/", exist_ok=True)

    client = OpenAI()
    verifier_A = Agent(name="verifier_a", model=model, next_agent=None, pre_agent=None, api_type=client)
    verifier_B = Agent(name="verifier_b", model=model, next_agent=None, pre_agent=None, api_type=client)
    verifier_C = Agent(name="verifier_c", model=model, next_agent=None, pre_agent=None, api_type=client)

    for batch in tqdm(batched_input_data, desc=str(rank), position=rank):
        item = batch[0]
        index             = item["index"]
        question          = item["question"]
        context           = item["context"]
        score             = item["score"]
        original_response = item["single_log"]["messages"][2]["content"]

        all_path       = f"{log_dir}/ALL/{index}_judgement.jsonl"
        accept_path    = f"{log_dir}/ACCEPT/{index}_accept.jsonl"
        reject_path    = f"{log_dir}/REJECT/{index}_reject.jsonl"
        uncertain_path = f"{log_dir}/UNCERTAIN/{index}_uncertain.jsonl"

        if os.path.exists(all_path):
            print(f"Problem {index} already processed.")
            continue

        user_prompt = user_verifier_prompt.format(
            context=context,
            question=question,
            original_response=original_response,
        )

        log_A = verifier_A.call_agent(sys_verifier_A, user_prompt, temperature=0.0, max_tokens=1024)
        log_B = verifier_B.call_agent(sys_verifier_B, user_prompt, temperature=0.0, max_tokens=1024)
        log_C = verifier_C.call_agent(sys_verifier_C, user_prompt, temperature=0.0, max_tokens=1024)

        vote_A = extract_verdict(log_A["messages"][2]["content"])
        vote_B = extract_verdict(log_B["messages"][2]["content"])
        vote_C = extract_verdict(log_C["messages"][2]["content"])

        votes         = [vote_A, vote_B, vote_C]
        p_wrong       = votes.count("False") / len(votes)
        gate_decision = apply_gate(votes)
        label         = compute_label(score, gate_decision)

        info = {
            **item,
            "verifier_votes":  {"A_evidence": vote_A, "B_logic": vote_B, "C_conclusion": vote_C},
            "p_wrong":         p_wrong,
            "gate_decision":   gate_decision,
            "label":           label,
            "verifier_A_log":  log_A,
            "verifier_B_log":  log_B,
            "verifier_C_log":  log_C,
        }

        line = json.dumps(info, ensure_ascii=False) + "\n"

        with open(all_path, "w", encoding="utf-8") as f:
            f.write(line)

        if gate_decision == "ACCEPT":
            with open(accept_path, "w", encoding="utf-8") as f:
                f.write(line)
        elif gate_decision == "REJECT":
            with open(reject_path, "w", encoding="utf-8") as f:
                f.write(line)
        else:
            with open(uncertain_path, "w", encoding="utf-8") as f:
                f.write(line)

        counts[label] = counts.get(label, 0) + 1

    count_dict[rank] = counts


if __name__ == "__main__":
    args = parse_args()
    print(args)

    inputfile = args.input_file
    model     = args.model
    log_dir   = inputfile.replace("sol.jsonl", f"ensemble_judgement-{model}/")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output dir: {log_dir}")

    input_datas = load_jsonl_objects(inputfile)
    batched_dataset = [
        input_datas[i : i + DATA_BATCH_SIZE]
        for i in range(0, len(input_datas), DATA_BATCH_SIZE)
    ]
    print(f"Total items: {len(batched_dataset)}")

    # 32 processes × 3 verifier calls = 96 concurrent requests → hits TPM/RPM limits.
    # 8 processes × 3 = 24 concurrent requests → safely within 200K TPM / 500 RPM.
    num_processes = min(8, len(batched_dataset))
    manager       = multiprocessing.Manager()
    count_dict    = manager.dict()

    start_time = datetime.datetime.now()
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=get_ensemble_judgement,
            args=(batched_dataset[i::num_processes], i, log_dir, model, count_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    totals = {"PT": 0, "PF": 0, "NT": 0, "NF": 0, "UT": 0, "UF": 0}
    for worker_counts in count_dict.values():
        for label, n in worker_counts.items():
            totals[label] = totals.get(label, 0) + n

    total = sum(totals.values())
    print(f"\n{'='*55}")
    print(f"  Ensemble Judgement Results  (total: {total})")
    print(f"{'='*55}")
    print(f"  PT  (accept, correct)    : {totals['PT']}")
    print(f"  PF  (reject, correct)    : {totals['PF']}  ← false rejections (minimized)")
    print(f"  NT  (reject, wrong)      : {totals['NT']}")
    print(f"  NF  (accept, wrong)      : {totals['NF']}")
    print(f"  UT  (uncertain, correct) : {totals['UT']}  ← rescued correct answers")
    print(f"  UF  (uncertain, wrong)   : {totals['UF']}")
    if (totals["PT"] + totals["PF"]) > 0:
        tp_acc = totals["PT"] / total * 100
        pf_rate = totals["PF"] / (totals["PT"] + totals["PF"]) * 100
        print(f"\n  TP Accuracy : {tp_acc:.2f}%")
        print(f"  PF Rate     : {pf_rate:.2f}%  (lower is better)")
    print(f"{'='*55}")

    elapsed = round((datetime.datetime.now() - start_time).total_seconds() / 60, 2)
    print(f"Time cost: {elapsed} mins")
