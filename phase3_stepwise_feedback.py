"""
Phase 3 — Step-Level Feedback (CGEV).

Replaces get_critic_feedback.py from original SiriuS.

Key difference: uses step-level critic prompt instead of holistic Janusian prompt.
The step-level prompt attributes errors to specific reasoning steps, giving the
Actor actionable fix instructions instead of vague "consider alternative answers".

Input: reject.jsonl (REJECT items only — items where ≥2/3 verifiers agreed answer is wrong)
Output: stepwise_feedback-{model}/feedback/{PMID}_feedback.jsonl

NO ground truth is passed to the critic — same blind-feedback constraint as SiriuS.

Usage (run from project root with PYTHONPATH=.):
  PYTHONPATH=. python phase3_stepwise_feedback.py \\
    --model gpt-3.5-turbo-0125 \\
    --input_file logs/.../ensemble_judgement-{model}/reject.jsonl
"""

import os
import json
import datetime
import multiprocessing
from tqdm import tqdm

from args import parse_args
from libs.data_loader import load_jsonl_objects
from agent import Agent
from prompt_ensemble import stepwise_feedback_sys, stepwise_feedback_user
from openai import OpenAI

DATA_BATCH_SIZE = 1


def get_stepwise_feedback(batched_input_data, rank, log_dir, model):
    os.makedirs(f"{log_dir}/feedback", exist_ok=True)

    client  = OpenAI()
    critic  = Agent(name="critic", model=model, next_agent=None, pre_agent=None, api_type=client)

    for batch in tqdm(batched_input_data, desc=str(rank), position=rank):
        item = batch[0]
        index             = item["index"]
        question          = item["question"]
        context           = item["context"]
        original_response = item["single_log"]["messages"][2]["content"]

        feedback_path = f"{log_dir}/feedback/{index}_feedback.jsonl"
        if os.path.exists(feedback_path):
            print(f"Problem {index} feedback exists, skipping.")
            continue

        print(index)
        user_prompt = stepwise_feedback_user.format(
            context=context,
            question=question,
            original_response=original_response,
        )

        feedback_log  = critic.call_agent(
            sys_prompt=stepwise_feedback_sys,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=2048,
            stop=None,
            n=1,
        )
        feedback_text = feedback_log["messages"][2]["content"]

        info = {
            **item,
            "feedback":           feedback_text,
            "feedback_agent_log": feedback_log,
        }

        with open(feedback_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    inputfile = args.input_file
    model     = args.model
    print(f"Input : {inputfile}")
    print(f"Model : {model}")

    log_dir = inputfile.replace("reject.jsonl", f"stepwise_feedback-{model}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output: {log_dir}")

    input_datas = load_jsonl_objects(inputfile)
    batched_dataset = [
        input_datas[i : i + DATA_BATCH_SIZE]
        for i in range(0, len(input_datas), DATA_BATCH_SIZE)
    ]
    print(f"Total REJECT items: {len(batched_dataset)}")

    num_processes = min(32, len(batched_dataset))
    start_time    = datetime.datetime.now()

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=get_stepwise_feedback,
            args=(batched_dataset[i::num_processes], i, log_dir, model),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = round((datetime.datetime.now() - start_time).total_seconds() / 60, 2)
    print(f"Time cost: {elapsed} mins")
