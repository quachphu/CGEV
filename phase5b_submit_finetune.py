"""
Phase 5B — Fine-Tuning (CGEV — 5 agents).

Extended from original SiriuS fine_tune_critic.py.

Changes vs original:
  - 5 agents: actor, verifier_a, verifier_b, verifier_c, critic  (was 3: feedback, judge, actor)
  - Minimum examples guard: skips agents with fewer than 10 training examples
  - data_model vs ft_model split (explicit separation for clarity)
  - Polls file status before submitting job (avoids OpenAI server_error race condition)

Usage (run from project root with PYTHONPATH=.):
  PYTHONPATH=. python Actor_Critic_CGEV/fine_tune_critic.py --model gpt-4.1-nano-2025-04-14
"""

import os
import json
import time
import datetime
from openai import OpenAI
import openai
from Actor_Critic_CGEV.agent import Agent
from Actor_Critic_CGEV.args import parse_args

client = OpenAI()

_args      = parse_args()
data_model = _args.model   # model used during Phases 1-4 (determines log directory key)
ft_model   = _args.model   # model to submit to OpenAI fine-tuning API

# 5 agents: actor + 3 verifiers + critic
agents = [
    Agent(name="actor",      model=ft_model, next_agent=None, pre_agent=None, api_type=client),
    Agent(name="verifier_a", model=ft_model, next_agent=None, pre_agent=None, api_type=client),
    Agent(name="verifier_b", model=ft_model, next_agent=None, pre_agent=None, api_type=client),
    Agent(name="verifier_c", model=ft_model, next_agent=None, pre_agent=None, api_type=client),
    Agent(name="critic",     model=ft_model, next_agent=None, pre_agent=None, api_type=client),
]

MIN_EXAMPLES = 10  # OpenAI fine-tuning API requires at least 10 examples
MAX_CONCURRENT = 3  # OpenAI's concurrent fine-tuning job limit per org
SLOT_POLL_INTERVAL = 60  # seconds to wait between slot-availability checks


def _count_active_jobs(model_name: str) -> int:
    """Count fine-tuning jobs that are currently running or queued for this model."""
    active = 0
    for job in client.fine_tuning.jobs.list(limit=20).data:
        if job.model == model_name and job.status in ("running", "queued", "validating_files"):
            active += 1
    return active


def _wait_for_slot(model_name: str):
    """Block until fewer than MAX_CONCURRENT jobs are active for this model."""
    while True:
        active = _count_active_jobs(model_name)
        if active < MAX_CONCURRENT:
            print(f"  Slot available ({active}/{MAX_CONCURRENT} active). Proceeding.")
            return
        print(f"  Rate limit: {active}/{MAX_CONCURRENT} jobs active for {model_name}. "
              f"Waiting {SLOT_POLL_INTERVAL}s...")
        time.sleep(SLOT_POLL_INTERVAL)


def fine_tune(data_model_name: str):
    job_ids   = {}
    dict_path = (
        f"Actor_Critic_CGEV/logs/actor_critic/generate/{data_model_name}_{data_model_name}"
    )
    os.makedirs(dict_path, exist_ok=True)
    id_file_path = f"{dict_path}/finetuning_ids.jsonl"

    for agent in agents:
        file_path = f"{dict_path}/finetune_{agent.name}.jsonl"

        # Guard: skip missing files
        if not os.path.exists(file_path):
            print(f"Skipping {agent.name}: {file_path} not found.")
            continue

        # Guard: skip files with fewer than MIN_EXAMPLES lines
        n_examples = sum(1 for line in open(file_path, encoding="utf-8") if line.strip())
        if n_examples < MIN_EXAMPLES:
            print(f"Skipping {agent.name}: only {n_examples} examples (need ≥{MIN_EXAMPLES}).")
            continue

        # Wait until a concurrent slot is free before uploading + submitting
        print(f"Checking slot availability for {agent.name}...")
        _wait_for_slot(ft_model)

        print(f"Submitting fine-tuning for {agent.name}: {file_path} ({n_examples} examples)")
        # Retry loop in case the slot check races with another submission
        for attempt in range(5):
            try:
                job = agent.fine_tune(file_path)
                break
            except openai.RateLimitError as e:
                print(f"  RateLimitError on attempt {attempt + 1}: {e}. Waiting 60s...")
                time.sleep(60)
        else:
            raise RuntimeError(f"Failed to submit fine-tuning for {agent.name} after 5 attempts.")

        job_ids[agent.name] = job.id

        with open(id_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"agent": agent.name, "id": job.id}) + "\n")

    if not job_ids:
        print("No fine-tuning jobs submitted. Check that training files exist and have ≥10 examples.")
        return

    # Poll all submitted jobs until terminal status
    terminal_statuses = {"succeeded", "failed", "cancelled"}
    while True:
        all_done = True
        for agent_name, job_id in job_ids.items():
            job = client.fine_tuning.jobs.retrieve(job_id)
            if job.status == "failed":
                print(f"\nError for {agent_name} job {job_id}:")
                print(f"  status : {job.status}")
                print(f"  error  : {job.error}")
                raise RuntimeError(
                    f"{agent_name}'s fine-tuning job {job_id} failed. Error: {job.error}"
                )
            if job.status == "cancelled":
                raise RuntimeError(
                    f"{agent_name}'s fine-tuning job {job_id} was cancelled."
                )
            if job.status not in terminal_statuses:
                print(f"  {agent_name} still in progress (status: {job.status})...")
                all_done = False
        if all_done:
            break
        time.sleep(60)

    # Record completed model IDs
    for agent in agents:
        if agent.name not in job_ids:
            print(f"Skipping {agent.name} — no job submitted.")
            continue
        job_id = job_ids[agent.name]
        job    = client.fine_tuning.jobs.retrieve(job_id)
        if job.fine_tuned_model:
            agent.model = job.fine_tuned_model
            print(f"{agent.name} fine-tuned model: {agent.model}")
            with open(id_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"agent": agent.name, "model": agent.model}) + "\n")


if __name__ == "__main__":
    if _args.only_agents:
        only = set(_args.only_agents.split(","))
        agents[:] = [a for a in agents if a.name in only]
        print(f"Submitting only: {[a.name for a in agents]}")

    start = datetime.datetime.now()
    fine_tune(data_model)
    end   = datetime.datetime.now()
    print("Time cost:", round((end - start).total_seconds() / 60, 2), "mins")
