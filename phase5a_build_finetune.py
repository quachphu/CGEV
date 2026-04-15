"""
Phase 5A — Build Fine-Tuning Data for CGEV (5 agents).

Builds 5 supervised training files in OpenAI SFT format ({"messages": [...]}).

  finetune_actor.jsonl
    Pool A: library.jsonl items where score=True and gate_decision='ACCEPT'
            → direct correct solves, clean signal
    Pool B: correct_re/ items where Actor self-corrected via step-level feedback
            → contamination filter applied

  finetune_verifier_a.jsonl  (and _b, _c — same logic, different logs)
    Source: ALL/ items where label is NOT 'UT' or 'UF' (skip UNCERTAIN — noisy signal)
    Positive examples: verifier voted correctly → keep log as-is
    Negative examples: verifier voted wrong    → flip assistant content to correct verdict

  finetune_critic.jsonl
    Source: feedback/ items whose index appears in correct_re/
            (Critic's step-level feedback led to a successful regeneration)

Usage (run from project root with PYTHONPATH=.):
  PYTHONPATH=. python phase5a_build_finetune.py \\
    --model gpt-3.5-turbo-0125 \\
    --library_file       .../ensemble_judgement-{model}/library.jsonl \\
    --correct_re_dir     .../stepwise_feedback-{model}/feedback/regenerate-{model}/correct_re \\
    --all_judgement_dir  .../ensemble_judgement-{model}/ALL \\
    --feedback_dir       .../stepwise_feedback-{model}/feedback

Output: logs/actor_critic/generate/{model}_{model}/
"""

import copy
import json
import os
import argparse
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _iter_per_item_jsonl(directory: str):
    """
    Yield items from per-item JSONL files only, skipping the merged output file.
    merge.py names its output {dirname}.jsonl — skip it to avoid duplicates.
    """
    dir_path    = Path(directory)
    merged_name = dir_path.name + ".jsonl"
    for jsonl_file in sorted(dir_path.glob("*.jsonl")):
        if jsonl_file.name == merged_name:
            continue
        yield from load_jsonl(str(jsonl_file))


# ── Main builder ──────────────────────────────────────────────────────────────

CONTAMINATION_PHRASES = (
    "upon reviewing the feedback",
    "upon reevaluation",
    "upon re-evaluation",
    "considering the feedback",
    "based on the feedback",
    "taking the feedback",
)


def build_finetune_data_cgev(
    library_file: str,
    correct_re_dir: str,
    all_judgement_dir: str,
    feedback_dir: str,
    output_dir: str,
):
    # Validate inputs
    for path, label in [
        (library_file,      "library_file"),
        (correct_re_dir,    "correct_re_dir"),
        (all_judgement_dir, "all_judgement_dir"),
        (feedback_dir,      "feedback_dir"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: build correct_re index (used by actor pool B and critic) ──────
    correct_re_indices = set()
    for item in _iter_per_item_jsonl(correct_re_dir):
        correct_re_indices.add(item["index"])
    print(f"  correct_re indices collected: {len(correct_re_indices)}")

    # ── File 1: finetune_actor.jsonl ──────────────────────────────────────────
    actor_path  = os.path.join(output_dir, "finetune_actor.jsonl")
    actor_count = 0

    with open(actor_path, "w", encoding="utf-8") as f_actor:
        # Pool A — ACCEPT items that were actually correct → clean direct-solve signal
        for item in load_jsonl(library_file):
            if item.get("score") is True and item.get("gate_decision") == "ACCEPT":
                f_actor.write(json.dumps(item["single_log"], ensure_ascii=False) + "\n")
                actor_count += 1

        # Pool B — correct_re items (Actor self-corrected) with contamination filter
        for item in _iter_per_item_jsonl(correct_re_dir):
            re_log         = item["re_log"]
            assistant_text = re_log["messages"][2]["content"]
            if any(p in assistant_text.lower()[:120] for p in CONTAMINATION_PHRASES):
                continue
            f_actor.write(json.dumps(re_log, ensure_ascii=False) + "\n")
            actor_count += 1

    print(f"[actor]      {actor_count} training examples → {actor_path}")

    # ── Files 2-4: finetune_verifier_{a,b,c}.jsonl ───────────────────────────
    verifier_configs = [
        ("a", "A_evidence",   "verifier_A_log"),
        ("b", "B_logic",      "verifier_B_log"),
        ("c", "C_conclusion", "verifier_C_log"),
    ]

    for name, vote_key, log_key in verifier_configs:
        verifier_path  = os.path.join(output_dir, f"finetune_verifier_{name}.jsonl")
        verifier_count = 0

        with open(verifier_path, "w", encoding="utf-8") as f_ver:
            for item in _iter_per_item_jsonl(all_judgement_dir):
                label      = item.get("label", "")
                gt_correct = item.get("score", False)

                # Skip UNCERTAIN items — verifiers disagreed, noisy training signal
                if label in ("UT", "UF"):
                    continue

                vote = item["verifier_votes"][vote_key]

                # Did this verifier vote correctly?
                # Correct vote: True if answer is right, False if answer is wrong
                verifier_correct = (vote == "True") if gt_correct else (vote == "False")

                training_log = item[log_key]

                if verifier_correct:
                    # Positive example: keep the verifier's log as-is
                    f_ver.write(json.dumps(training_log, ensure_ascii=False) + "\n")
                else:
                    # Negative example: flip assistant content to the correct verdict
                    corrected = copy.deepcopy(training_log)
                    correct_verdict = "True" if gt_correct else "False"
                    corrected["messages"][2]["content"] = f"Opinion: {correct_verdict}"
                    f_ver.write(json.dumps(corrected, ensure_ascii=False) + "\n")

                verifier_count += 1

        print(f"[verifier_{name}] {verifier_count} training examples → {verifier_path}")

    # ── File 5: finetune_critic.jsonl ─────────────────────────────────────────
    critic_path  = os.path.join(output_dir, "finetune_critic.jsonl")
    critic_count = 0

    with open(critic_path, "w", encoding="utf-8") as f_critic:
        for item in _iter_per_item_jsonl(feedback_dir):
            if item.get("index") in correct_re_indices:
                f_critic.write(
                    json.dumps(item["feedback_agent_log"], ensure_ascii=False) + "\n"
                )
                critic_count += 1

    print(f"[critic]     {critic_count} training examples → {critic_path}")
    print(f"\nAll fine-tune data written to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build 5 fine-tuning JSONL files for CGEV (actor + 3 verifiers + critic)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-nano-2025-04-14",
        help="Model name used for the output directory path.",
    )
    parser.add_argument(
        "--library_file",
        type=str,
        required=True,
        help="Path to library.jsonl (ACCEPT + UNCERTAIN items from Phase 2.5).",
    )
    parser.add_argument(
        "--correct_re_dir",
        type=str,
        required=True,
        help="Path to correct_re/ directory from Phase 4.",
    )
    parser.add_argument(
        "--all_judgement_dir",
        type=str,
        required=True,
        help="Path to ALL/ directory from Phase 2.",
    )
    parser.add_argument(
        "--feedback_dir",
        type=str,
        required=True,
        help="Path to feedback/ directory from Phase 3.",
    )
    args = parser.parse_args()

    output_dir = f"logs/actor_critic/generate/{args.model}_{args.model}"
    build_finetune_data_cgev(
        library_file=args.library_file,
        correct_re_dir=args.correct_re_dir,
        all_judgement_dir=args.all_judgement_dir,
        feedback_dir=args.feedback_dir,
        output_dir=output_dir,
    )
