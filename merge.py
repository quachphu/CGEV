#!/usr/bin/env python3
"""
Merge per-item JSONL files in a directory into a single JSONL file.

Usage examples (run from project root with PYTHONPATH=.):

  # After get_a_sol.py — merge sol files for ensemble judgement input:
  python Actor_Critic_CGEV/merge.py --dir Actor_Critic_CGEV/logs/solve_PubMedQA_gpt-4.1-nano-2025-04-14/sol

  # After get_stepwise_feedback.py — merge feedback files for regenerate input:
  python Actor_Critic_CGEV/merge.py --dir Actor_Critic_CGEV/logs/solve_PubMedQA_gpt-4.1-nano-2025-04-14/sol/ensemble_judgement-gpt-4.1-nano-2025-04-14/stepwise_feedback-gpt-4.1-nano-2025-04-14/feedback

Output: <dir>/<dirname>.jsonl  (filename matches what the next script's path replace() expects)
  sol/       -> sol/sol.jsonl
  False/     -> False/False.jsonl
  feedback/  -> feedback/feedback.jsonl
"""

import argparse
from pathlib import Path


def merge_jsonl_in_dir(target_dir: str):
    dir_path = Path(target_dir)
    if not dir_path.exists():
        print(f"[WARN] Directory not found: {dir_path}")
        return

    # Output filename = directory name + .jsonl
    out_name = dir_path.name + ".jsonl"
    out_path = dir_path / out_name

    jsonl_files = sorted(
        f
        for f in dir_path.glob("*.jsonl")
        if f.name != out_name  # skip the merged output itself on re-runs
    )

    if not jsonl_files:
        print(f"[INFO] No .jsonl files found in {dir_path}")
        return

    print(f"[INFO] Found {len(jsonl_files)} files in {dir_path}")

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for f in jsonl_files:
            with f.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        written += 1

    print(f"[DONE] Merged {written} records from {len(jsonl_files)} files → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge per-item JSONL files in a directory into a single JSONL file."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the directory containing per-item .jsonl files to merge",
    )
    args = parser.parse_args()
    merge_jsonl_in_dir(args.dir)
