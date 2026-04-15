"""
Phase 2.5 — CGEV Merge.

Performs two targeted merge operations after get_ensemble_judgement.py:

  Merge A (library items):  ACCEPT/ + UNCERTAIN/ → library.jsonl
  Merge B (feedback items): REJECT/              → reject.jsonl

library.jsonl  — items that bypass the Critic entirely (core anti-corruption mechanism)
reject.jsonl   — items routed to step-level Critic in Phase 3

Critical difference from original SiriuS:
  SiriuS: ALL False-judged items → Critic (includes PF = wrongly rejected correct answers)
  CGEV:   Only REJECT items      → Critic (ACCEPT + UNCERTAIN preserved, never corrupted)

Usage (run from project root with PYTHONPATH=.):
  PYTHONPATH=. python merge_cgev.py \\
    --base_dir logs/solve_PubMedQA_{model}/sol/ensemble_judgement-{model}
"""

import argparse
import json
from pathlib import Path


def load_jsonl_files_from_dir(directory: Path) -> list:
    items = []
    for f in sorted(directory.glob("*.jsonl")):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def merge_cgev(base_dir: str):
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")

    accept_dir    = base / "ACCEPT"
    uncertain_dir = base / "UNCERTAIN"
    reject_dir    = base / "REJECT"

    for d in [accept_dir, reject_dir]:
        if not d.exists():
            raise FileNotFoundError(
                f"Expected directory not found: {d}\n"
                "Run get_ensemble_judgement.py first."
            )

    # Merge A: ACCEPT + UNCERTAIN → library.jsonl
    library_items  = load_jsonl_files_from_dir(accept_dir)
    n_accept = len(library_items)

    if uncertain_dir.exists():
        uncertain_items = load_jsonl_files_from_dir(uncertain_dir)
        library_items.extend(uncertain_items)
        n_uncertain = len(uncertain_items)
    else:
        n_uncertain = 0
        print(f"[INFO] No UNCERTAIN/ directory found — skipping UNCERTAIN items.")

    library_path = base / "library.jsonl"
    with library_path.open("w", encoding="utf-8") as f:
        for item in library_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] library.jsonl: {len(library_items)} items "
          f"(ACCEPT={n_accept}, UNCERTAIN={n_uncertain}) → {library_path}")

    # Merge B: REJECT → reject.jsonl
    reject_items = load_jsonl_files_from_dir(reject_dir)
    reject_path  = base / "reject.jsonl"
    with reject_path.open("w", encoding="utf-8") as f:
        for item in reject_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] reject.jsonl:  {len(reject_items)} items → {reject_path}")

    total = len(library_items) + len(reject_items)
    print(f"\n  Total routed : {total}")
    print(f"  → Library    : {len(library_items)}  (bypass Critic)")
    print(f"  → Critic     : {len(reject_items)}  (step-level feedback)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CGEV merge: route ACCEPT+UNCERTAIN to library, REJECT to critic."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to ensemble_judgement-{model}/ directory",
    )
    args = parser.parse_args()
    merge_cgev(args.base_dir)
