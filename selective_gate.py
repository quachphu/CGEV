"""
CGEV Confidence Gate — routes Actor answers to ACCEPT / REJECT / UNCERTAIN.

Thresholds (fraction of "False" votes out of 3 verifiers):
  p_wrong < 0.34  → ACCEPT    (0 or 1 False vote: 0/3=0.00, 1/3=0.33)
  p_wrong ≥ 0.67  → REJECT    (2 or 3 False votes: 2/3=0.67, 3/3=1.00)
  else            → UNCERTAIN  (exactly the split case: 1/3 when threshold is strict)

6-label system (extends SiriuS PT/PF/NT/NF with UT/UF):
  PT — ACCEPT + score True   True Positive: verifiers correctly passed correct answer
  PF — REJECT + score True   False Rejection: verifiers wrongly rejected correct answer (minimized by CGEV)
  NT — REJECT + score False  True Negative: verifiers correctly rejected wrong answer
  NF — ACCEPT + score False  False Acceptance: verifiers missed an error
  UT — UNCERTAIN + score True  Rescued: correct answer preserved instead of sent to Critic
  UF — UNCERTAIN + score False Preserved: wrong answer bypassed Critic (cost of UNCERTAIN)
"""

HIGH_THRESHOLD = 0.34   # p_wrong < HIGH  → ACCEPT
LOW_THRESHOLD  = 0.67   # p_wrong ≥ LOW   → REJECT


def apply_gate(votes: list[str]) -> str:
    """
    votes: list of 'True'/'False' strings (one per verifier).
    Returns: 'ACCEPT', 'REJECT', or 'UNCERTAIN'.
    """
    if not votes:
        return "REJECT"
    p_wrong = votes.count("False") / len(votes)
    if p_wrong < HIGH_THRESHOLD:
        return "ACCEPT"
    elif p_wrong >= LOW_THRESHOLD:
        return "REJECT"
    else:
        return "UNCERTAIN"


def compute_label(score: bool, gate_decision: str) -> str:
    """
    score:         True if Actor's answer matched ground truth (Phase 1).
    gate_decision: output of apply_gate().
    Returns one of: PT, PF, NT, NF, UT, UF.
    """
    if score and gate_decision == "ACCEPT":
        return "PT"
    elif score and gate_decision == "REJECT":
        return "PF"
    elif not score and gate_decision == "REJECT":
        return "NT"
    elif not score and gate_decision == "ACCEPT":
        return "NF"
    elif score and gate_decision == "UNCERTAIN":
        return "UT"
    else:
        return "UF"
