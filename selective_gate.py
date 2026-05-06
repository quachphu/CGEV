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
