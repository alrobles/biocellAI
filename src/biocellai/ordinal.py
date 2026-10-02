"""Ordinal level ordering for pathology dimensions (M11-B2 soft objective,
M11-C1 retrieval tiers). Single source of truth for label → position maps."""
from __future__ import annotations

import re

ORDERS: dict[str, list[str]] = {
    "adnc": ["Not AD", "Low", "Intermediate", "High"],
    "braak": ["Braak 0", "Braak II", "Braak III", "Braak IV", "Braak V",
              "Braak VI"],
    "cerad": ["Absent", "Sparse", "Moderate", "Frequent"],
    "cognitive": ["No dementia", "Dementia"],
    "cognitive_status": ["No dementia", "Dementia"],
    "thal": ["Thal 0", "Thal 1", "Thal 2", "Thal 3", "Thal 4", "Thal 5"],
    "pathology": ["nonAD", "earlyAD", "lateAD"],
    "rosmap": ["nonAD", "earlyAD", "lateAD"],
}


def level_order(labels: list[str], dim: str = "") -> list[str]:
    """Return labels sorted by pathology severity.

    Uses the explicit ORDERS map when `dim` is known; otherwise falls back
    to a trailing-number sort (e.g. CPS_Global_q0 < _q1 < _q2 < _q3).
    'nan'/unrecognised labels sort last.
    """
    valid = [l for l in labels if l.lower() != "nan"]
    if dim in ORDERS:
        known = [l for l in ORDERS[dim] if l in valid]
        return known + [l for l in valid if l not in known]

    def suf(k: str) -> int:
        m = re.search(r"(\d+)$", k)
        return int(m.group(1)) if m else 10**6

    return sorted(valid, key=suf)


def level_ids(labels, order: list[str]) -> list[int]:
    """Map each label to its ordinal position; -1 when not in `order`."""
    pos = {l: i for i, l in enumerate(order)}
    return [pos.get(str(l), -1) for l in labels]
