from __future__ import annotations

import hashlib
import math
from decimal import Decimal
from pathlib import Path


def improvement_gate(candidate, reference, paired_ci_low, integrity_ok, null_ok,
                     threshold=0.05):
    values = (candidate, reference, paired_ci_low, threshold)
    if integrity_ok is not True or null_ok is None:
        return "NOT_EVALUABLE"
    if any(v is None or not math.isfinite(v) for v in values):
        return "NOT_EVALUABLE"
    delta = Decimal(str(candidate)) - Decimal(str(reference))
    passed = delta >= Decimal(str(threshold)) and paired_ci_low > 0 and null_ok is True
    return "PASS" if passed else "FAIL"


def validate_plan(plan):
    tasks = plan["tasks"]
    by_id = {t["id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate task IDs")
    visited, visiting = set(), set()

    def visit(task_id):
        if task_id not in by_id:
            raise ValueError(f"unknown dependency {task_id}")
        if task_id in visiting:
            raise ValueError(f"dependency cycle at {task_id}")
        if task_id in visited:
            return
        task = by_id[task_id]
        if task["status"] not in {"pending", "in_progress", "completed", "blocked"}:
            raise ValueError(f"invalid status for {task_id}")
        if task["status"] == "completed" and not task.get("evidence"):
            raise ValueError(f"completed task {task_id} needs evidence")
        visiting.add(task_id)
        for dep in task.get("depends_on", []):
            visit(dep)
            if task["status"] == "completed" and by_id[dep]["status"] != "completed":
                raise ValueError(f"completed task {task_id} has unfinished dependency {dep}")
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)


def plan_status(plan):
    validate_plan(plan)
    by_id = {t["id"]: t for t in plan["tasks"]}
    statuses = {}
    for task_id, task in by_id.items():
        state = task["status"]
        if state == "pending":
            ready = all(by_id[d]["status"] == "completed" for d in task.get("depends_on", []))
            state = "ready" if ready else "blocked"
        statuses[task_id] = state
    return statuses


def reserve_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_ids_sha256(ids):
    digest = hashlib.sha256()
    for value in ids:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
