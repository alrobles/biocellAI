from __future__ import annotations

import argparse
import json
from pathlib import Path

from biocellai.revalidation import plan_status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path,
                   default=Path(__file__).resolve().parents[1] / "spec/revalidation_tasks.json")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    statuses = plan_status(plan)
    if args.json:
        print(json.dumps(statuses, indent=2))
        return
    for task in plan["tasks"]:
        state = statuses[task["id"]]
        print(f"{task['id']:<20} {state:<12} {task['title']}")
        if task.get("blocker"):
            print(f"  blocker: {task['blocker']}")
        if state == "blocked" and task.get("depends_on"):
            deps = [d for d in task["depends_on"] if statuses[d] != "completed"]
            if deps:
                print(f"  needs: {', '.join(deps)}")


if __name__ == "__main__":
    main()
