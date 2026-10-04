"""R7-ORCHESTRATION: DAG driver tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import v2_pipeline as vp


def _dag():
    return vp.confirmation_dag()


def test_dag_declared_before_deps():
    jobs = _dag()
    seen = set()
    for j in jobs:
        assert all(d in seen for d in j.deps), f"{j.jid} dep order"
        seen.add(j.jid)


def test_every_job_reaches_inference():
    jobs = _dag()
    infer = next(j for j in jobs if j.jid == "infer")
    upstream = set()
    frontier = list(infer.deps)
    while frontier:
        cur = frontier.pop()
        if cur in upstream:
            continue
        upstream.add(cur)
        frontier += next(j for j in jobs if j.jid == cur).deps
    producers = {j.jid for j in jobs if j.jid != "infer"}
    # every non-inference job must feed the aggregator
    assert upstream == producers


def test_afterok_wires_failure_blocking():
    jobs = _dag()
    ids = {j.jid: str(1000 + i) for i, j in enumerate(jobs)}
    for j in jobs:
        cmd = j.cmd(ids)
        if j.deps:
            dep = cmd[cmd.index("--dependency") + 1]
            assert dep.startswith("afterok:")
            assert set(dep.split(":", 1)[1].split(":")) == \
                {ids[d] for d in j.deps}
        else:
            assert "--dependency" not in cmd


def test_null_arrays_cover_all_perms():
    jobs = _dag()
    arrs = [j for j in jobs if j.jid.startswith("null_")]
    assert len(arrs) == 9  # 3 cohorts x 3 seeds
    for j in arrs:
        assert j.array == "0-99"
        assert len(j.expect) == 100


def test_validate_reports_incomplete(tmp_path, monkeypatch):
    jobs = _dag()
    monkeypatch.setattr(vp, "ROOT", tmp_path)
    assert vp.validate(jobs) == 1  # nothing exists yet
    # create the inference artifacts -> infer completes
    (tmp_path / "experiments/v2_confirmation/inference").mkdir(parents=True)
    for f in ("gates.json", "paired_comparisons.csv", "manifest.json"):
        (tmp_path / "experiments/v2_confirmation/inference" / f).touch()
    cap = next(j for j in jobs if j.jid == "infer")
    assert vp.validate([cap]) == 0
