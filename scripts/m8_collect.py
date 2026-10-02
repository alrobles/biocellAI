"""M8 sweep collector — builds experiments/m8_sweep/index.csv.

ecoreasoner-style run index: one row per (arm, text_model, target)
joining cell-level metrics.csv with donor-level progression_metrics.csv
across all experiments/m8_* directories. Source of truth for the
pre-registered GO gate:
  GO = rho(CPS_Global) >= 0.5 AND gap vs shuffle-null >= 0.15
       AND >= 1 positive ordinal trajectory on both encoders.

Usage: python scripts/m8_collect.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = []
    for exp in sorted(ROOT.glob("experiments/m8_*")):
        arm_tag = exp.name.replace("m8_", "").replace("pathology_", "")
        prog = exp / "progression" / "progression_metrics.csv"
        if not prog.exists():
            prog = exp / "progression_metrics.csv"
        cell = exp / "metrics.csv"
        losses = exp / "losses.json"

        cell_f1 = {}
        if cell.exists():
            m = pd.read_csv(cell)
            for (arm, tm), g in m.groupby(["arm", "text_model"]):
                cell_f1[(arm, str(tm))] = g["macro_f1"].mean()

        if prog.exists():
            p = pd.read_csv(prog)
            for _, r in p.iterrows():
                tm = str(r["arm"]).split("_", 1)[-1]  # type_llm_<encoder>
                rows.append(dict(
                    sweep_arm=arm_tag,
                    arm=r["arm"], target=r["target"], rho=r["rho"],
                    n_test=r.get("n_test"),
                    cell_f1_zeroshot=cell_f1.get(("grounded_zeroshot", tm)),
                    cell_f1_probe=cell_f1.get(("grounded_probe", tm)),
                    n_losses=len(json.loads(losses.read_text()))
                    if losses.exists() else None,
                ))
        else:
            rows.append(dict(sweep_arm=arm_tag, arm=None, target=None,
                             rho=None, n_test=None, cell_f1_zeroshot=None,
                             cell_f1_probe=None, n_losses=None))

    df = pd.DataFrame(rows)
    outdir = ROOT / "experiments" / "m8_sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "index.csv", index=False)

    # pivot: one row per arm x encoder, CPS + trajectories as columns
    piv = df[df.rho.notna()].copy()
    piv["encoder"] = piv["arm"].str.split("_llm_").str[-1].str.split("@").str[0]
    piv["region"] = piv["arm"].str.extract(r"@(\w+)$")[0].fillna("all")
    tab = piv.pivot_table(index=["sweep_arm", "encoder", "region"],
                          columns="target", values="rho",
                          aggfunc="mean").round(3).reset_index()
    sd = piv.pivot_table(index=["sweep_arm", "encoder", "region"],
                         columns="target", values="rho",
                         aggfunc="std").round(3).reset_index()
    sd.columns = [f"{c}_sd" if c not in ("sweep_arm", "encoder", "region")
                  else c for c in sd.columns]
    tab = tab.merge(sd, on=["sweep_arm", "encoder", "region"])
    tab.to_csv(outdir / "index_pivot.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
