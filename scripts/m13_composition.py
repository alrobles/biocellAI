#!/usr/bin/env python3
"""M13 diagnostic — how much donor-pathology signal is cell-type
composition alone?

Motivation (M12 outcome): shuffle-nulls on scGPT features retained
rho ~0.5 on CPS_Global, meaning the donor-pathology axis is largely
intrinsic to the features. The simplest intrinsic axis is composition:
AD donors lose excitatory neurons and gain reactive glia, so donor-mean
*anything* separates by cell-type fractions. This script quantifies it.

For each emb_*.npz (which carries per-cell `donor`, `cell_type`,
`is_test` on the identical donor split), builds three donor-level
feature sets and evaluates CPS_Global ridge + ordinal trajectories
exactly like m7_progression:

  composition  donor x cell-type fraction matrix (no expression)
  emb          donor-mean of the arm's embeddings (as in M7-M12)
  comp+emb     concatenation — does the embedding add over composition?

Usage:
    python scripts/m13_composition.py \
        --emb-dir experiments/m12_s2/embeddings \
        --donor-table .../seaad_donors_allregions.csv \
        --out experiments/m13_composition
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from biocellai.revalidation import reserve_output
from biocellai.progression import (
    ADNC_ORDER,
    BRAAK_ORDER,
    CERAD_ORDER,
    donor_embeddings,
    ordinal_map,
    regress_heldout,
    trajectory_correlation,
)


def composition(donor, cell_type) -> pd.DataFrame:
    """donor x cell-type fraction matrix (rows sum to 1)."""
    df = pd.DataFrame({"donor": np.asarray(donor).astype(str),
                       "ct": np.asarray(cell_type).astype(str)})
    return pd.crosstab(df["donor"], df["ct"], normalize="index")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--donor-table", required=True)
    p.add_argument("--out", default="experiments/v2_revalidation/composition_readout")
    args = p.parse_args()
    out = reserve_output(args.out)

    donors = pd.read_csv(args.donor_table, index_col=0)
    donors_g = donors[~donors.index.duplicated(keep="first")]
    ords = {"braak_trajectory": ordinal_map(donors_g["braak"], BRAAK_ORDER),
            "cerad_trajectory": ordinal_map(donors_g["cerad"], CERAD_ORDER),
            "adnc_trajectory": ordinal_map(donors_g["adnc"], ADNC_ORDER)}

    rows = []
    for f in sorted(glob.glob(str(Path(args.emb_dir) / "emb_*.npz"))):
        m = re.match(r"emb_s(\d+)_(.*?)_((?:all-)?MiniLM-L6-v2|SapBERT-.*)\.npz",
                     Path(f).name)
        if not m:
            continue
        seed, mode, tex = int(m.group(1)), m.group(2), m.group(3)
        z = np.load(f, allow_pickle=True)

        comp = composition(z["donor"], z["cell_type"])
        demb = donor_embeddings(z["emb"], z["donor"])
        demb.columns = demb.columns.astype(str)
        both = pd.concat([comp, demb], axis=1)

        te_mask = z["is_test"].astype(bool)
        te_donors = np.unique(z["donor"][te_mask])
        tr_donors = [d for d in demb.index if d not in set(te_donors)]

        for arm_name, feats in (("composition", comp),
                                ("emb", demb), ("comp+emb", both)):
            arm = f"{arm_name}__{mode}_{tex}"
            for target in ("CPS_Global", "CPS_Global_pTau",
                           "CPS_Global_ABeta"):
                if target in donors_g.columns:
                    r = regress_heldout(feats, donors_g[target],
                                        tr_donors, te_donors)
                    rows.append(dict(arm=arm, seed=seed, target=target, **r))
            for name, ordinal in ords.items():
                r = trajectory_correlation(feats, ordinal,
                                           train_donors=set(tr_donors))
                rows.append(dict(arm=arm, seed=seed, target=name, **r))
        print(f, len(comp), "donors,", comp.shape[1], "cell types")

    if not rows:
        raise ValueError("no recognized embedding files; no metrics produced")
    df = pd.DataFrame(rows)
    df["evaluation_protocol"] = "v2_readout_on_unrevalidated_inputs"
    df.to_csv(out / "composition_metrics.csv", index=False)
    piv = (df[df.target == "CPS_Global"]
           .assign(base=lambda d: d.arm.str.split("__").str[0])
           .pivot_table(index="arm", values="rho", aggfunc="mean"))
    print(piv.round(3).sort_values("rho", ascending=False).to_string())


if __name__ == "__main__":
    main()
