"""R4-BENCHMARK tables: external-scFM results vs the v2 arm matrix.

Walks experiments/v2_revalidation/<coh>/run_s<seed>_<model>_<input>/
metrics.csv (v2run schema) and joins <coh>/scfm/<model>_<input>_s<seed>
.manifest.json for parameter count, wall time, gene coverage, and
dropped cells. Emits:

  scfm_long.csv            one row per (cohort, seed, model, input)
  scfm_summary_native.csv  cohort x model, mean+-sd over seeds
  scfm_summary_hvg.csv     same for the HVG-2000 ablation
  scfm_models.csv          per-model info (params, dim, wall, coverage)
  scfm_table.json          manifest

    python scripts/v2_scfm_table.py \
        --base experiments/v2_revalidation --out <base>/aggregate_scfm
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RUN_RE = re.compile(r"run_s(\d+)_(gf_v1_10m|gf_v2_104m|gf_v2_316m"
                    r"|scgpt_wh|cw_clip_v1)_(native|hvg)$")


def collect(base: Path):
    rows, infos = [], []
    for csv in sorted(base.glob("*/run_s*_*/metrics.csv")):
        m = RUN_RE.fullmatch(csv.parent.name)
        if not m:
            continue
        seed, model, inp = int(m.group(1)), m.group(2), m.group(3)
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            row = {"cohort": csv.parent.parent.name, "seed": seed,
                   "model": model, "input": inp}
            row.update(r.to_dict())
            rows.append(row)
        man = csv.parent.parent / "scfm" / f"{model}_{inp}_s{seed}.manifest.json"
        if man.exists():
            j = json.loads(man.read_text())
            infos.append({
                "cohort": csv.parent.parent.name, "seed": seed,
                "model": model, "input": inp,
                "n_cells": j.get("n_cells"),
                "n_cells_embedded": j.get("n_cells_embedded"),
                "n_genes_input": j.get("n_genes_input"),
                "emb_dim": j.get("emb_dim"),
                "wall_s": (j.get("model_info") or {}).get("wall_s"),
                "params": (j.get("model_info") or {}).get("params"),
                "hidden": (j.get("model_info") or {}).get("hidden"),
                "vocab_genes": (j.get("model_info") or {}).get("vocab_genes"),
                "n_dropped_empty": (j.get("model_info") or {})
                .get("n_dropped_empty"),
                "gene_id_source": (j.get("model_info") or {})
                .get("gene_id_source"),
                "id_map_hit_rate": (j.get("model_info") or {})
                .get("id_map_hit_rate"),
                "patches": "; ".join((j.get("model_info") or {})
                                     .get("patches", [])),
            })
    if not rows:
        raise SystemExit(f"no scfm run dirs under {base}")
    return pd.DataFrame(rows), pd.DataFrame(infos)


def summarize(df, metric):
    g = (df.pivot_table(index=["cohort", "model"], columns="seed",
                        values=metric, aggfunc="first"))
    g["mean"], g["sd"] = g.mean(axis=1), g.std(axis=1)
    return g.reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path,
                   default=Path("experiments/v2_revalidation"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    df, info = collect(args.base)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "scfm_long.csv", index=False)
    info.to_csv(args.out / "scfm_models.csv", index=False)

    for inp in ("native", "hvg"):
        sub = df[df.input == inp]
        if sub.empty:
            continue
        for metric in ("path_rho", "identity_f1_probe"):
            if sub[metric].isna().all():
                continue
            g = summarize(sub, metric)
            g.to_csv(args.out / f"scfm_summary_{inp}_{metric}.csv",
                     index=False)

    (args.out / "scfm_table.json").write_text(json.dumps({
        "step": "v2_scfm_table",
        "cohorts": sorted(df.cohort.unique()),
        "models": sorted(df.model.unique()),
        "inputs": sorted(df.input.unique()),
        "seeds": sorted(int(s) for s in df.seed.unique()),
        "n_rows": len(df),
    }, indent=2))
    print(f"wrote {args.out}/scfm_long.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
