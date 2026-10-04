#!/usr/bin/env python3
"""R8-PAPER: generate paper figures and LaTeX tables from v2 metrics.

All numbers come from committed artifacts — nothing is hand-copied.
Outputs land in paper/figures/ (PDFs + .tex includes).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP, CONF = ROOT / "experiments/v2_revalidation", ROOT / "experiments/v2_confirmation"
OUT = ROOT / "paper/figures"
ARMS = ["B0", "B1", "B2", "T0", "T1", "T2", "T3", "N1", "N2"]
ARM_STYLE = {"B0": "#888888", "B1": "#1f77b4", "B2": "#d62728",
             "T0": "#c7c7c7", "T1": "#9467bd", "T2": "#2ca02c",
             "T3": "#17becf", "N1": "#8c564b", "N2": "#e377c2"}


def run_metrics() -> pd.DataFrame:
    rows = []
    for tree, phase in ((EXP, "exploratory"), (CONF, "confirmatory")):
        for f in glob.glob(str(tree / "*/run_s*_full/metrics.csv")):
            df = pd.read_csv(f)
            df["cohort"] = Path(f).parent.parent.name
            df["seed"] = int(Path(f).parent.name.split("run_s")[1][0])
            df["phase"] = phase
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def transfer_metrics() -> pd.DataFrame:
    rows = []
    for tree, phase in ((EXP, "exploratory"), (CONF, "confirmatory")):
        for f in glob.glob(str(tree / "transfer/*/s*/metrics.csv")):
            df = pd.read_csv(f)
            df["direction"] = Path(f).parent.parent.name
            df["seed"] = Path(f).parent.name
            df["phase"] = phase
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def perm_nulls() -> pd.DataFrame:
    rows = []
    for tree, phase in ((EXP, "exploratory"), (CONF, "confirmatory")):
        for f in glob.glob(str(tree / "*/perm_null_s*/perm_*.csv")):
            df = pd.read_csv(f)
            df["cohort"] = Path(f).parent.parent.name
            df["phase"] = phase
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- figures

def fig_arms_rho(m: pd.DataFrame) -> None:
    """Per-arm donor-pathology rho, exploratory vs confirmatory panels."""
    # blood has no donor-pathology endpoint (identity task only)
    cohorts = [c for c in ["seaad_mtg", "rosmap", "seaad_pfc"]
               if c in set(m.cohort)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, phase in zip(axes, ("exploratory", "confirmatory")):
        sub = m[m.phase == phase]
        x = np.arange(len(cohorts))
        w = 0.085
        for i, arm in enumerate(ARMS):
            d = sub[sub.arm == arm]
            mean = d.groupby("cohort").path_rho.mean().reindex(cohorts)
            ax.bar(x + (i - 4) * w, mean, w, color=ARM_STYLE[arm],
                   label=arm, zorder=2)
            for j, co in enumerate(cohorts):
                pts = d[d.cohort == co].path_rho
                ax.scatter(np.full(len(pts), j + (i - 4) * w), pts,
                           s=5, color="k", alpha=.55, zorder=3)
        ax.axhline(0, color="k", lw=.6)
        ax.set_xticks(x); ax.set_xticklabels(
            [c.replace("seaad_", "SEA-AD ") for c in cohorts])
        ax.set_title(f"{phase} (seeds "
                     f"{'0-2' if phase == 'exploratory' else '3-5 / PFC 0-2'})")
        ax.set_ylabel(r"donor-level pathology $\rho$" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=.3, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=9, loc="lower center",
               frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, .08, 1, 1])
    fig.savefig(OUT / "fig_arms_rho.pdf")
    plt.close(fig)


def fig_null(m: pd.DataFrame, nulls: pd.DataFrame) -> None:
    """Shared perm-null distribution vs observed rho, per cohort.

    Two null designs: arms other than B2 use readout-permutation nulls
    (frozen features, permuted ridge target); B2's violin is the
    endpoint-supervision null (encoder retrained on permuted labels,
    true-label readout). The extra 'B2-ro' violin is the revision check:
    readout-perm on the REAL B2 checkpoint's frozen features.
    """
    nulls = nulls[nulls.phase == "confirmatory"]
    obs = (m[(m.phase == "confirmatory")]
           .groupby(["cohort", "arm"]).path_rho.mean().reset_index())
    ro = []
    for f in glob.glob(str(CONF / "*/b2_readout_null_s*/b2_readout_perm.csv")):
        d = pd.read_csv(f)
        d["cohort"] = Path(f).parent.parent.name
        ro.append(d)
    ro = pd.concat(ro) if ro else None
    cohorts = sorted(nulls.cohort.unique())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(13, 4.2), sharey=True)
    for ax, co in zip(axes, cohorts):
        n = nulls[nulls.cohort == co]
        arms = [a for a in ARMS if a != "N1"]
        data = [n[n.arm == a].rho for a in arms]
        labels = list(arms)
        if ro is not None and co in set(ro.cohort):
            b2i = labels.index("B2") + 1
            labels.insert(b2i, "B2-ro")
            data.insert(b2i, ro[ro.cohort == co].rho)
        vp = ax.violinplot(data, showextrema=False, widths=.85)
        for b, a in zip(vp["bodies"], labels):
            b.set_facecolor(ARM_STYLE.get(a, "#f0a0a0")); b.set_alpha(.55)
        q95 = [np.quantile(d, .95) for d in data]
        xs = range(1, len(labels) + 1)
        ax.scatter(xs, q95, marker="_", s=140,
                   color="k", label="null q95")
        o = obs[obs.cohort == co].set_index("arm").path_rho
        ax.scatter(xs, [o.get(a, np.nan) for a in labels], marker="D",
                   s=28, color=[ARM_STYLE.get(a, "w") for a in labels],
                   edgecolor="k", zorder=5, label="observed")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(co.replace("seaad_", "SEA-AD "))
        ax.axhline(0, color="k", lw=.6); ax.grid(axis="y", alpha=.3, zorder=0)
    axes[0].set_ylabel(r"donor-level pathology $\rho$")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_null.pdf")
    plt.close(fig)


def fig_transfer(t: pd.DataFrame) -> None:
    """Strict vs calibrated-refit transfer rho across directions/seeds."""
    strict = t[(t["mode"] == "strict") & (t.eval_subset == "all_donors")
               & (t.arm == "B2")]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    dirs = sorted(strict.direction.unique())
    x = np.arange(len(dirs)); w = 0.35
    for i, phase in enumerate(("exploratory", "confirmatory")):
        d = strict[strict.phase == phase]
        mean = d.groupby("direction").rho.mean().reindex(dirs)
        ax.bar(x + (i - .5) * w, mean, w, label=phase, zorder=2,
               color=["#1f77b4", "#d62728"][i], alpha=.85)
        for j, dr in enumerate(dirs):
            pts = d[d.direction == dr].rho
            ax.scatter(np.full(len(pts), j + (i - .5) * w), pts, s=12,
                       color="k", zorder=3)
    ax.axhline(.35, color="k", ls="--", lw=1)
    ax.text(len(dirs) - .05, .36, "gate $\\rho \\geq 0.35$", ha="right",
            fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_to_", r" $\rightarrow$ ")
                        .replace("seaad_", "SEA-AD ")
                        .replace("rosmap", "ROSMAP") for d in dirs],
                       fontsize=8)
    ax.set_ylabel(r"strict source-fitted $\rho$ (all target donors)")
    ax.grid(axis="y", alpha=.3, zorder=0); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_transfer.pdf"); plt.close(fig)


def fig_paired() -> None:
    """Forest plot of primary paired comparisons from both phases."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=False)
    for ax, (tree, phase) in zip(axes, ((EXP, "exploratory"),
                                         (CONF, "confirmatory"))):
        pc = pd.read_csv(tree / "inference/paired_comparisons.csv")
        keep = {"T2-B2", "T3-B2", "T2-T1", "T3-T1"}
        pc = pc[pc.comparison.isin(keep)]
        rows, labels, colors = [], [], []
        for _, r in pc.iterrows():
            rows.append(r); labels.append(
                f"{r.cohort.replace('seaad_','SEA-AD ')}: "
                f"{r.candidate}$-${r.reference}")
            colors.append(ARM_STYLE.get(r.candidate, "k"))
        rows = rows[::-1]; labels = labels[::-1]; colors = colors[::-1]
        y = np.arange(len(rows))
        for yi, r, c in zip(y, rows, colors):
            ax.errorbar(r.delta_rho, yi,
                        xerr=[[r.delta_rho - r.ci_low],
                              [r.ci_high - r.delta_rho]],
                        fmt="o", color=c, capsize=2, ms=4)
        ax.axvline(0, color="k", lw=.7); ax.axvline(-.05, color="k",
                                                  ls=":", lw=.7)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(phase); ax.set_xlabel(r"paired $\Delta\rho$ (95% CI)")
        ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_paired.pdf"); plt.close(fig)


# ----------------------------------------------------------------- tables

COHORT_LABEL = {"seaad_mtg": "SEA-AD MTG", "seaad_pfc": "SEA-AD PFC",
                "rosmap": "ROSMAP", "blood": "blood"}


def tex_tables(m: pd.DataFrame, t: pd.DataFrame) -> None:
    """Emit LaTeX includes: arm means table + strict transfer matrix."""
    piv = (m[m.phase == "confirmatory"]
           .pivot_table(index="arm", columns="cohort",
                        values="path_rho", aggfunc="mean")
           .reindex(ARMS))
    piv.columns = [COHORT_LABEL[c] for c in piv.columns]
    piv.index.name = "arm"
    piv.to_latex(OUT / "tab_rho_confirmatory.tex", float_format="%.3f")

    strict = t[(t["mode"] == "strict") & (t.eval_subset == "all_donors")]
    tt = (strict.pivot_table(index=["direction", "arm"], columns="phase",
                             values="rho", aggfunc="mean"))
    tt.index = tt.index.set_levels(
        [d.replace("_to_", " $\\to$ ").replace("seaad_", "SEA-AD ")
            .replace("rosmap", "ROSMAP").replace("mtg", "MTG")
            .replace("pfc", "PFC")
         for d in tt.index.levels[0]], level=0)
    tt.to_latex(OUT / "tab_transfer.tex", float_format="%.3f",
                escape=False, na_rep="---")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = run_metrics()
    t = transfer_metrics()
    fig_arms_rho(m)
    fig_null(m, perm_nulls())
    fig_transfer(t)
    fig_paired()
    tex_tables(m, t)
    print(f"wrote figures+tables -> {OUT}")


if __name__ == "__main__":
    main()
