#!/usr/bin/env python3
"""Generate paper figures from consolidated metrics (M5).

Reads experiments/*/metrics.csv + losses.json and writes paper/figures/*.pdf.
Deterministic: no training, just plotting.

Usage:
    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M3 = ROOT / "experiments/m3_ablation_85k"
M4H = ROOT / "experiments/m4_llm_honest_85k"
M4L = ROOT / "experiments/m4_llm_labeled_85k"
FIGS = ROOT / "paper/figures"

SHORT = {"all-MiniLM-L6-v2": "MiniLM",
         "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "SapBERT"}
MODE_LBL = {"own_genes": "per-cell genes", "type_markers": "type markers",
            "type_llm": "LLM prose (honest)"}
COLORS = {"MiniLM": "#4C9BD6", "SapBERT": "#E8833A"}


def load(d: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(d / "metrics.csv")
    df["exp"] = label
    for c in ("caption_mode", "text_model"):
        if c not in df:
            df[c] = "-"
    return df


def fig1(df: pd.DataFrame):
    """Zero-shot macro-F1 by grounding source; supervised baseline as ref line."""
    zs = df[df.arm == "grounded_zeroshot"].copy()
    sup = df[df.arm == "cell_only_supervised"]
    sup_f1, sup_ba = sup.macro_f1.mean(), sup.balanced_acc.mean()

    order = ["per-cell genes", "type markers", "LLM prose (honest)",
             "LLM prose (leak)"]
    agg = (zs.groupby(["mode_lbl", "tm"], observed=True)
           .agg(m=("macro_f1", "mean"), s=("macro_f1", "std")).reset_index())

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(order))
    w = 0.36
    for i, tm in enumerate(["MiniLM", "SapBERT"]):
        sub = agg[agg.tm == tm].set_index("mode_lbl").reindex(order)
        ax.bar(x + (i - 0.5) * w, sub.m, w, yerr=sub.s, capsize=3,
               color=COLORS[tm], label=tm, edgecolor="white", linewidth=0.4)
        # hatch the leakage-probe group
        ax.bar(x[-1] + (i - 0.5) * w, sub.m.iloc[-1], w,
               facecolor="none", edgecolor=COLORS[tm], hatch="//",
               linewidth=0.6)
    ax.axhline(sup_f1, color="k", ls="--", lw=1.1,
               label=f"supervised ({sup_f1:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("macro-F1 (zero-shot, held-out donors)")
    ax.set_ylim(0, 0.72)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(3, 0.02, "leakage\nprobe", ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_grounding_source.pdf")


def fig2(df: pd.DataFrame):
    """Probe vs zero-shot gap: how much of the aligned structure is reachable
    without labels. Also contrastive loss curves (inset-style second panel)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))

    # left: loss curves (MiniLM arms, seed 0)
    import json
    losses = {k: v for k, v in
              json.loads((M3 / "losses.json").read_text()).items()
              if "s0" in k and "MiniLM" in k}
    for k, v in losses.items():
        lbl = k.replace("contrastive_s0_", "").replace("_all-MiniLM-L6-v2", "")
        a1.plot(v, lw=1.2, label=lbl.replace("own_genes", "per-cell genes")
                .replace("type_markers", "type markers"))
    a1.set_xlabel("epoch"); a1.set_ylabel("InfoNCE loss")
    a1.legend(frameon=False, fontsize=7)
    a1.spines[["top", "right"]].set_visible(False)
    a1.set_title("contrastive training (seed 0)", fontsize=9)

    # right: probe - zeroshot gap per condition (how much label info adds)
    g = (df[df.arm.isin(["grounded_zeroshot", "grounded_probe"])]
         .groupby(["mode_lbl", "tm", "arm"]).macro_f1.mean().unstack("arm"))
    g["gap"] = g.grounded_probe - g.grounded_zeroshot
    g = g.reset_index()
    order = ["per-cell genes", "type markers", "LLM prose (honest)",
             "LLM prose (leak)"]
    x = np.arange(len(order))
    for i, tm in enumerate(["MiniLM", "SapBERT"]):
        sub = g[g.tm == tm].set_index("mode_lbl").reindex(order)
        a2.bar(x + (i - 0.5) * 0.36, sub.gap, 0.36, color=COLORS[tm])
    a2.set_xticks(x)
    a2.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    a2.set_ylabel("probe F1 $-$ zero-shot F1")
    a2.spines[["top", "right"]].set_visible(False)
    a2.set_title("label information still needed", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_loss_gap.pdf")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})

    df = pd.concat([load(M3, "m3"), load(M4H, "m4h"), load(M4L, "m4l")])
    df["tm"] = df.text_model.map(SHORT).fillna("-")
    df["mode_lbl"] = df.caption_mode.map(MODE_LBL)
    df.loc[df.exp.eq("m4l") & df.arm.ne("cell_only_supervised"),
           "mode_lbl"] = "LLM prose (leak)"

    fig1(df)
    fig2(df)
    print("wrote", *sorted(p.name for p in FIGS.glob("*.pdf")))


if __name__ == "__main__":
    main()
