"""M1 experiment runner: cell-only vs text-grounded, held-out donors.

Usage:
    python -m biocellai.experiment m1 --dataset pbmc3k --seeds 1 --outdir experiments/m1_smoke
    python -m biocellai.experiment m1 --dataset tabula_blood --seeds 3 \
        --outdir experiments/m1_grounding_benchmark

Produces in outdir: metrics.csv, manifest.json, report.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .captions import build_cell_captions, build_class_captions, marker_table
from .data import donor_split, load_pbmc3k, preprocess, query_tabula_blood
from .model import TextEncoder
from .ordinal import level_ids, level_order
from .train import (
    TrainConfig,
    _to_tensor,
    encode_labels,
    linear_probe,
    metrics,
    train_contrastive,
    train_contrastive_multihead,
    train_supervised,
    zeroshot_predict,
)


def load_dataset(name: str, seed: int, max_cells: int, n_hvg: int):
    if name == "pbmc3k":
        adata = load_pbmc3k(n_hvg=n_hvg, seed=seed)
        return adata, None
    if name == "tabula_blood":
        return query_tabula_blood(max_cells=max_cells, n_hvg=n_hvg, seed=seed)
    if name.endswith(".h5ad"):
        import anndata

        from .data import use_gene_symbols

        adata = use_gene_symbols(anndata.read_h5ad(name))
        return preprocess(adata, n_hvg=n_hvg, seed=seed), None
    raise ValueError(f"unknown dataset: {name}")


def run_m1(
    dataset: str,
    outdir: Path,
    seeds: list[int],
    max_cells: int = 50_000,
    n_hvg: int = 2000,
    k_genes: int = 8,
    tissue: str = "blood",
    epochs: int = 40,
    text_model: str = "all-MiniLM-L6-v2",
    caption_mode: str = "own_genes",
) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    losses: dict[str, list] = {}

    adata, manifest = load_dataset(dataset, seed=seeds[0], max_cells=max_cells, n_hvg=n_hvg)
    if manifest:
        manifest.write(outdir / "manifest.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tex = TextEncoder(text_model, device=device)

    for seed in seeds:
        tr_mask, te_mask = donor_split(adata, seed=seed)
        tr, te = adata[tr_mask], adata[te_mask]
        y, cats = encode_labels(adata)
        y_tr, y_te = y[tr_mask], y[te_mask]

        markers = marker_table(tr, n_markers=5)
        class_caps = build_class_captions(markers, tissue=tissue)
        class_emb = tex.embed([class_caps[c] for c in cats])

        for label_mode in (True, False):
            caps_tr = build_cell_captions(
                tr, include_label=label_mode, tissue=tissue,
                k_genes=k_genes, mode=caption_mode, markers=markers,
            )
            emb_tr = tex.embed(caps_tr)

            cfg = TrainConfig(seed=seed, epochs=epochs, device=device)
            enc_c, proj, hist = train_contrastive(tr.X, emb_tr, cfg)
            losses[f"contrastive_s{seed}_label{label_mode}"] = hist
            pred_zs = zeroshot_predict(enc_c, proj, te.X, class_emb, cfg)
            m = metrics(y_te, pred_zs)
            rows.append(dict(seed=seed, arm="grounded_zeroshot", label_mode=label_mode, **m))

            pred_lp = linear_probe(enc_c, tr.X, y_tr, te.X, cfg)
            m = metrics(y_te, pred_lp)
            rows.append(dict(seed=seed, arm="grounded_probe", label_mode=label_mode, **m))

        cfg = TrainConfig(seed=seed, epochs=epochs, device=device)
        enc_s, head, hist = train_supervised(tr.X, y_tr, cfg)
        losses[f"supervised_s{seed}"] = hist
        with torch.no_grad():
            pred_s = head(enc_s(_to_tensor(te.X).to(cfg.device))).argmax(-1).cpu().numpy()
        m = metrics(y_te, pred_s)
        rows.append(dict(seed=seed, arm="cell_only_supervised", label_mode=None, **m))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)
    (outdir / "losses.json").write_text(json.dumps(losses))
    return df


def load_input_emb(path: str | Path, adata) -> np.ndarray:
    """M12-S1: precomputed per-cell embedding matrix to replace .X as the
    contrastive encoder input (e.g. frozen scGPT features).

    Aligns by obs_names when the npz carries them; otherwise requires
    exact length + donor/cell_type array agreement with adata.
    """
    z = np.load(path, allow_pickle=True)
    emb = np.asarray(z["emb"], dtype=np.float32)
    if "obs_names" in z.files:
        names = z["obs_names"].astype(str)
        pos = pd.Index(names.astype(object)).get_indexer(
            adata.obs_names.astype(str))
        if (pos < 0).any():
            raise ValueError("input-emb obs_names do not cover dataset cells")
        emb = emb[pos]
    if emb.shape[0] != adata.n_obs:
        raise ValueError(
            f"input-emb rows {emb.shape[0]} != n_obs {adata.n_obs}")
    for key, col in (("donor", "donor_id"), ("cell_type", "cell_type")):
        if (key in z.files and "obs_names" not in z.files
                and not np.array_equal(z[key].astype(str),
                                       adata.obs[col].astype(str).to_numpy())):
            raise ValueError(f"input-emb {key} array misaligned")
    return emb


def run_m2(
    dataset: str,
    outdir: Path,
    seeds: list[int],
    caption_modes: list[str],
    text_models: list[str],
    max_cells: int = 50_000,
    n_hvg: int = 2000,
    k_genes: int = 8,
    tissue: str = "blood",
    epochs: int = 40,
    type_descriptions: dict | None = None,
    class_descriptions: dict | None = None,
    markers_override: dict | None = None,
    save_embeddings: Path | None = None,
    label_key: str = "cell_type",
    shuffle_donor_labels: bool = False,
    label_bin: int = 0,
    dual_head: str | None = None,
    pathology_descriptions: dict | None = None,
    soft_ordinal: float = 0.0,
    input_emb: str | Path | np.ndarray | None = None,
) -> pd.DataFrame:
    """Ablation matrix: caption_mode x text_model, honest arm only
    (label_mode=False) plus one leakage-probe column for reference.

    `markers_override` ({cell_type: [genes]}, e.g. PanglaoDB curated
    markers) replaces the train-derived marker table for captions AND
    class captions — the fully label-free grounding arm (M6).

    If `class_descriptions` is given ({cell_type: text}, e.g. retrieved
    literature captions), a second zero-shot eval is reported under
    arm="grounded_zeroshot_matched": class captions from the SAME register
    as the training captions (M4 lesson: register mismatch costs ~0.2 F1).
    The marker-register "grounded_zeroshot" is always reported for
    comparability with M3.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    losses: dict[str, list] = {}

    adata, manifest = load_dataset(dataset, seed=seeds[0], max_cells=max_cells, n_hvg=n_hvg)
    if manifest:
        manifest.write(outdir / "manifest.json")

    if input_emb is not None and not isinstance(input_emb, np.ndarray):
        input_emb = load_input_emb(input_emb, adata)
    elif isinstance(input_emb, np.ndarray) and input_emb.shape[0] != adata.n_obs:
        raise ValueError(
            f"input_emb rows {input_emb.shape[0]} != n_obs {adata.n_obs}")

    # resolve effective label column:
    #   "A|B"  -> composite cell-level label (e.g. Subclass x pathology)
    #   label_bin > 1 -> donor-level continuous label, qcut per seed below
    lk = label_key
    if "|" in label_key:
        c1, c2 = label_key.split("|")
        if shuffle_donor_labels:
            # negative control: permute the donor-level component only
            rng0 = np.random.default_rng(seeds[0])
            dlab = adata.obs.groupby("donor_id", observed=True)[c2].first()
            donor2lab = dict(zip(dlab.index, rng0.permutation(dlab.values)))
            adata.obs[c2] = adata.obs["donor_id"].map(donor2lab)
        lk = "__combo"
        adata.obs[lk] = (adata.obs[c1].astype(str) + " | "
                         + adata.obs[c2].astype(str))
    elif shuffle_donor_labels and label_bin <= 1 and dual_head is None:
        # negative control: permute donor -> label map; breaks real
        # cell-pathology pairing while preserving label marginals
        rng0 = np.random.default_rng(seeds[0])
        dlab = adata.obs.groupby("donor_id", observed=True)[label_key].first()
        donor2lab = dict(zip(dlab.index, rng0.permutation(dlab.values)))
        adata.obs[label_key] = adata.obs["donor_id"].map(donor2lab)

    if label_bin > 1 and "|" in label_key:
        raise ValueError("--label-bin only applies to plain donor-level keys")

    if dual_head:
        # M10: shared encoder + pathology head alongside the cell-type
        # head. Shuffle (if set) permutes the donor -> pathology map.
        if dual_head not in adata.obs.columns:
            raise ValueError(f"--dual-head column missing: {dual_head}")
        if shuffle_donor_labels:
            rng0 = np.random.default_rng(seeds[0])
            dlab = adata.obs.groupby("donor_id", observed=True)[dual_head].first()
            donor2lab = dict(zip(dlab.index, rng0.permutation(dlab.values)))
            adata.obs[dual_head] = adata.obs["donor_id"].map(donor2lab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoders = {name: TextEncoder(name, device=device) for name in text_models}
    y, cats = (None, None) if label_bin > 1 else encode_labels(adata, key=lk)

    for seed in seeds:
        tr_mask, te_mask = donor_split(adata, seed=seed)
        lk_eff = lk
        if label_bin > 1:
            # donor-level quantile bins; cutpoints from TRAIN donors only
            dv = adata.obs.groupby("donor_id", observed=True)[lk].first() \
                .astype(float)
            tr_d = pd.Index(adata.obs["donor_id"][tr_mask].unique())
            edges = np.quantile(
                dv.loc[dv.index.isin(tr_d)].dropna(),
                np.linspace(0, 1, label_bin + 1)[1:-1])
            donor2bin = {d: f"{lk}_q{b}"
                         for d, b in zip(dv.index, np.digitize(dv, edges))}
            if shuffle_donor_labels:
                rng0 = np.random.default_rng(seeds[0])
                donor2bin = dict(zip(
                    donor2bin, rng0.permutation(list(donor2bin.values()))))
            lk_eff = "__label"
            adata.obs[lk_eff] = adata.obs["donor_id"].map(donor2bin)
            y, cats = encode_labels(adata, key=lk_eff)
        tr, te = adata[tr_mask], adata[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]
        # M12-S1: contrastive arms may read a precomputed embedding matrix
        # (frozen scFM features) instead of expression; the supervised
        # reference and marker table always stay on expression.
        Xc = input_emb if input_emb is not None else None
        Xc_tr = Xc[tr_mask] if Xc is not None else tr.X
        Xc_te = Xc[te_mask] if Xc is not None else te.X
        markers = markers_override or marker_table(tr, groupby=lk_eff,
                                                   n_markers=5)
        if lk_eff != "cell_type" and type_descriptions:
            # pathology/other-key captions: class captions = same descriptions
            # (matched register by construction)
            class_caps = {c: f"A cell from human {tissue}. "
                             f"{type_descriptions.get(c, 'No description.')}"
                          for c in cats}
        else:
            class_caps = build_class_captions(markers, tissue=tissue)

        # supervised reference (caption-independent), once per seed
        cfg = TrainConfig(seed=seed, epochs=epochs, device=device)
        enc_s, head, hist = train_supervised(tr.X, y_tr, cfg)
        losses[f"supervised_s{seed}"] = hist
        with torch.no_grad():
            pred_s = head(enc_s(_to_tensor(te.X).to(cfg.device))).argmax(-1).cpu().numpy()
        rows.append(dict(seed=seed, caption_mode="-", text_model="-",
                         arm="cell_only_supervised", **metrics(y_te, pred_s)))
        if Xc is not None:
            # supervised reference on the scFM-feature input too
            enc_s2, head2, hist2 = train_supervised(Xc_tr, y_tr, cfg)
            losses[f"supervised_inputemb_s{seed}"] = hist2
            with torch.no_grad():
                pred_s2 = head2(enc_s2(_to_tensor(Xc_te).to(cfg.device))).argmax(-1).cpu().numpy()
            rows.append(dict(seed=seed, caption_mode="-", text_model="-",
                             arm="cell_only_supervised_inputemb",
                             **metrics(y_te, pred_s2)))

        for cm in caption_modes:
            caps_tr = build_cell_captions(
                tr, include_label=False, tissue=tissue,
                k_genes=k_genes, mode=cm, markers=markers,
                type_descriptions=type_descriptions, label_key=lk_eff,
            )
            for tm_name, tex in encoders.items():
                emb_tr = tex.embed(caps_tr)
                class_emb = tex.embed([class_caps[c] for c in cats])

                cfg = TrainConfig(seed=seed, epochs=epochs, device=device)
                if dual_head:
                    pdesc = pathology_descriptions or {}
                    pcaps = [f"A cell from human {tissue}. "
                             f"{pdesc.get(str(l), 'No description.')}"
                             for l in tr.obs[dual_head]]
                    emb_p = tex.embed(pcaps)
                    kw = {}
                    if soft_ordinal and soft_ordinal > 0:
                        order = level_order(
                            sorted(tr.obs[dual_head].astype(str).unique()),
                            dim=dual_head)
                        lv = torch.tensor(
                            level_ids(tr.obs[dual_head].tolist(), order))
                        bank = tex.embed(
                            [f"A cell from human {tissue}. "
                             f"{pdesc.get(l, 'No description.')}"
                             for l in order])
                        kw = dict(path_levels=lv, path_bank=bank,
                                  ord_tau=soft_ordinal)
                    enc_c, proj, proj_p, hist = train_contrastive_multihead(
                        Xc_tr, emb_tr, emb_p, cfg, **kw)
                else:
                    proj_p = None
                    kw = {}
                    if soft_ordinal and soft_ordinal > 0:
                        order = level_order(
                            sorted(tr.obs[lk_eff].astype(str).unique()),
                            dim=lk_eff)
                        if len(order) > 1:
                            lv = torch.tensor(
                                level_ids(tr.obs[lk_eff].tolist(), order))
                            bank = tex.embed([class_caps[c] for c in order])
                            kw = dict(levels=lv, bank=bank,
                                      ord_tau=soft_ordinal)
                    enc_c, proj, hist = train_contrastive(
                        Xc_tr, emb_tr, cfg, **kw)
                losses[f"contrastive_s{seed}_{cm}_{tm_name}"] = hist

                pred_zs = zeroshot_predict(enc_c, proj, Xc_te, class_emb, cfg)
                rows.append(dict(seed=seed, caption_mode=cm, text_model=tm_name,
                                 arm="grounded_zeroshot", **metrics(y_te, pred_zs)))

                if class_descriptions:
                    class_emb_m = tex.embed(
                        [class_descriptions.get(c, class_caps[c]) for c in cats])
                    pred_m = zeroshot_predict(enc_c, proj, Xc_te, class_emb_m, cfg)
                    rows.append(dict(seed=seed, caption_mode=cm, text_model=tm_name,
                                     arm="grounded_zeroshot_matched",
                                     **metrics(y_te, pred_m)))

                pred_lp = linear_probe(enc_c, Xc_tr, y_tr, Xc_te, cfg)
                rows.append(dict(seed=seed, caption_mode=cm, text_model=tm_name,
                                 arm="grounded_probe", **metrics(y_te, pred_lp)))

                if save_embeddings is not None:
                    Path(save_embeddings).mkdir(parents=True, exist_ok=True)
                    with torch.no_grad():
                        Xc_all = Xc if Xc is not None else adata.X
                        H = enc_c(_to_tensor(Xc_all).to(cfg.device))
                        emb_all = (proj_p if proj_p is not None else proj)(H) \
                            .cpu().numpy()
                        emb_type = proj(H).cpu().numpy() if proj_p is not None \
                            else None
                    npz = dict(
                        emb=emb_all,
                        donor=adata.obs["donor_id"].astype(str).to_numpy(),
                        cell_type=adata.obs["cell_type"].astype(str).to_numpy(),
                        region=adata.obs["region"].astype(str).to_numpy()
                        if "region" in adata.obs.columns else np.array([""] * adata.n_obs),
                        is_test=te_mask)
                    if emb_type is not None:
                        npz["emb_type"] = emb_type
                    np.savez_compressed(
                        save_embeddings / f"emb_s{seed}_{cm}_{tm_name.split('/')[-1]}.npz",
                        **npz)
                    torch.save(
                        dict(enc=enc_c.state_dict(), proj=proj.state_dict(),
                             proj_p=proj_p.state_dict() if proj_p is not None
                             else None,
                             var_names=list(map(str, adata.var_names)),
                             embed_dim=cfg.embed_dim, hidden=list(cfg.hidden),
                             dropout=cfg.dropout),
                        save_embeddings /
                        f"model_s{seed}_{cm}_{tm_name.split('/')[-1]}.pt")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)
    (outdir / "losses.json").write_text(json.dumps(losses))
    return df


def write_m2_report(df: pd.DataFrame, outdir: Path, dataset: str):
    outdir = Path(outdir)
    summary = (
        df.groupby(["arm", "caption_mode", "text_model"])[["macro_f1", "balanced_acc"]]
        .agg(["mean", "std"])
        .round(4)
    )
    lines = [
        "# M2 — ablation: grounding source x text encoder (held-out donors)",
        "",
        f"Dataset: `{dataset}` | seeds: {sorted(df.seed.unique())}",
        "label_mode=False throughout (honest grounding only).",
        "",
        "## Results",
        "",
        "```",
        summary.to_string(),
        "```",
        "",
    ]
    (outdir / "report.md").write_text("\n".join(lines))


def write_report(df: pd.DataFrame, outdir: Path, dataset: str):
    outdir = Path(outdir)
    summary = (
        df.groupby(["arm", "label_mode"], dropna=False)[["macro_f1", "balanced_acc"]]
        .agg(["mean", "std"])
        .round(4)
    )
    lines = [
        "# M1 — cell-only vs text-grounded (held-out donors)",
        "",
        f"Dataset: `{dataset}` | seeds: {sorted(df.seed.unique())}",
        "",
        "## Results",
        "",
        "```",
        summary.to_string(),
        "```",
        "",
        "## Reading",
        "",
        "- `grounded_zeroshot label_mode=True` = captions name the cell type",
        "  (leakage probe — upper bound, NOT a grounding result).",
        "- `label_mode=False` = marker-only captions (the honest grounding arm).",
        "- `grounded_probe` = frozen encoder + linear probe on train donors.",
        "- `cell_only_supervised` = same encoder, end-to-end supervised.",
        "",
    ]
    (outdir / "report.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["m1", "m2"])
    p.add_argument("--dataset", default="pbmc3k")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--outdir", default="experiments/m1_smoke")
    p.add_argument("--max-cells", type=int, default=50_000)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--k-genes", type=int, default=8)
    p.add_argument("--tissue", default="blood")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--text-model", default="all-MiniLM-L6-v2")
    p.add_argument("--caption-mode", default="own_genes",
                   choices=["own_genes", "type_markers", "type_llm", "all"])
    p.add_argument("--text-models", nargs="+",
                   default=["all-MiniLM-L6-v2"])
    p.add_argument("--type-descriptions", default=None,
                   help="JSON {cell_type: prose} for caption-mode type_llm")
    p.add_argument("--class-descriptions", default=None,
                   help="JSON {cell_type: text} for matched-register zero-shot eval")
    p.add_argument("--markers-override", default=None,
                   help="JSON {cell_type: [genes]} replacing train-derived markers "
                        "(e.g. PanglaoDB curated markers — label-free grounding)")
    p.add_argument("--save-embeddings", default=None,
                   help="dir for per-arm cell-embedding npz (donor/progression eval)")
    p.add_argument("--label-key", default="cell_type",
                   help="obs column used as the contrastive/eval target "
                        "(e.g. 'adnc' for donor-pathology grounding, M8)")
    p.add_argument("--shuffle-donor-labels", action="store_true",
                   help="negative control: permute donor -> label map")
    p.add_argument("--label-bin", type=int, default=0,
                   help="if >1, qcut a continuous donor-level --label-key "
                        "into N bins with train-donor cutpoints (M8 sweep)")
    p.add_argument("--dual-head", default=None,
                   help="obs column for a second (pathology) projection head "
                        "alongside the cell-type head (M10 hierarchical)")
    p.add_argument("--pathology-descriptions", default=None,
                   help="JSON {pathology_label: prose} for the --dual-head "
                        "captions (e.g. data/text/m8/pathology_adnc.json)")
    p.add_argument("--soft-ordinal", type=float, default=0.0,
                   help="M11-B2: if >0, pathology head uses distance-weighted "
                        "soft target over the ordinal level-caption bank "
                        "(tau = this value, in level units)")
    p.add_argument("--input-emb", default=None,
                   help="M12-S1: npz with a per-cell 'emb' matrix replacing "
                        "expression as the contrastive-encoder input "
                        "(e.g. frozen scGPT features). Aligns by 'obs_names' "
                        "when present, else asserts donor/cell_type match.")
    args = p.parse_args()

    type_desc = None
    if args.type_descriptions:
        type_desc = json.loads(Path(args.type_descriptions).read_text())
    class_desc = None
    if args.class_descriptions:
        class_desc = json.loads(Path(args.class_descriptions).read_text())
    markers_ovr = None
    if args.markers_override:
        markers_ovr = json.loads(Path(args.markers_override).read_text())
    path_desc = None
    if args.pathology_descriptions:
        path_desc = json.loads(Path(args.pathology_descriptions).read_text())

    cm_list = [args.caption_mode] if args.caption_mode != "all" else ["own_genes", "type_markers"]
    if args.caption_mode == "all" and type_desc:
        cm_list.append("type_llm")

    if args.experiment == "m1":
        df = run_m1(
            dataset=args.dataset,
            outdir=Path(args.outdir),
            seeds=args.seeds,
            max_cells=args.max_cells,
            n_hvg=args.n_hvg,
            k_genes=args.k_genes,
            tissue=args.tissue,
            epochs=args.epochs,
            text_model=args.text_model,
            caption_mode=args.caption_mode,
        )
        write_report(df, Path(args.outdir), args.dataset)
        print(df.groupby(["arm", "label_mode"], dropna=False)[["macro_f1", "balanced_acc"]].mean().round(4))
    elif args.experiment == "m2":
        df = run_m2(
            dataset=args.dataset,
            outdir=Path(args.outdir),
            seeds=args.seeds,
            caption_modes=cm_list,
            text_models=args.text_models,
            max_cells=args.max_cells,
            n_hvg=args.n_hvg,
            k_genes=args.k_genes,
            tissue=args.tissue,
            epochs=args.epochs,
            type_descriptions=type_desc,
            class_descriptions=class_desc,
            markers_override=markers_ovr,
            save_embeddings=Path(args.save_embeddings) if args.save_embeddings else None,
            label_key=args.label_key,
            shuffle_donor_labels=args.shuffle_donor_labels,
            label_bin=args.label_bin,
            dual_head=args.dual_head,
            pathology_descriptions=path_desc,
            soft_ordinal=args.soft_ordinal,
            input_emb=args.input_emb,
        )
        write_m2_report(df, Path(args.outdir), args.dataset)
        print(df.groupby(["arm", "caption_mode", "text_model"])[["macro_f1", "balanced_acc"]].mean().round(4))


if __name__ == "__main__":
    main()
