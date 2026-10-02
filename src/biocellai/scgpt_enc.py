"""M12-S2 — scGPT whole-human backbone for the dual-head contrastive
objective (ADR-009).

The checkpoint is rebuilt exactly as ``scgpt.tasks.cell_emb.embed_data``
constructs it (whole-human release: 12L/512d/8H, binned input, 51 bins,
``pad_value=-2``), then fine-tuned — fully or last-N encoder blocks —
with the same InfoNCE heads as the MLP pipeline. Tokenisation reuses
the official ``DataCollator`` (do_binning, max_length sampling,
keep_first_n_tokens=1) so the input distribution matches G2 exactly.

scgpt is an optional dependency — imported lazily inside functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .train import info_nce, soft_ordinal_nce


def load_scgpt(model_dir, device: str, use_fast_transformer: bool = False):
    """Rebuild the whole-human checkpoint as embed_data does.

    Returns (model, vocab, model_configs). use_fast_transformer=False
    selects the standard nn.TransformerEncoder path; load_pretrained
    remaps the checkpoint's Wqkv weights onto in_proj_* accordingly.
    """
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    model_dir = Path(model_dir)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in ("<pad>", "<cls>", "<eoc>"):
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    cfg = json.loads((model_dir / "args.json").read_text())

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        nlayers_cls=cfg["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=cfg["dropout"],
        pad_token=cfg["pad_token"],
        pad_value=cfg["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    load_pretrained(
        model,
        torch.load(model_dir / "best_model.pt", map_location=device),
        verbose=False,
    )
    return model.to(device), vocab, cfg


def make_collator(vocab, cfg, max_length: int = 1200):
    """Official tokeniser settings, identical to get_batch_cell_embeddings."""
    from scgpt.data_collator import DataCollator

    return DataCollator(
        do_padding=True,
        pad_token_id=vocab[cfg["pad_token"]],
        pad_value=cfg["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )


def collate_keep_id(collator):
    """Wrap the official DataCollator so batch["id"] survives — the stock
    collator drops it, but the train loop needs it to index captions."""
    def _fn(examples):
        out = collator(examples)
        out["id"] = torch.tensor([e["id"] for e in examples], dtype=torch.long)
        return out
    return _fn


class ScgptCellSet(torch.utils.data.Dataset):
    """Per-cell nonzero genes + values with <cls> prepended.

    Mirrors the inner Dataset of get_batch_cell_embeddings. `index`
    optionally maps local positions to rows of X (train subsets).
    """

    def __init__(self, X, gene_ids: np.ndarray, cls_id: int, pad_value: float,
                 index: np.ndarray | None = None):
        Xs = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        self.X = np.asarray(Xs, dtype=np.float32)
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.cls_id = int(cls_id)
        self.pad_value = float(pad_value)
        self.index = np.arange(self.X.shape[0]) if index is None else np.asarray(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.X[self.index[i]]
        nz = np.nonzero(row)[0]
        genes = np.insert(self.gene_ids[nz], 0, self.cls_id)
        values = np.insert(row[nz], 0, self.pad_value)
        return {
            "id": i,
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
        }


def gene_ids_in_vocab(adata, vocab, gene_col: str = "gene_name") -> np.ndarray:
    """Vocab id per var column (-1 when missing), like embed_data."""
    if gene_col in adata.var.columns:
        col = adata.var[gene_col].astype(str)
    else:
        col = adata.var_names.astype(str)
    return np.array([vocab[g] if g in vocab else -1 for g in col],
                    dtype=int)


def freeze_all_but_last_blocks(model, n_blocks: int) -> int:
    """PEFT arm (S2b): freeze the whole backbone except the last n
    transformer encoder layers. Returns the trainable-param count."""
    for p in model.parameters():
        p.requires_grad_(False)
    enc = getattr(model, "transformer_encoder", None)
    layers = getattr(enc, "layers", None)
    if layers is None:
        raise RuntimeError(
            "expected a standard nn.TransformerEncoder "
            "(use_fast_transformer=False) to expose .layers")
    for layer in list(layers)[-n_blocks:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def encode_cells(model, dataset, collator, batch_size: int, device: str,
                 pad_id: int, out_dim: int,
                 proj: nn.Module | None = None) -> np.ndarray:
    """[B,L,512] encoder -> <cls> row (position 0) -> optional projection."""
    from torch.utils.data import DataLoader, SequentialSampler

    loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=SequentialSampler(dataset),
                        collate_fn=collator, num_workers=4,
                        pin_memory=True)
    model.eval()
    if proj is not None:
        proj.eval()
    out = np.zeros((len(dataset), out_dim), dtype=np.float32)
    c = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=device.startswith("cuda")):
        for batch in loader:
            g = batch["gene"].to(device)
            m = g.eq(pad_id)
            h = model._encode(g, batch["expr"].to(device),
                              src_key_padding_mask=m)[:, 0, :]
            if proj is not None:
                h = proj(h)
            h = h.float().cpu().numpy()
            out[c:c + len(h)] = h
            c += len(h)
    return out


def emb_dim(model) -> int:
    d = getattr(model, "d_model", None)
    if d is None:
        d = model.encoder.weight.shape[1]
    return int(d)


def train_contrastive_scgpt(
    model,
    cellset: ScgptCellSet,
    emb_type: torch.Tensor,
    emb_path: torch.Tensor,
    cfg,
    collator,
    device: str,
    pad_id: int,
    ft_mode: str = "full",
    ft_blocks: int = 2,
    lr: float = 1e-5,
    batch_size: int = 32,
    epochs: int = 3,
    max_cells_epoch: int = 0,
    donors_per_pos: np.ndarray | None = None,
    path_levels: torch.Tensor | None = None,
    path_bank: torch.Tensor | None = None,
    ord_tau: float = 0.0,
    seed: int = 0,
):
    """Dual-head InfoNCE on top of the scGPT <cls> embedding.

    emb_type/emb_path: [n_train, text_dim] frozen caption embeddings
    aligned to cellset positions (cellset.index == train cells).

    ft_mode "full" trains everything; "lastN" trains the last ft_blocks
    encoder layers only (PEFT arm). Batch ids are local cellset
    positions, so caption tensors are indexed directly by batch["id"].
    """
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if ft_mode == "full":
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif ft_mode == "lastN":
        n_train = freeze_all_but_last_blocks(model, ft_blocks)
    else:
        raise ValueError(f"ft_mode {ft_mode}")

    d_model = emb_dim(model)
    proj_t = nn.Linear(d_model, emb_type.shape[1]).to(device)
    proj_p = nn.Linear(d_model, emb_path.shape[1]).to(device)
    params = ([p for p in model.parameters() if p.requires_grad]
              + list(proj_t.parameters()) + list(proj_p.parameters()))
    opt = torch.optim.AdamW(params, lr=lr)

    T_t = emb_type.to(device)
    T_p = emb_path.to(device)
    soft = (path_levels is not None and path_bank is not None
            and ord_tau and ord_tau > 0)
    if soft:
        lv = path_levels.to(device)
        bank = path_bank.to(device)

    n = len(cellset)
    if max_cells_epoch and max_cells_epoch < n and donors_per_pos is not None:
        donor_groups = pd_groupby_positions(donors_per_pos)
        quota = max(1, max_cells_epoch // max(1, len(donor_groups)))
        epoch_sampler = lambda e: np.sort(np.concatenate([
            rng.choice(g, size=min(len(g), quota), replace=False)
            for g in donor_groups]))
    else:
        epoch_sampler = lambda e: rng.permutation(n)

    history = []
    collate_fn = collate_keep_id(collator)
    for ep in range(epochs):
        order = epoch_sampler(ep)
        loader = DataLoader(
            cellset, batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(order.tolist()),
            collate_fn=collate_fn, num_workers=8, pin_memory=True,
            drop_last=False)
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            g = batch["gene"].to(device, non_blocking=True)
            mask = g.eq(pad_id)
            idx = batch["id"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                h = model._encode(g, batch["expr"].to(device),
                                  src_key_padding_mask=mask)[:, 0, :]
                z_t = proj_t(h)
                z_p = proj_p(h)
                loss_p = (soft_ordinal_nce(z_p, bank, lv[idx],
                                           cfg.temperature, ord_tau)
                          if soft else info_nce(z_p, T_p[idx], cfg.temperature))
                loss = info_nce(z_t, T_t[idx], cfg.temperature) + loss_p
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        history.append(ep_loss / max(1, nb))
    model.eval()
    meta = {"trainable_params": n_train, "epoch_cells": len(order)}
    return model, proj_t, proj_p, history, meta


def pd_groupby_positions(labels: np.ndarray) -> list[np.ndarray]:
    """positions of each unique label (donor-stratified subsampling)."""
    labels = np.asarray(labels)
    groups: dict[str, list[int]] = {}
    for i, v in enumerate(labels):
        groups.setdefault(str(v), []).append(i)
    return [np.asarray(sorted(v)) for v in groups.values()]
