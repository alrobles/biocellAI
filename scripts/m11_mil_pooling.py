"""M11-B1 — MIL/attention donor pooling vs mean pooling.

Treats each donor as a bag of cells. A gated attention pooler
(Ilse et al. 2018) learns which cells carry the donor-level signal,
replacing the mean-pool used so far:

    w_i = softmax( w^T tanh(V h_i) )          donor_emb = sum_i w_i h_i
    donor_emb -> linear -> target (ridge-style ridge-regression by SGD)

Evaluation is identical to the mean-pool protocol: train pooling+head on
train donors only, ridge-eval (Spearman/R2) on held-out donors, 3 seeds.
Also reports non-parametric pooling baselines (median, top-quantile means)
so the attention gain is judged against free alternatives.

Usage:
    python scripts/m11_mil_pooling.py \
        --emb experiments/m10_dual_cog/embeddings/emb_minilm_s0.npz \
        --donor-table /beegfs/.../seaad_donors_allregions.csv \
        --target CPS_Global --seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from biocellai.data import donors_from_mask, split_donor_ids
from biocellai.progression import regress_heldout
from biocellai.revalidation import reserve_output


class GatedAttnPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1, bias=False)
        self.out = nn.Linear(dim, 1)

    def forward(self, bags, lengths):
        """bags: (B, L, D) padded cell embeddings; lengths: (B,) valid counts."""
        a = self.w(torch.tanh(self.V(bags))).squeeze(-1)          # (B, L)
        mask = torch.arange(bags.shape[1], device=bags.device)
        a = a.masked_fill(mask[None, :] >= lengths[:, None], -1e9)
        w = torch.softmax(a, dim=1)
        pooled = (w.unsqueeze(-1) * bags).sum(1)                   # (B, D)
        return self.out(pooled).squeeze(-1), w


def donor_bags(emb, donors, tr_d, max_cells=1500, seed=0):
    """Subsample each donor's cells to max_cells; return padded tensor."""
    rng = np.random.default_rng(seed)
    bags, ys = [], []
    for d in tr_d:
        idx = np.where(donors == d)[0]
        if len(idx) > max_cells:
            idx = rng.choice(idx, max_cells, replace=False)
        bags.append(torch.from_numpy(emb[idx]))
        ys.append(d)
    L = max(len(b) for b in bags)
    D = bags[0].shape[1]
    padded = torch.zeros(len(bags), L, D)
    lens = torch.tensor([len(b) for b in bags])
    for i, b in enumerate(bags):
        padded[i, : len(b)] = b
    return padded, lens, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--donor-table", required=True)
    ap.add_argument("--target", default="CPS_Global")
    ap.add_argument("--seeds", type=int, nargs="+", required=True,
                    help="one seed matching the split saved in --emb")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-cells", type=int, default=1500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if len(args.seeds) != 1:
        raise ValueError("a trained embedding file must retain its original donor split")
    out = reserve_output(args.out) if args.out else None

    z = np.load(args.emb, allow_pickle=True)
    emb, donors = z["emb"], z["donor"].astype(str)
    saved_train, saved_test = donors_from_mask(donors, z["is_test"])
    table = pd.read_csv(args.donor_table).groupby("donor_id").first()
    uniq = np.unique(donors)
    y_all = table.loc[uniq, args.target].to_numpy(dtype=float)
    ok = ~np.isnan(y_all)
    uniq, y_all = uniq[ok], y_all[ok]

    rows = []
    for seed in args.seeds:
        # identical split as biocellai.data.donor_split(seed): sorted donors,
        # default_rng shuffle, first ~25% -> test
        expected_train, expected_test = split_donor_ids(donors, seed=seed)
        if set(expected_train) != set(saved_train) or set(expected_test) != set(saved_test):
            raise ValueError("requested seed differs from the embedding training split")
        tr_d = [d for d in saved_train if d in set(uniq)]
        te_d = [d for d in saved_test if d in set(uniq)]
        y = pd.Series(y_all, index=uniq)

        # --- mean-pool ridge baseline (identical to existing protocol) ---
        demb = pd.DataFrame(emb).assign(donor=donors).groupby("donor").mean()
        base = regress_heldout(demb, y, pd.Index(tr_d), pd.Index(te_d))
        rows.append(dict(seed=seed, model="mean_ridge", **base))

        # --- free pooling baselines ---
        for q in (0.5, 0.9):
            dq = (pd.DataFrame(emb).assign(donor=donors)
                  .groupby("donor").quantile(q))
            r = regress_heldout(dq, y, pd.Index(tr_d), pd.Index(te_d))
            rows.append(dict(seed=seed, model=f"quantile_{q}", **r))

        # --- attention MIL pooler ---
        torch.manual_seed(seed)
        pool = GatedAttnPool(emb.shape[1])
        opt = torch.optim.AdamW(pool.parameters(), lr=3e-4, weight_decay=1e-2)
        bags_tr, lens_tr, ids_tr = donor_bags(emb, donors, tr_d,
                                            args.max_cells, seed)
        y_tr = torch.tensor(y.loc[ids_tr].to_numpy(), dtype=torch.float32)
        for ep in range(args.epochs):
            pool.train()
            opt.zero_grad()
            pred, _ = pool(bags_tr, lens_tr)
            loss = nn.functional.mse_loss(pred, y_tr)
            loss.backward()
            opt.step()
        # end-to-end eval: model's own prediction on held-out donors
        pool.eval()
        with torch.no_grad():
            bags_te, lens_te, ids_te = donor_bags(
                emb, donors, te_d, args.max_cells, seed + 999)
            pred_te, _ = pool(bags_te, lens_te)
        from scipy.stats import spearmanr
        from sklearn.metrics import r2_score
        y_te = y.loc[ids_te].to_numpy()
        rows.append(dict(seed=seed, model="attn_mil",
                         rho=spearmanr(pred_te.numpy(), y_te).statistic,
                         r2=r2_score(y_te, pred_te.numpy())))

    df = pd.DataFrame(rows)
    df["evaluation_protocol"] = "v2_readout_on_unrevalidated_inputs"
    if out is not None:
        df.to_csv(out / "mil_pooling_metrics.csv", index=False)
    print(df.groupby("model")[["rho", "r2"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
