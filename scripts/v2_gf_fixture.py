"""R4-GENEFORMER fixture: parity of biocellai.scfm.gf_tokenize vs the
upstream geneformer TranscriptomeTokenizer (v0.1.0).

Upstream input contract: AnnData keyed by var['ensembl_id'] (loom/h5ad),
collapse_gene_ids=True sums duplicated ensembl rows. Our extracts are
symbol-keyed, so scfm.gf_tokenize adapts via gene_name_id_dict first --
the symbol->ens adapter is unit-tested in tests/test_scfm.py; here we
verify the shared core (norm -> rank -> specials -> truncate) against
upstream on an ensembl-keyed fixture with duplicated ensembl rows.

    python scripts/v2_gf_fixture.py --snapshot <GF snapshot> \
        --out experiments/v2_revalidation/gf_fixture.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from biocellai.scfm import gf_load_dicts, gf_tokenize

VERSIONS = [  # (key, model_key, special, max_len)
    ("V1", "gf_v1_10m", False, 2048),
    ("V2", "gf_v2_104m", True, 4096),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--n-cells", type=int, default=24)
    p.add_argument("--n-genes", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from geneformer import TranscriptomeTokenizer

    rng = np.random.default_rng(args.seed)
    report = {"step": "v2_gf_fixture", "snapshot": str(args.snapshot)}

    for key, mkey, special, mlen in VERSIONS:
        tok, med, ensmap, nid = gf_load_dicts(args.snapshot, mkey)
        pkg = args.snapshot / "geneformer"
        # var ids = ensmap keys (alias/ensg space); pool keys whose
        # mapped ens is vocab+median-covered — upstream input contract
        items = sorted((k, v) for k, v in ensmap.items()
                       if isinstance(v, str) and v in med)
        pairs = items[: args.n_genes]
        # a true collapse pair: two DISTINCT keys -> same ens
        by_ens = {}
        for k, v in items[args.n_genes:]:
            if v in by_ens:
                dup_keys = (by_ens[v], k)
                break
            by_ens[v] = k
        else:
            dup_keys = None
        ids = [k for k, _ in pairs] + (list(dup_keys) if dup_keys else [])

        counts = sp.csr_matrix(
            rng.poisson(1.5, (args.n_cells, len(ids))).astype(np.float32))
        obs = pd.DataFrame({"donor_id": "d0"},
                           index=[f"fx{i}" for i in range(args.n_cells)])
        obs["n_counts"] = np.asarray(counts.sum(axis=1)).ravel()
        a = ad.AnnData(X=counts, obs=obs,
                       var=pd.DataFrame({"ensembl_id": ids},
                                        index=ids))

        # upstream: file-level tokenizer on the h5ad (V1 needs the gc30M
        # dicts explicitly; pkg bundles only gc104M)
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / "fx.h5ad"
            a.write_h5ad(hp)
            kw = {}
            if key == "V1":
                d30 = pkg / "gene_dictionaries_30m"
                kw = dict(
                    gene_median_file=str(d30 / "gene_median_dictionary_gc30M.pkl"),
                    token_dictionary_file=str(d30 / "token_dictionary_gc30M.pkl"),
                    gene_mapping_file=str(d30 / "ensembl_mapping_dict_gc30M.pkl"))
            tk = TranscriptomeTokenizer(nproc=1, model_version=key, **kw)
            seqs_up, _meta, _cnt = tk.tokenize_anndata(str(hp),
                                                      target_sum=10_000)
        seqs_up = [np.asarray(s) for s in seqs_up]
        if special:
            seqs_up = [np.concatenate(
                [[tok["<cls>"]], s[:mlen - 2], [tok["<eos>"]]])
                for s in seqs_up]
        else:
            seqs_up = [s[:mlen] for s in seqs_up]

        # ours: ens-keyed fixture -> ensmap dict (the nid symbol adapter
        # is unit-tested in tests/test_scfm.py)
        seqs_mine = gf_tokenize(a, tok, med, ensmap, nid, special, mlen,
                                dict_choice="ensmap")

        n = min(len(seqs_up), len(seqs_mine))
        diffs = [i for i in range(n)
                 if not np.array_equal(seqs_up[i], seqs_mine[i])]
        report[key] = {
            "cells_compared": n,
            "cells_identical": n - len(diffs),
            "match_rate": (n - len(diffs)) / max(n, 1),
            "first_diff_cell": diffs[0] if diffs else None,
            "upstream_len_sample": [int(len(s)) for s in seqs_up[:3]],
            "mine_len_sample": [int(len(s)) for s in seqs_mine[:3]],
            "special_tokens": special, "max_len": mlen,
            "mapping_file": kw.get("gene_mapping_file", "pkg default"),
        }
        print(f"{key}: {n - len(diffs)}/{n} cells identical "
              f"(first_diff={diffs[0] if diffs else None})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
