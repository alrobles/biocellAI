"""R4 audit manifest for external scFMs (R4-GENEFORMER / R4-SCGPT-CW).

Writes one JSON per model recording checkpoint identity, package
versions, source revision, compatibility shims, and parameter counts —
the "checkpoint parity + reproducible environment, no hidden patches"
evidence. Run once on HPC:

    python scripts/v2_scfm_audit.py \
        --gf-snapshot <snapshot> --scgpt-dir <dir> --cw-ckpt <ckpt> \
        --out experiments/v2_revalidation/scfm_audit/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from biocellai.scfm import GF_SPECS, PATCHES

ENV_PKGS = ["torch", "transformers", "anndata", "numpy", "scipy",
            "scikit-learn", "geneformer", "scgpt", "cellwhisperer",
            "pytorch-lightning", "torchmetrics", "sentence-transformers"]


def _sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _git_rev(repo):
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_dirty(repo):
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=15)
        return out.stdout.strip().splitlines() if out.returncode == 0 else []
    except Exception:
        return []


def _env_versions():
    env = {}
    for p in ENV_PKGS:
        try:
            env[p] = version(p)
        except Exception:
            env[p] = None
    return env


def _hf_snapshot_rev(snap: Path):
    """HF cache snapshot dir name == commit sha; also read refs/main."""
    refs = snap.parent.parent / "refs" / "main"
    ref = refs.read_text().strip() if refs.exists() else None
    return {"snapshot_id": snap.name, "refs_main": ref}


def _model_files(d: Path):
    return [{"file": f.name, "size": f.stat().st_size,
             "sha256": _sha256(f)} for f in sorted(d.iterdir())
            if f.is_file() and f.suffix in
            {".bin", ".safetensors", ".pt", ".ckpt"}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gf-snapshot", type=Path, required=True)
    p.add_argument("--scgpt-dir", type=Path, required=True)
    p.add_argument("--cw-ckpt", type=Path, required=True)
    p.add_argument("--cw-repo", type=Path,
                   default=Path("/beegfs/a474r867/bioai/repos/cellwhisperer"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    env = _env_versions()
    args.out.mkdir(parents=True, exist_ok=True)

    for key in GF_SPECS:
        spec = GF_SPECS[key]
        mdir = args.gf_snapshot / spec["model_dir"]
        rec = {
            "model": key,
            "checkpoint": {"dir": str(mdir), **_hf_snapshot_rev(
                args.gf_snapshot), "files": _model_files(mdir)},
            "dicts": {
                "token": str(args.gf_snapshot / "geneformer" / spec["dict_subdir"] /
                             f"token_dictionary_{spec['tag']}.pkl"),
                "median": str(args.gf_snapshot / "geneformer" / spec["dict_subdir"] /
                              f"gene_median_dictionary_{spec['tag']}.pkl"),
                "mapping": str(args.gf_snapshot / "geneformer" / spec["dict_subdir"] /
                               f"ensembl_mapping_dict_{spec['tag']}.pkl")},
            "tokenizer": "biocellai.scfm.gf_tokenize — upstream-parity "
                         "fixture: experiments/v2_revalidation/gf_fixture.json "
                         f"(special={spec['special']}, max_len={spec['max_len']})",
            "patches": PATCHES[key],
            "environment": env,
        }
        (args.out / f"{key}.audit.json").write_text(json.dumps(rec, indent=2))
        print("wrote", key)

    sc = args.scgpt_dir
    rec = {
        "model": "scgpt_wh",
        "checkpoint": {"dir": str(sc), "files": _model_files(sc),
                       "extra": [f.name for f in sorted(sc.iterdir())]},
        "loader": "scgpt.tasks.embed_data (0.2.4 upstream entry point)",
        "patches": PATCHES["scgpt_wh"],
        "environment": env,
    }
    (args.out / "scgpt_wh.audit.json").write_text(json.dumps(rec, indent=2))
    print("wrote scgpt_wh")

    cw = args.cw_ckpt
    rec = {
        "model": "cw_clip_v1",
        "checkpoint": {"file": str(cw), "size": cw.stat().st_size,
                       "sha256": _sha256(cw)},
        "loader": "cellwhisperer.utils.model_io.load_cellwhisperer_model",
        "clone": {"path": str(args.cw_repo),
                  "git_rev": _git_rev(args.cw_repo),
                  "dirty_files": _git_dirty(args.cw_repo)},
        "patches": PATCHES["cw_clip_v1"],
        "environment": env,
    }
    (args.out / "cw_clip_v1.audit.json").write_text(json.dumps(rec, indent=2))
    print("wrote cw_clip_v1")


if __name__ == "__main__":
    main()
