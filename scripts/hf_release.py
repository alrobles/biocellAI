#!/usr/bin/env python
"""Whitelist-based Hugging Face release packager.

Only paths in WHITELIST are packaged; every staged file is scanned for
secret-looking strings and the upload aborts on any hit. The token is read
from --token-file (default ~/env/hf-token) or HF_TOKEN; it is never logged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WHITELIST = [
    "data/text",
    "data/retrieval",
    "data/manifests",
    "data/gene_aliases.json",
    "docs",
    "spec",
    "experiments/v2_revalidation",
    "experiments/v2_confirmation",
    "experiments/v2_marker_recovery",
    "paper",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
]

# Safety net: even inside whitelisted dirs, only these extensions ship.
# Blocks raw matrices, checkpoints and donor tables regardless of layout.
ALLOWED_EXTENSIONS = {
    ".json", ".csv", ".md", ".txt", ".pdf", ".tex", ".cff", ".yaml",
    ".yml", ".bib", ".png",
}
DENIED_EXTENSIONS = {".h5ad", ".pt", ".ckpt", ".pth", ".npz", ".npy",
                     ".h5", ".zarr", ".loom"}

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
]

# Bare 40-hex tokens are treated separately: scoped keys look identical to
# git commit SHAs and HF revision IDs, so a match is only suspicious when it
# is not explainable as a public identifier (git rev-list, snapshot paths,
# .lock names, sha/hash/commit/git JSON fields).
HEX40 = re.compile(r"\b[a-f0-9]{40}\b")
SAFE_HEX40_CONTEXT = re.compile(
    r"(?i)[\w]*(sha|hash|rev|ref|snapshot|commit|git|version)[\w]*"
    r"[\"'\s]*[:=]")


def _hex40_hits(path: Path) -> list[str]:
    """Flag bare 40-hex tokens that are not explainable public IDs."""
    import subprocess
    try:
        git_shas = set(subprocess.run(
            ["git", "rev-list", "--all"], cwd=ROOT,
            capture_output=True, text=True).stdout.split())
    except OSError:
        git_shas = set()
    hits = []
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.stat().st_size > 4_000_000:
            continue
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for m in HEX40.finditer(text):
            tok = m.group(0)
            if tok in git_shas:
                continue
            ctx = text[max(0, m.start() - 80):m.start()]
            if ("snapshots/" in ctx or ".lock" in ctx or
                    "models--" in ctx or ".no_exist/" in ctx or
                    "hf_cache/" in ctx or "blobs/" in ctx or
                    ".git/" in ctx or "pack-" in ctx or
                    "objects/" in ctx or
                    SAFE_HEX40_CONTEXT.search(ctx)):
                continue
            hits.append(f"{f.relative_to(path)}:{tok[:8]}…")
    return hits


def scan(path: Path) -> list[str]:
    hits = []
    for f in sorted(path.rglob("*")):
        if not f.is_file() or f.stat().st_size > 4_000_000:
            continue
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                hits.append(str(f.relative_to(path)))
                break
    hits.extend(_hex40_hits(path))
    return hits


def _copy_filtered(src: Path, dst: Path) -> list[str]:
    """Copy a whitelisted dir keeping only allowed extensions."""
    copied = []
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() in DENIED_EXTENSIONS:
            raise SystemExit(
                f"ABORT: denied extension staged: {f.relative_to(src)}")
        if f.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied.append(str(target))
    return copied


def stage(out: Path) -> dict:
    staged = {}
    for rel in WHITELIST:
        src = ROOT / rel
        dst = out / rel
        if not src.exists():
            continue
        if src.is_dir():
            files = _copy_filtered(src, dst)
            staged[rel] = sorted(
                str(p.relative_to(out)) for p in dst.rglob("*") if p.is_file()
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            staged[rel] = [rel]
    return staged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="e.g. alrobles/biocellai-foundations")
    p.add_argument("--token-file", type=Path, default=Path.home() / "env/hf-token")
    p.add_argument("--dry-run", action="store_true", help="stage+scan only, no upload")
    args = p.parse_args()

    import os
    token = os.environ.get("HF_TOKEN") or args.token_file.read_text().strip()
    if not token.startswith("hf_"):
        raise SystemExit("token file does not look like a Hugging Face token")

    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / "release"
        stage_dir.mkdir()
        staged = stage(stage_dir)
        hits = scan(stage_dir)
        if hits:
            raise SystemExit(f"ABORT: secret-pattern hits in staged files: {hits}")
        files = []
        for f in sorted(stage_dir.rglob("*")):
            if f.is_file():
                files.append({
                    "path": str(f.relative_to(stage_dir)),
                    "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                    "bytes": f.stat().st_size,
                })
        manifest = {
            "release": "biocellai-foundations-v2",
            "license": {
                "code": "Apache-2.0",
                "artifacts": "CC-BY-4.0",
                "restrictions": {
                    "rosmap": "controlled-access (Synapse DUO); "
                              "no matrices, donor tables or ROSMAP-trained "
                              "weights are distributed",
                    "seaad": "CC BY-NC 4.0 + Allen Institute terms; "
                             "SEA-AD-trained weights withheld",
                    "third_party_weights": "not re-hosted; see "
                                           "spec/redistribution_manifest.json",
                },
            },
            "source_repo": "https://github.com/alrobles/biocellAI",
            "whitelist": WHITELIST,
            "n_files": len(files),
            "files": files,
        }
        (stage_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"staged {len(files)} files, secret scan clean")
        if args.dry_run:
            print(json.dumps({k: len(v) for k, v in staged.items()}, indent=2))
            return
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=str(stage_dir),
            commit_message="BioCellAI Foundations v2: corrected-protocol evidence, manifests, paper",
        )
        print(f"uploaded -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
