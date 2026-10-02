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
    "README.md",
    "LICENSE",
    "NOTICE",
]

SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"[a-f0-9]{48}"),  # hermes-style scoped keys
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
]


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
    return hits


def stage(out: Path) -> dict:
    staged = {}
    for rel in WHITELIST:
        src = ROOT / rel
        dst = out / rel
        if src.is_dir():
            shutil.copytree(src, dst)
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
            "release": "biocellai-foundations-alpha",
            "license": {"code": "Apache-2.0", "artifacts": "CC-BY-4.0"},
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
            commit_message="BioCellAI Foundations: captions, retrieval corpora, manifests, docs",
        )
        print(f"uploaded -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
