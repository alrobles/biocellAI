#!/usr/bin/env python3
"""Generate per-cell-type biological descriptions via a local Ollama LLM.

Runs on KU HPC login node (or anywhere that can reach the Ollama endpoint).
Produces data/text/type_descriptions.json — deterministic artifact that the
`type_llm` caption mode then consumes (LLM is only needed once; experiments
stay reproducible).

Usage:
    python scripts/gen_type_descriptions.py \
        --cell-types "CD4 T cell,CD8 T cell,B cell" \
        --markers-json markers.json \
        --ollama-url http://<node>:11434 \
        --model qwen2.5:14b-instruct-q4_K_M \
        --out data/text/type_descriptions.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

PROMPT = """You are a cell biologist. Write exactly two sentences describing the human blood cell type "{cell_type}" for a machine-learning caption.

Requirements:
- Do NOT use the literal phrase "{cell_type}" anywhere in the description.
- Mention its biological function and its characteristic marker genes: {markers}.
- Plain scientific prose, no headers, no lists.
"""


def ollama_generate(url: str, model: str, prompt: str, timeout: int = 300) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 120},
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell-types", required=True, help="comma-separated")
    p.add_argument("--markers-json", required=True, help="{cell_type: [genes]}")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--model", default="qwen2.5:14b-instruct-q4_K_M")
    p.add_argument("--out", default="data/text/type_descriptions.json")
    args = p.parse_args()

    markers = json.loads(Path(args.markers_json).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    descs = {}
    for ct in args.cell_types.split(","):
        ct = ct.strip()
        prompt = PROMPT.format(cell_type=ct, markers=", ".join(markers.get(ct, [])))
        desc = ollama_generate(args.ollama_url, args.model, prompt)
        # safety: strip literal label if the model ignored the instruction
        descs[ct] = desc.replace(ct, "this cell type")
        print(f"--- {ct}\n{descs[ct]}\n")

    out.write_text(json.dumps(descs, indent=2))
    print(f"wrote {out} ({len(descs)} types)")


if __name__ == "__main__":
    main()
