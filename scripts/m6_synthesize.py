#!/usr/bin/env python3
"""M6 — RAG synthesis: retrieved PubMed docs → LLM functional description.

Reads the retrieval pool (data/retrieval/m6/{arm}/*.jsonl, which carries
abstracts) and asks a local LLM to synthesize a two-sentence biological
description grounded in the retrieved literature. This is the quality lever
over raw abstract concatenation: retrieval recall stays high, the LLM
filters topicality.

Variants:
  honest  — markers + retrieved excerpts, NO cell-type name in prompt;
            literal name occurrences stripped from output (flagged).
  labeled — name + markers + excerpts (upper bound / leakage probe).

Writes the same {cell_type: text} contract as m4_generate.py so the
type_llm caption mode consumes it unchanged.

Usage (GPU node):
    python scripts/m6_synthesize.py --variant honest \
        --docdir data/retrieval/m6/rag_markers_alias \
        --markers-json data/manifests/markers_train_s0.json \
        --out data/text/type_desc_ragllm_honest.json \
        --prov-out data/text/type_desc_ragllm_honest.prov.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m4_generate import generate, strip_label  # noqa: E402  (same dir)

HONEST_PROMPT = (
    "The following excerpts from PubMed abstracts describe a human "
    "{tissue} cell population that highly expresses the marker genes: "
    "{markers}.\n\n"
    "Literature excerpts:\n{excerpts}\n\n"
    "In exactly two sentences, describe this population's biological "
    "function, developmental lineage, and characteristic molecular program. "
    "Base the description only on the excerpts and marker genes above. "
    "Plain scientific prose, no lists, and do not use any conventional "
    "{kind} cell-type name."
)

LABELED_PROMPT = (
    'The following PubMed excerpts concern the human {tissue} cell type '
    '"{cell_type}".\n\nLiterature excerpts:\n{excerpts}\n\n'
    "In exactly two sentences, describe this cell type's biological "
    "function, developmental lineage, and characteristic marker genes "
    "({markers}). Plain scientific prose, no lists."
)

_TISSUE = {"blood": ("blood", "immune"), "brain": ("brain", "brain")}


def load_excerpts(docdir: Path, cell_type: str, n_docs: int = 12,
                  max_chars: int = 6000) -> str:
    safe = cell_type.replace("/", "_").replace(" ", "_")
    path = docdir / f"{safe}.jsonl"
    if not path.exists():
        return "(no retrieved documents)"
    parts, used = [], 0
    with open(path) as f:
        for line in f:
            if len(parts) >= n_docs or used >= max_chars:
                break
            d = json.loads(line)
            frag = f"{d.get('title') or ''}. {str(d.get('abstract') or '')[:400]}"
            parts.append(frag.strip())
            used += len(frag)
    return "\n".join(f"- {p}" for p in parts) or "(no retrieved documents)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["honest", "labeled"], required=True)
    p.add_argument("--docdir", required=True)
    p.add_argument("--markers-json", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--out", required=True)
    p.add_argument("--prov-out", required=True)
    p.add_argument("--context", choices=list(_TISSUE), default="blood")
    p.add_argument("--prior-only", action="store_true",
                   help="control arm: no excerpts — isolates LLM parametric "
                        "knowledge from retrieved-literature grounding")
    p.add_argument("--n-docs", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=160)
    args = p.parse_args()

    markers = json.loads(Path(args.markers_json).read_text())
    docdir = Path(args.docdir)
    cell_types = list(markers.keys())

    prompts, excerpts_used = [], {}
    for ct in cell_types:
        m = ", ".join(markers.get(ct, []))
        exc = ("(no retrieved documents)" if args.prior_only
               else load_excerpts(docdir, ct, n_docs=args.n_docs))
        excerpts_used[ct] = exc[:200]
        tissue, kind = _TISSUE[args.context]
        if args.variant == "honest":
            prompts.append(HONEST_PROMPT.format(
                tissue=tissue, kind=kind, markers=m, excerpts=exc))
        else:
            prompts.append(LABELED_PROMPT.format(
                tissue=tissue, cell_type=ct, markers=m, excerpts=exc))

    raws = generate(args.model, prompts, args.temperature, args.max_new_tokens)

    descs, prov = {}, {"_meta": {
        "model": args.model, "variant": args.variant, "docdir": str(docdir),
        "n_docs": args.n_docs, "temperature": args.temperature,
        "prior_only": args.prior_only,
    }}
    n_leaked = 0
    for ct, prompt, raw in zip(cell_types, prompts, raws):
        if args.variant == "honest":
            text, leaked = strip_label(raw, ct)
            n_leaked += leaked
        else:
            text, leaked = raw, True
        descs[ct] = text
        prov[ct] = {"prompt_head": prompt[:400], "raw": raw, "text": text,
                    "contains_type_name": leaked,
                    "excerpts_head": excerpts_used[ct]}
        print(f"--- {ct} (leaked={leaked})\n{text}\n", flush=True)

    prov["_meta"]["n_types_with_label_in_output"] = n_leaked
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(descs, indent=2))
    Path(args.prov_out).write_text(json.dumps(prov, indent=2))
    print(f"wrote {args.out} | {n_leaked}/{len(cell_types)} leaked")


if __name__ == "__main__":
    main()
