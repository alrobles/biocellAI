#!/usr/bin/env python3
"""M4: generate per-cell-type biological descriptions with a local HF LLM.

Two prompt variants (the scientific point of M4):
  honest  — the model sees ONLY the marker-gene list (never the type name).
            Any literal occurrence of the type name in the output is stripped
            and flagged in provenance. Tests whether biological prose inferred
            from markers grounds better than the bare marker list.
  labeled — the model is told the type name. Prose naturally contains it, so
            captions built from these descriptions are a LEAKAGE PROBE
            (upper bound), not an honest grounding result.

Writes:
  --out       {cell_type: cleaned description}   (consumed by caption mode type_llm)
  --prov-out  full provenance: prompts, raw outputs, label-leak flags, model meta

Usage (on a GPU node):
    python scripts/m4_generate.py --variant honest \
        --model Qwen/Qwen2.5-14B-Instruct \
        --markers-json data/manifests/markers_train_s0.json \
        --cell-types-json data/manifests/cell_types.json \
        --out data/text/type_desc_qwen_honest.json \
        --prov-out data/text/type_desc_qwen_honest.prov.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HONEST_PROMPT = (
    "A cell population from human peripheral blood highly expresses these "
    "marker genes: {markers}.\n\n"
    "Write exactly two sentences describing this population's biological "
    "function, developmental lineage, and characteristic molecular program, "
    "suitable as a caption for representation learning. Plain scientific "
    "prose, no lists, and do not use any conventional immune cell-type name."
)

LABELED_PROMPT = (
    'Write exactly two sentences describing the human blood cell type '
    '"{cell_type}" — its biological function, developmental lineage, and '
    "characteristic marker genes ({markers}). Plain scientific prose, no "
    "lists."
)


def generate(model_name: str, prompts: list[str], temperature: float,
             max_new_tokens: int) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto",
        )
    except ValueError:  # accelerate not installed -> single-GPU fallback
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16,
        ).to("cuda")
    model.eval()

    outs = []
    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                top_p=0.9, pad_token_id=tok.eos_token_id,
            )
        n_in = enc["input_ids"].shape[1]
        outs.append(tok.decode(gen[0][n_in:], skip_special_tokens=True).strip())
    return outs


def strip_label(text: str, cell_type: str) -> tuple[str, bool]:
    """Remove literal occurrences of the type name; return (text, leaked?)."""
    pat = re.compile(re.escape(cell_type), re.IGNORECASE)
    leaked = bool(pat.search(text))
    return pat.sub("this cell type", text), leaked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["honest", "labeled"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--markers-json", required=True)
    p.add_argument("--cell-types-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prov-out", required=True)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=140)
    args = p.parse_args()

    markers = json.loads(Path(args.markers_json).read_text())
    cell_types = json.loads(Path(args.cell_types_json).read_text())

    prompts = []
    for ct in cell_types:
        m = ", ".join(markers.get(ct, []))
        if args.variant == "honest":
            prompts.append(HONEST_PROMPT.format(markers=m))
        else:
            prompts.append(LABELED_PROMPT.format(cell_type=ct, markers=m))

    raws = generate(args.model, prompts, args.temperature, args.max_new_tokens)

    descs, prov = {}, {"_meta": {
        "model": args.model, "variant": args.variant,
        "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
    }}
    n_leaked = 0
    for ct, prompt, raw in zip(cell_types, prompts, raws):
        if args.variant == "honest":
            text, leaked = strip_label(raw, ct)
            n_leaked += leaked
        else:
            text, leaked = raw, True
        descs[ct] = text
        prov[ct] = {"prompt": prompt, "raw": raw, "text": text,
                    "contains_type_name": leaked}
        print(f"--- {ct} (leaked={leaked})\n{text}\n")

    prov["_meta"]["n_types_with_label_in_output"] = n_leaked
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(descs, indent=2))
    Path(args.prov_out).write_text(json.dumps(prov, indent=2))
    print(f"wrote {args.out} | {n_leaked}/{len(cell_types)} outputs contained the type name")


if __name__ == "__main__":
    main()
