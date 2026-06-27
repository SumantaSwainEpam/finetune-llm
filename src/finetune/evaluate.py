"""Evaluate: load trained adapters, generate on the val set, print side-by-side."""
from __future__ import annotations

import argparse

import torch
from unsloth import FastLanguageModel
from finetune.data import build_text, load_and_split, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--adapters", default=None,
                    help="Path to saved LoRA adapters (default: outputs/lora_model)")
    ap.add_argument("--n", type=int, default=5, help="How many val examples to show")
    args = ap.parse_args()

    cfg = load_config(args.config)
    adapter_path = args.adapters or f"{cfg['output_dir']}/lora_model"

    # Load the fine-tuned adapters on top of the base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    # Reload raw val split (not the pre-templated text) to get instruction/output
    raw = load_and_split(cfg, tokenizer)[1]

    print("=" * 70)
    for i in range(min(args.n, len(raw))):
        ex = raw[i]
        prompt = build_text(ex, tokenizer, cfg["system_prompt"],
                            include_answer=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs,
                             max_new_tokens=cfg["eval_max_new_tokens"],
                             temperature=0.3, do_sample=True)
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        # Strip the prompt portion to isolate the model's answer
        answer = gen[len(tokenizer.decode(inputs["input_ids"][0],
                                          skip_special_tokens=True)):].strip()

        print(f"PROMPT:    {ex['instruction']}")
        print(f"EXPECTED:  {ex['output']}")
        print(f"GENERATED: {answer}")
        print("=" * 70)


if __name__ == "__main__":
    main()
