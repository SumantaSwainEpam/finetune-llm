"""Shared utilities: config loading, model setup, dataset prep."""
from __future__ import annotations

import yaml
from datasets import Dataset, load_dataset


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, for_inference: bool = False):
    """Load a 4-bit model and (optionally) attach LoRA adapters for training."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )

    if not for_inference:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora_r"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg["seed"],
        )
    return model, tokenizer


def build_text(example: dict, tokenizer, system_prompt: str,
               include_answer: bool = True) -> str:
    """Render one example into the model's chat format."""
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]}]
    if include_answer:
        msgs.append({"role": "assistant", "content": example["output"]})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=not include_answer
    )


def load_and_split(cfg: dict, tokenizer):
    """Load JSONL, format for training, and split into train/val."""
    ds = load_dataset("json", data_files=cfg["data_path"], split="train")

    sp = cfg["system_prompt"]
    ds = ds.map(lambda ex: {"text": build_text(ex, tokenizer, sp)})

    split = ds.train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    return split["train"], split["test"]
