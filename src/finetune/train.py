"""Train: load model -> attach LoRA -> split data -> SFT -> save."""
from __future__ import annotations

import argparse

import unsloth  # must be first, before trl/transformers/peft
import torch
from trl import SFTConfig, SFTTrainer

from finetune.data import load_and_split, load_config, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model(cfg, for_inference=False)
    train_ds, val_ds = load_and_split(cfg, tokenizer)

    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            num_train_epochs=cfg["epochs"],
            learning_rate=float(cfg["learning_rate"]),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=cfg["logging_steps"],
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=cfg["seed"],
            output_dir=cfg["output_dir"],
            eval_strategy="epoch",
            save_strategy="epoch",
        ),
    )

    trainer.train()

    save_path = f"{cfg['output_dir']}/lora_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved LoRA adapters to {save_path}")


if __name__ == "__main__":
    main()
