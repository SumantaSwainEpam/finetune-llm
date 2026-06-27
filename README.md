# finetune-llm

Minimal LoRA fine-tuning pipeline for small open LLMs, using `uv` + Unsloth + TRL.

## Project structure

```
finetune-llm/
├── pyproject.toml          # uv project + deps + CLI scripts
├── configs/
│   └── config.yaml         # all knobs: model, LoRA, data, training
├── data/
│   └── dataset.jsonl       # your {"instruction","output"} pairs
└── src/finetune/
    ├── data.py             # config load, model load, format + split
    ├── train.py            # training entrypoint  -> ft-train
    └── evaluate.py         # eval on held-out val -> ft-eval
```

## Setup

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the project root: create venv + install deps
uv sync
```

> Note: Unsloth needs a CUDA GPU. Run on Colab/Kaggle or a local NVIDIA box.
> On Colab you can still `uv pip install -e .` inside the notebook's environment.

## Usage

1. Put your data in `data/dataset.jsonl` (one JSON object per line):
   ```json
   {"instruction": "...", "output": "..."}
   ```
   Aim for hundreds to thousands of consistent examples.

2. Edit `configs/config.yaml` — at minimum set `model_name` and `system_prompt`.

3. Train:
   ```bash
   uv run ft-train
   ```

4. Evaluate on the held-out validation split:
   ```bash
   uv run ft-eval --n 5
   ```

## Swapping models

Change `model_name` in the config to any Unsloth/HF id, e.g.
`unsloth/Qwen2.5-Coder-3B-Instruct`, `unsloth/gemma-2-2b-it`,
`unsloth/mistral-7b-instruct-v0.3-bnb-4bit`. Nothing else changes.

## The 5 conceptual steps

load model → attach LoRA adapters → format + split data → SFT train → save & eval
