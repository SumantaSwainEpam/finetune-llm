# finetune-llm

Minimal LoRA fine-tuning pipeline for small open LLMs using Unsloth + TRL.

**Author:** Sumanta Swain

> **Requires an NVIDIA GPU.** Run on [Google Colab](#run-on-google-colab) (free T4) or a local CUDA machine.

## Project structure

```text
finetune-llm/
├── colab_train.ipynb       # one-click Colab notebook (clone → install → train → eval)
├── pyproject.toml          # project deps + CLI scripts (ft-train, ft-eval)
├── configs/
│   └── config.yaml         # all knobs: model, LoRA, data, training
├── data/
│   └── dataset.jsonl       # your {"instruction", "output"} pairs
└── src/finetune/
    ├── data.py             # config load, model load, format + split
    ├── train.py            # training entrypoint  -> ft-train
    └── evaluate.py         # eval on held-out val -> ft-eval
```

## Run on Google Colab

1. Open the notebook directly in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SumantaSwainEpam/finetune-llm/blob/main/colab_train.ipynb)

   > Replace `SumantaSwainEpam/finetune-llm` in the badge URL above with your actual GitHub path.

2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`

3. In the first cell, set your repo URL:

   ```python
   REPO_URL = "https://github.com/SumantaSwainEpam/finetune-llm.git"
   ```

4. Run all cells top to bottom. The notebook will:
   - Clone this repo
   - Install all dependencies
   - Run training and save LoRA adapters to `outputs/lora_model/`
   - Evaluate on the held-out validation split
   - (Optional) Download the trained adapters as a zip

## Local setup (NVIDIA GPU required)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv sync

# Train
uv run ft-train

# Evaluate
uv run ft-eval --n 5
```

## Your data

Put your examples in `data/dataset.jsonl` — one JSON object per line:

```json
{"instruction": "Convert 10 kilometers to miles.", "output": "10 kilometers = 10 × 0.621371 miles\n10 kilometers ≈ 6.21371 miles\n\nSo, 10 kilometers is approximately 6.21 miles."}
{"instruction": "What is 15% of 200?", "output": "15% of 200:\n\nStep-by-step:\n- Convert percentage to decimal: 15% = 0.15\n- Multiply: 0.15 × 200 = 30\n\nSo, 15% of 200 is 30."}
```

The dataset includes **31 examples** covering:

- Unit conversions (km ↔ miles, °C ↔ °F, kg ↔ lbs, grams, liters, hours)
- Arithmetic with step-by-step working (%, ÷, ×, squares, cubes, square roots)
- Geometry (area, perimeter, circumference)
- Physics (speed = distance / time, simple interest)
- Python code snippets
- General knowledge and vocabulary

Aim for **hundreds to thousands** of consistent examples for a meaningful fine-tune.

## Configuration

All settings live in `configs/config.yaml`:

| Key | Default | Description |
| --- | --- | --- |
| `model_name` | `unsloth/Llama-3.2-3B-Instruct` | Any Unsloth or HF model ID |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `batch_size` | `2` | Per-device train batch size |
| `epochs` | `3` | Number of training epochs |
| `learning_rate` | `2e-4` | Learning rate |
| `val_split` | `0.1` | Fraction of data held out for eval |
| `output_dir` | `outputs` | Where checkpoints and adapters are saved |

## Swapping models

Change `model_name` in `config.yaml` to any Unsloth or HuggingFace model ID — nothing else needs to change:

```yaml
model_name: "unsloth/Qwen2.5-Coder-3B-Instruct"
# model_name: "unsloth/gemma-2-2b-it"
# model_name: "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```

## Pipeline overview

```text
load model (4-bit) → attach LoRA adapters → format + split data → SFT train → save & eval
```

## Author

Sumanta Swain

- GitHub: [SumantaSwainEpam](https://github.com/SumantaSwainEpam)
- LinkedIn: [linkedin.com/in/sumanta-swain-556518242](https://www.linkedin.com/in/sumanta-swain-556518242/)
