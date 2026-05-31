"""
train.py — LoRA fine-tune a small LLM on RAPHI's filing classification data.

Base model: Qwen/Qwen2.5-3B-Instruct (Apache 2.0, no HF token needed)
Method:     LoRA (r=16) via PEFT — only ~0.5% of parameters trained
Quantize:   4-bit NF4 on CUDA; full float32 on Apple Silicon / CPU

Hardware requirements:
  CUDA GPU (A100 40GB)  — recommended, ~4 hrs, cost ~$20 on Lambda Labs
  Apple Silicon (M2+)   — works via MPS, ~12 hrs on M2 Max
  CPU                   — extremely slow, not recommended

Steps:
  1. Run label_builder first:  python -m backend.finetune.label_builder
  2. Then fine-tune:           python -m backend.finetune.train
  3. Model saved to:           data/finetune/model/

Usage:
  python -m backend.finetune.train
  python -m backend.finetune.train --base-model Qwen/Qwen2.5-1.5B-Instruct
  python -m backend.finetune.train --epochs 5 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("raphi.finetune.train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_ROOT       = Path(__file__).resolve().parent.parent.parent
_DATA_PATH  = _ROOT / "data" / "finetune" / "training_data.jsonl"
_MODEL_OUT  = _ROOT / "data" / "finetune" / "model"
_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _label_counts(records: list[dict]) -> str:
    counts: dict[str, int] = {}
    for r in records:
        sig = r.get("meta", {}).get("signal", "?")
        counts[sig] = counts.get(sig, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def train(
    base_model:   str   = _BASE_MODEL,
    data_path:    Path  = _DATA_PATH,
    output_dir:   Path  = _MODEL_OUT,
    epochs:       int   = 3,
    batch_size:   int   = 4,
    grad_accum:   int   = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int  = 1024,
    eval_split:   float = 0.1,
) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}\n"
            "Install: pip install transformers trl peft datasets bitsandbytes accelerate"
        ) from exc

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Run `python -m backend.finetune.label_builder` first."
        )

    records = _load_jsonl(data_path)
    if len(records) < 10:
        raise ValueError(
            f"Only {len(records)} examples in {data_path}. "
            "Run label_builder with more tickers first (need ≥ 10)."
        )

    device = _detect_device()
    logger.info("Device: %s | Base model: %s", device, base_model)
    logger.info("Examples: %d  (%s)", len(records), _label_counts(records))

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    texts = [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in records
    ]

    n_eval     = max(1, int(len(texts) * eval_split))
    train_ds   = Dataset.from_dict({"text": texts[n_eval:]})
    eval_ds    = Dataset.from_dict({"text": texts[:n_eval]})
    logger.info("Train: %d  Eval: %d", len(train_ds), len(eval_ds))

    # 4-bit quantization only on CUDA — bitsandbytes doesn't support MPS/CPU
    bnb_config = None
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto" if device in ("cuda", "mps") else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if device == "cuda" and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    output_dir.mkdir(parents=True, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=(device == "cuda"),
        fp16=False,
        logging_steps=10,
        save_steps=200,
        eval_steps=200,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    logger.info("Training started …")
    trainer.train()
    logger.info("Training complete.")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Record base model name so FilingClassifier can load it later
    with open(output_dir / "raphi_finetune_config.json", "w") as f:
        json.dump({
            "base_model":       base_model,
            "trained_examples": len(train_ds),
            "eval_examples":    len(eval_ds),
            "epochs":           epochs,
            "label_counts":     _label_counts(records),
        }, f, indent=2)

    logger.info("Adapter + tokenizer saved to %s", output_dir)
    return str(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SEC filing classifier with LoRA")
    parser.add_argument("--base-model",  default=_BASE_MODEL,       help="HuggingFace base model ID")
    parser.add_argument("--data",        default=str(_DATA_PATH),   help="JSONL training data path")
    parser.add_argument("--out",         default=str(_MODEL_OUT),    help="Output directory for adapter")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int,   default=1024)
    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_path=Path(args.data),
        output_dir=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_len,
    )


if __name__ == "__main__":
    main()
