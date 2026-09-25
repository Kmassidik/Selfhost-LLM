#!/usr/bin/env python3
"""Task 1 — LoRA fine-tune Qwen2.5-0.5B to emit the extraction schema.

Prompt/completion format: the system message states the schema, the user turn
is the raw message, the assistant turn is the exact gold JSON. trl masks the
prompt, so the loss only ever lands on the JSON we want produced.

bf16 LoRA on a 0.5B base fits one 8 GB card with room to spare — no
quantisation, so nothing here needs nvcc. Pin to card 1 (CUDA_VISIBLE_DEVICES=1)
and the arena keeps card 0.
"""
import json, argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

SYS = (
    "Extract the fields into JSON with exactly these keys: "
    "intent, time, amount, ref, person. "
    "intent is one of meeting, call, payment, reminder, deadline. "
    "time is 24-hour HH:MM or null. amount is a number or null. "
    "ref is a string or null. person is a string or null. "
    "Reply with only the JSON."
)

def to_pc(row):
    # prompt/completion format -> trl applies the chat template and masks prompt
    return {
        "prompt": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": row["text"]},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"])}
        ],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/hf/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="train/1_extract/data")
    ap.add_argument("--out",  default="train/1_extract/adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    ds = load_dataset("json", data_files={
        "train": f"{args.data}/train.jsonl",
        "val":   f"{args.data}/val.jsonl",
    })
    # gold is a dict already from JSON; keep as object for to_pc
    ds = ds.map(lambda r: to_pc(r), remove_columns=ds["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype="bfloat16")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        max_length=320,
        report_to=[],
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    print("saved adapter ->", args.out)

if __name__ == "__main__":
    main()
