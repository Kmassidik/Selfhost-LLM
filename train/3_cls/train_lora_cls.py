#!/usr/bin/env python3
"""Task 3 — LoRA fine-tune Qwen2.5-0.5B for ticket routing (12 classes)."""
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from make_data_cls import LABELS

SYS = ("Classify the support ticket into exactly one category:\n" +
       ", ".join(LABELS) + ".\nReply with only the category.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/hf/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="train/3_cls/data")
    ap.add_argument("--out", default="train/3_cls/adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    ds = load_dataset("json", data_files={"train": f"{args.data}/train.jsonl",
                                          "val": f"{args.data}/val.jsonl"})
    def to_pc(r):
        return {"prompt": [{"role": "system", "content": SYS},
                           {"role": "user", "content": r["text"]}],
                "completion": [{"role": "assistant", "content": r["label"]}]}
    ds = ds.map(to_pc, remove_columns=ds["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype="bfloat16")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs,
                    per_device_train_batch_size=16, learning_rate=2e-4,
                    logging_steps=50, save_strategy="epoch", eval_strategy="epoch",
                    bf16=True, max_length=256, report_to=[], completion_only_loss=True,
                    disable_tqdm=True)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds["train"],
                         eval_dataset=ds["val"], peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    print("saved adapter ->", args.out)

if __name__ == "__main__":
    main()
