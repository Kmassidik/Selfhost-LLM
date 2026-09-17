#!/usr/bin/env python3
"""R4 — QLoRA vs LoRA on the extraction task. Same data, same LoRA config; the
only difference is whether the base is bf16 (LoRA) or 4-bit NF4 (QLoRA). Reports
peak training VRAM so the memory saving is measured, not assumed."""
import json, argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

SYS = ("Extract the fields into JSON with exactly these keys: intent, time, amount, ref, person. "
 "intent is one of meeting, call, payment, reminder, deadline. time is 24-hour HH:MM or null. "
 "amount is a number or null. ref is a string or null. person is a string or null. Reply with only the JSON.")
def to_pc(r):
    return {"prompt": [{"role":"system","content":SYS},{"role":"user","content":r["text"]}],
            "completion": [{"role":"assistant","content":json.dumps(r["gold"])}]}

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="models/hf/Qwen2.5-0.5B-Instruct")
ap.add_argument("--data", default="train/1_extract/data")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--qlora", action="store_true")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
ds = load_dataset("json", data_files={"train": f"{args.data}/train.jsonl"})["train"]
ds = ds.map(to_pc, remove_columns=ds.column_names)

if args.qlora:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="cuda")
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype="bfloat16", device_map="cuda")

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, per_device_train_batch_size=args.bs,
                learning_rate=2e-4, logging_steps=100, save_strategy="no", bf16=True, max_length=320,
                report_to=[], completion_only_loss=True, disable_tqdm=True)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
torch.cuda.reset_peak_memory_stats()
trainer.train()
print(f"PEAK_VRAM {torch.cuda.max_memory_allocated()/1e9:.2f} GB  mode={'QLoRA' if args.qlora else 'LoRA'}  base={args.base.split('/')[-1]}")
trainer.save_model(args.out)
print("saved ->", args.out)
