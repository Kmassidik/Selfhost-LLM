#!/usr/bin/env python3
"""R3 — fine-tune the 0.5B closed-book on the KB QA (Q -> short answer)."""
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

SYS = "Answer the question about a self-hosted-LLM field guide with a short, direct answer."
BASE = "models/hf/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
ds = load_dataset("json", data_files={"train": "rag/data/qa_train.jsonl"})["train"]
def to_pc(r):
    return {"prompt": [{"role": "system", "content": SYS}, {"role": "user", "content": r["q"]}],
            "completion": [{"role": "assistant", "content": r["a"]}]}
ds = ds.map(to_pc, remove_columns=ds.column_names)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype="bfloat16")
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
cfg = SFTConfig(output_dir="rag/r3_adapter", num_train_epochs=5, per_device_train_batch_size=16,
                learning_rate=2e-4, logging_steps=50, save_strategy="epoch", bf16=True,
                max_length=256, report_to=[], completion_only_loss=True, disable_tqdm=True)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
trainer.train()
trainer.save_model("rag/r3_adapter")
print("saved -> rag/r3_adapter")
