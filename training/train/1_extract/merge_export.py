#!/usr/bin/env python3
"""Task 1 — merge the LoRA adapter into the base and save a standalone HF model.

A LoRA adapter is a diff; llama.cpp wants whole weights. merge_and_unload folds
the adapter back into the base so the result is an ordinary model, ready for
convert_hf_to_gguf.py. Output goes to models/hf/<name>-merged.
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="models/hf/Qwen2.5-0.5B-Instruct")
ap.add_argument("--adapter", default="train/1_extract/adapter")
ap.add_argument("--out", default="models/hf/Qwen2.5-0.5B-extract-merged")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, dtype="float16")
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print("merged ->", args.out)
