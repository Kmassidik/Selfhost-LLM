#!/usr/bin/env python3
"""Stress the L4b forward pass: every frozen prompt, every position.

A single prompt agreeing proves little. The logits differ by up to 0.36, which
is enough to flip an argmax wherever the top two candidates are close together.
This measures how close they actually get.
"""
import json, sys, os, torch
sys.path.insert(0, "/root/Desktop/selfhostllm/source")
import importlib.util
spec = importlib.util.spec_from_file_location("fwd", "/root/Desktop/selfhostllm/source/15b_forward_pass.py")
fwd = importlib.util.module_from_spec(spec); spec.loader.exec_module(fwd)

MODEL = fwd.MODEL
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
W, cfg = fwd.load(dev)

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()

prompts = json.load(open("/root/Desktop/selfhostllm/bench/prompts.json"))
cases = [(p["id"], p["text"]) for p in prompts["prompts"]]

print(f"{'prompt':<10}{'tokens':>8}{'agree':>10}{'max dlogit':>13}{'min margin':>13}  verdict")
print("-" * 70)
bad = 0
for name, text in cases:
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    if ids.shape[1] > 1024:
        ids = ids[:, :1024]
    with torch.no_grad():
        ours = fwd.forward(ids, W, cfg, dev)
        ref = ref_model(ids).logits
    d = (ours.float() - ref.float()).abs().max().item()
    a = (ours[0].argmax(-1) == ref[0].argmax(-1))
    agree = a.sum().item(); n = ids.shape[1]
    # how close is the race at each position? top1 - top2 on the reference
    top2 = ref[0].float().topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).min().item()
    ok = agree == n
    bad += (not ok)
    print(f"{name:<10}{n:>8}{agree:>7}/{n:<3}{d:>13.3e}{margin:>13.3e}  {'ok' if ok else 'DIVERGES'}")

print("-" * 70)
print("max dlogit = worst logit gap ours vs reference")
print("min margin = smallest gap between the top two candidates on the reference;")
print("             an argmax flips when max dlogit exceeds it")
sys.exit(1 if bad else 0)
