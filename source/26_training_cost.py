#!/usr/bin/env python3
"""
26_training_cost.py — what training actually costs, measured (ch.26).

The calculator built alongside this guide says a full fine-tune needs about
16 bytes per parameter. That figure is DERIVED — published arithmetic, never
run on this machine. Part V is going to depend on it, so it gets measured.

Memory is sampled at each stage of one training step, because the stages are
where the 16 comes from: the weights, then a gradient for every weight, then
the optimizer's two running averages, then a high-precision master copy.

    uv run source/26_training_cost.py
"""
import json, math, os, sys, time
import torch

MODEL = "/root/Desktop/selfhostllm/models/hf/SmolLM2-360M-Instruct"
dev = torch.device("cuda:0")


def mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(dev) / 1e6


def peak():
    return torch.cuda.max_memory_allocated(dev) / 1e6


def stage(label, before):
    now = mb()
    print(f"  {label:<38}{now:>10,.0f} MB{now-before:>+12,.0f}")
    return now


def main():
    torch.cuda.set_device(dev)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    print("=" * 76)
    print("1 · WHERE THE MEMORY GOES, ONE STAGE AT A TIME")
    print("=" * 76)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    base = mb()
    print(f"  {'stage':<38}{'held':>13}{'added':>12}")
    print("  " + "-" * 63)

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    P = sum(p.numel() for p in m.parameters())
    after_w = stage("weights, 32-bit", base)

    m.gradient_checkpointing_enable()
    m.train()
    ids = torch.randint(0, 40000, (1, 512), device=dev)
    out = m(ids, labels=ids)
    after_f = stage("+ forward pass (activations)", after_w)

    out.loss.backward()
    after_b = stage("+ backward pass (one gradient per weight)", after_f)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    opt.step()
    after_o = stage("+ optimizer state (two averages each)", after_b)

    print("  " + "-" * 63)
    print(f"  parameters: {P:,}")
    print(f"  peak during the step: {peak():,.0f} MB")
    print()
    print(f"  {'component':<38}{'bytes per parameter':>22}")
    print("  " + "-" * 60)
    for label, val in (("weights", after_w - base),
                       ("gradients", after_b - after_f),
                       ("optimizer state", after_o - after_b)):
        print(f"  {label:<38}{val*1e6/P:>22.2f}")
    total = (after_o - base) * 1e6 / P
    print(f"  {'TOTAL, excluding activations':<38}{total:>22.2f}")
    print(f"  {'the calculator predicted':<38}{16.0:>22.2f}")
    print(f"  {'error':<38}{(total-16)/16*100:>21.1f}%")

    # ---- 2 · how fast can this box train? -----------------------------
    print()
    print("=" * 76)
    print("2 · HOW FAST — one card, tokens per second of TRAINING")
    print("=" * 76)
    opt.zero_grad(set_to_none=True)
    for _ in range(2):                       # warm
        m(ids, labels=ids).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0 = time.perf_counter(); N = 5
    for _ in range(N):
        m(ids, labels=ids).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / N
    tps = ids.numel() / per_step
    print(f"  sequence length         {ids.shape[1]}")
    print(f"  time per step           {per_step*1000:,.0f} ms")
    print(f"  training throughput     {tps:,.0f} tokens/s")

    # ---- 3 · what that means for a real run ---------------------------
    print()
    print("=" * 76)
    print("3 · WHAT THAT BUYS, ON THIS BOX")
    print("=" * 76)
    print(f"  {'target':<34}{'tokens':>16}{'one card':>14}{'three cards':>14}")
    print("  " + "-" * 78)
    for label, toks in (("Chinchilla-optimal for 360M", 20 * 361_821_120),
                        ("a 100M-token fine-tune", 100_000_000),
                        ("a 1B-token run", 1_000_000_000)):
        d1 = toks / tps / 86400
        print(f"  {label:<34}{toks:>16,}{d1:>13.1f}d{d1/3:>13.1f}d")
    print()
    print("  The three-card column assumes perfect scaling, which Part II measured")
    print("  as false: data parallelism is the only split that scales, and only for")
    print("  throughput. Treat it as the optimistic bound rather than a plan.")

    json.dump({"params": P, "bytes_per_param": total, "tok_per_s": tps,
               "stages_mb": {"weights": after_w-base, "activations": after_f-after_w,
                             "gradients": after_b-after_f, "optimizer": after_o-after_b}},
              open("/root/Desktop/selfhostllm/bench/results/training-cost.json", "w"), indent=1)


if __name__ == "__main__":
    main()
