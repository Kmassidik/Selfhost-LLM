# selfhostllm - BOX WORKSPACE

This is the **runtime** half of the project. It holds everything too big, too
machine-specific, or too disposable to live in git: weights, compiled engines,
virtualenvs, run artifacts, datasets, checkpoints.

The **source** half lives on the Mac and is the git repo: PRD, plans, harness
code, our own engine, knowledge base.

    Mac  ~/Desktop/selfhostllm      source of truth, git, synced
    Box  /root/Desktop/selfhostllm  weights + runs + venvs, never synced

Lesson inherited from selfhostgenai: roughly 25 files there hardcode
/root/Desktop/selfhosted-minimaxi-h3, which made the repo unusable elsewhere.
This time paths come from $SELFHOSTLLM_ROOT (default: this directory), never
from a literal.

## Layout

    tools/        SYNCED FROM THE MAC — do not edit here, changes are overwritten
    models/
      hf/         full models — weights present, can actually be run
      configs/    architecture only (a few KB) — cannot run, only reasoned about
      gguf/       llama.cpp format
    engines/      one serving stack per level, each with its OWN venv
                  (vLLM and SGLang pin conflicting torch versions - a shared
                   venv WILL break. This is not paranoia.)
    bench/        the measurement harness
      prompts/    the FIXED prompt set - the experimental control
      results/    one JSON per run - the comparison table's source data
    runs/         raw artifacts per run: stdout, nvidia-smi samples, outputs
    data/         Part IV: datasets + tokenizer
    checkpoints/  Part IV: training checkpoints (resumability is a feature)
    scratch/      disposable

## What is here now

    models/hf/SmolLM2-360M-Instruct/     690 MiB, bf16, LlamaForCausalLM
                                         ch.02's subject: 361,821,120 params in 290 tensors
    models/configs/                      Meta-Llama-3-8B, Qwen2.5-7B, Qwen3-8B,
                                         Mistral-7B-v0.3 — 76 KB total, no weights
    models/configs/fetch.sh              ./fetch.sh <org>/<model> to add another

## Running the tools

`tools/` is pushed from the Mac by `./sync.sh` and is **one-way**: anything edited
here is destroyed on the next sync. Edit on the Mac, sync, run here.

    export SELFHOSTLLM_ROOT=/root/Desktop/selfhostllm
    V=/root/Desktop/selfhosted-minimaxi-h3/.venv/bin/python   # has torch + numpy

    # what is inside a model file, without loading it
    $V tools/inspect_model.py models/hf/SmolLM2-360M-Instruct/model.safetensors \
        --predict models/hf/SmolLM2-360M-Instruct/config.json

    # parameters, size and KV cache from a config alone (plain python3 is enough)
    python3 tools/budget_from_config.py models/configs/*/config.json

    # what this GPU actually does
    $V tools/bench_card.py

    # is meaning really in the embedding numbers?
    $V tools/probe_embedding.py

## Rules

1. Nothing here is a source of truth except measurements.
2. Every run writes bench/results/<run-id>.json. No JSON, no run.
3. Never stream a long job over SSH - launch detached, return.
4. runs/ and checkpoints/ grow without limit. Watch the 585 GB.
