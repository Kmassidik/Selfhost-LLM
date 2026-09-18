# source/ — the code. This is where it lives.

**The box is authoritative for code.** It is edited here, it runs here, and the Mac
holds a copy only so it can be committed to git. Anything written on the Mac that
was not pulled from here will be overwritten.

## Naming

    NN_<what_it_does>.py

`NN` is the knowledge-base chapter the script produced the numbers for. Every claim
in a chapter should be reproducible by running the matching file — that is the point
of the numbering, and it is the convention inherited from the sibling project's
`dalang/` engine (`00_probe_load.py`, `01_baseline_single_gpu.py`, and so on).

| File | Chapter | What it establishes |
|---|---|---|
| `01_budget_from_config.py` | ch.01 | parameter budget, size and KV cache from a config alone |
| `02_inspect_model.py` | ch.02 | what is inside a safetensors file, without loading it |
| `03_bench_card.py` | ch.03 | what this GPU actually does — matmul and bandwidth |
| `04_embedding_similarity.py` | ch.04 | whether meaning is measurably present in the embedding |

Numbers may repeat when a chapter needs several scripts — `04a_`, `04b_` — and a
script supporting no chapter yet gets the number of the chapter it will support.

## Running

    export SELFHOSTLLM_ROOT=/root/Desktop/selfhostllm
    V=/root/Desktop/selfhosted-minimaxi-h3/.venv/bin/python   # torch + numpy + tokenizers

    python3 source/01_budget_from_config.py models/configs/*/config.json
    $V source/02_inspect_model.py models/hf/SmolLM2-360M-Instruct/model.safetensors \
        --predict models/hf/SmolLM2-360M-Instruct/config.json
    $V source/03_bench_card.py
    $V source/04_embedding_similarity.py

## Rules

1. **No hardcoded paths.** Take a path argument, or read `$SELFHOSTLLM_ROOT`. The
   sibling project hardcoded `/root/Desktop/selfhosted-minimaxi-h3` into ~25 files
   and said so in its own README as the reason nobody else could clone it.
2. **Never stream a long job over SSH.** Launch detached and return.
3. A script that produced a number in a chapter must keep working, or the chapter
   is no longer reproducible.
