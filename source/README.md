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

**The chapter plan these numbers refer to** (`knowledge-base/assets/app.js` is the
authoritative list):

    Foundations       01-08   what a model is, through to speed limits
    I  One card       09-14   five engines, a single GPU, one variable
    II Many cards     15-18   parallelism over PCIe — NOT NVLink
    III The agent     19-21   endpoint, agent loop, evaluation harness
    IV The walls      22-24   KV quantization, long context, speculative decoding
    V  Our model      25-27   the long nights

**Part I is one card by constraint.** The engine is pinned to card 0 and the others are
made invisible to it — through *both* CUDA and Vulkan, because hiding them from one is not
enough. `bench/run.py` samples memory on all three cards, so a run that quietly spread
would show up in its own results.

| File | Chapter | What it establishes |
|---|---|---|
| `01_budget_from_config.py` | ch.01, ch.07 | parameter budget, size and KV cache from a config alone |
| `02_inspect_model.py` | ch.02 | what is inside a safetensors file, without loading it |
| `02b_tensor_slices.py` | ch.02 | the real scalar/vector/matrix values in the chapter's diagram |
| `02c_strides.py` | ch.02 | why a grid is a line: strides, and the free transpose |
| `03_bench_card.py` | ch.03, ch.08 | what this GPU actually does — matmul and bandwidth |
| `04_embedding_similarity.py` | ch.04 | whether meaning is measurably in the embedding |
| `05_prefill_vs_decode.py` | ch.05 | the two phases of generation, timed separately |
| `05b_why_cache.py` | ch.05 | what the KV cache is worth, at four context lengths |
| `06_box_survey.py` | ch.06 | every hardware fact the chapter claims, in one pass |
| `08_batching.py` | ch.08 | whether batching is free, batch 1 to 64 |
| `../bench/run.py` | ch.09 onward | **every serving engine**, six metrics, one JSON per run |

Chapters 09 and later are measured by `bench/`, not by a numbered script: the whole point of
Part I is that every engine is measured by *the same* instrument, so a per-chapter script
would defeat it.

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

## Environment — uv, never pip

Packages are installed with [uv](https://astral.sh/uv), not pip. `pyproject.toml` at
the repo root declares what is needed, so the environment is **stated rather than
remembered**.

    uv venv --python 3.12
    uv pip install -e .

    # torch is separate on purpose: the CUDA build is ~3 GB and is pinned to this
    # machine's driver, so it is not a plain dependency
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128

Adding a package means **editing `pyproject.toml` and re-syncing** — not running an
install command and hoping it is remembered. A dependency that only exists in
somebody's shell history is a dependency that breaks the next clone.

## Rules

0. **uv, never pip.** See above.
1. **No hardcoded paths.** Take a path argument, or read `$SELFHOSTLLM_ROOT`. The
   sibling project hardcoded `/root/Desktop/selfhosted-minimaxi-h3` into ~25 files
   and said so in its own README as the reason nobody else could clone it.
2. **Never stream a long job over SSH.** Launch detached and return.
3. A script that produced a number in a chapter must keep working, or the chapter
   is no longer reproducible.
