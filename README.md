# selfhostllm

**Serving and eventually training language models on three second-hand 8 GB GPUs.**

A build log with the numbers left in. Going from not understanding LLM serving at
all, to running one model through five serving engines, to writing an engine by
hand, to training a small model from scratch — on hardware that cost less than a
single datacenter GPU.

Nothing here is a benchmark roundup. **One model is held fixed and the engine is the
variable**, which is the opposite of how these comparisons are usually done and the
reason the numbers mean anything.

## The hardware

```
3x NVIDIA RTX 3060 Ti    8 GB each, 24 GB total, no NVLink
2x Intel Xeon E5-2665    32 threads
125 GB RAM               879 GB disk
Ubuntu 24.04             driver 595.84
```

Used parts. The GPUs are mid-range gaming cards from 2020; the CPUs come from a
retired server generation.

## Measured, not quoted

Every number below was produced by a script in `source/`, on this machine.

| | Spec sheet | **Measured** | |
|---|---|---|---|
| bf16 matrix multiply | ~40 TFLOPS | **36.59 TFLOPS** | 91% of spec |
| memory read | 448 GB/s | **423.9 GB/s** | 95% of spec |
| fp32 matrix multiply | — | 9.85 TFLOPS | bf16 is **3.7x** faster |

Reading a 360M-parameter model's weights once takes **1.71 ms**, which caps that
model at **584 tokens/second** on one card no matter what else is true. At that rate
the arithmetic units are busy **1.2%** of the time — this card multiplies **87x
faster than it can fetch**, and that single fact shapes everything downstream.

## The plan

```
PART I    SERVING     five engines, easiest to hardest      <- here
PART II   THE AGENT   a coding agent, and the harness that scores it
PART III  THE WALLS   KV quantization, long context, multi-GPU
PART IV   OUR MODEL   pretrain a small model from scratch
```

| Level | Engine | What it teaches |
|---|---|---|
| L0 | Ollama | the baseline every later level must match |
| L1 | llama.cpp | **fitting** — quantization, CPU offload, the memory model |
| L2 | vLLM | **throughput** — paged attention, continuous batching |
| L3 | SGLang | **scheduling** — prefix caching, structured output |
| L4 | ours | **understanding** — written by hand, token-identical at temperature 0 |

The exam at L4 is strict on purpose: at temperature zero the hand-written engine must
produce *token-identical* output to the reference. Not similar. Identical.

## Getting started

**[docs/SETUP.md](docs/SETUP.md) has every command, in order, with the traps** — the
service file Ollama's installer may not write, why hiding cards from CUDA is not enough
to pin to one GPU, and the CMake flags that keep a llama.cpp build short.

```bash
git clone <this repo> && cd selfhostllm

# uv handles the environment (https://astral.sh/uv) — not pip
uv venv --python 3.12
uv pip install -e .

# torch is separate: the CUDA build is ~3 GB and pinned to your driver
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

export SELFHOSTLLM_ROOT=$PWD
```

Then, without downloading a single weight:

```bash
# parameters, size at three quantizations, and KV cache per token —
# derived from a config alone. No torch needed.
python3 source/01_budget_from_config.py models/configs/*/config.json
```

```
model                       params     bf16     Q4   KV/tok  shape
Meta-Llama-3-8B      8,030,261,248    15.0G   4.1G     128K  h4096 L32 32/8 ffn14336
Mistral-7B-v0.3      7,248,023,552    13.5G   3.7G     128K  h4096 L32 32/8 ffn14336
Qwen2.5-7B           7,615,487,488    14.2G   3.9G      56K  h3584 L28 28/4 ffn18944
Qwen3-8B             8,190,726,144    15.3G   4.2G     144K  h4096 L36 32/8 ffn12288
```

That KV column is why the table exists. **Qwen2.5-7B needs less than half the
conversation memory of Llama-3-8B** — 28 layers and 4 key/value heads instead of 32
and 8. At 32k context that is 1.75 GiB against 4.00 GiB, which on an 8 GB card
decides whether the model runs at all.

## What is in here

```
source/       the code. NN_<name>.py, numbered by the chapter it produced
models/
  configs/    architecture only, a few KB each — committed
  hf/         real weights — gitignored, fetch them yourself
bench/        the measurement harness (results are committed; logs are not)
docs/         longer-form writing
engines/      one serving stack per level, each with its own venv
```

Scripts are numbered so that **every measured claim is reproducible by running one
file**:

| Script | Establishes |
|---|---|
| `01_budget_from_config.py` | parameter budget from a config alone |
| `02_inspect_model.py` | what is inside a safetensors file, without loading it |
| `03_bench_card.py` | what this GPU actually does |
| `04_embedding_similarity.py` | whether meaning is measurably in the embedding |

`02_inspect_model.py` is the one worth running first. It predicts a model's parameter
count from its config, then reads the real file and compares:

```
predicted    : 361,821,120
actual       : 361,821,120
difference   : 0   EXACT
```

## Method

- **One model, held fixed. The engine is the variable.** Comparisons that change both
  at once measure nothing.
- **Every run emits the same six metrics** — time to first token, single-stream
  tokens/sec, concurrent tokens/sec, peak VRAM per card, context ceiling, and an
  output hash at temperature 0.
- **Predicted before measured.** Each level publishes a prediction first. Where the
  measurement disagrees, the gap is explained rather than the prediction deleted.
- **A number not measured on this machine is labelled an estimate, or it is absent.**
  A vendor specification is not a measurement.

## Status

**Part I has not started.** What exists so far is the groundwork: the hardware
characterised, the file format understood, the arithmetic counted, and four
reproducible scripts. No engine has served a token yet.

The tables above are measurements. The plan below them is a prediction.

## Prior work

The sibling project ran MiniMax-H3 — a 33-billion-parameter video and audio model —
on a single 8 GB card, by finding and chunking the one matrix multiply actually
responsible for the memory wall. It then rendered 15 seconds of native 720p across
three cards using a hand-written sequence-parallel engine. The ring attention built
there is reused in Part III: it is attention, and it does not care what kind of model
it sits inside.
# Selfhost-LLM
# Selfhost-LLM
