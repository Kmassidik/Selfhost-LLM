# selfhostllm

**Serving, stressing, and training language models on three second-hand 8 GB GPUs — with every number measured on the box.**

A build log with the failures left in. It starts at "what is a model file" and ends
serving a 27-billion-parameter ternary model as a token-gated, prefix-cached API — on
gaming cards from 2020 and a retired server's CPUs, for less than one datacenter GPU.

Along the way: one model pushed through five serving engines, an engine written by hand
that must match the reference token-for-token, a 0.5B model fine-tuned to beat a 30B,
50 custom models packed onto one card, and a GPT, a diffusion model and an audio model
built from scratch.

> **The rule:** a number not measured on this machine is labelled an estimate, or it is
> absent. A vendor spec sheet is not a measurement.

---

## The machine

```
3x NVIDIA RTX 3060 Ti    8 GB each, 24 GB total, PCIe only (no NVLink)
2x Intel Xeon E5-2665    32 threads, no AVX2
125 GB RAM               879 GB disk
Ubuntu 24.04             CUDA driver 595.84
```

| | Spec sheet | **Measured** | |
|---|---|---|---|
| bf16 matrix multiply | ~40 TFLOPS | **36.59 TFLOPS** | 91% of spec |
| memory read | 448 GB/s | **423.9 GB/s** | 95% of spec |
| fp32 matrix multiply | — | 9.85 TFLOPS | bf16 is **3.7×** faster |

This card multiplies **87× faster than it can fetch** from memory. Reading a 360M model's
weights once takes 1.71 ms, which caps it at 584 tokens/s on one card whatever else is
true — and the arithmetic units sit idle 98.8% of that time. Almost every result below
is a consequence of that one ratio.

---

## Highlights

| Result | Measured | Where |
|---|---|---|
| Tuned **0.5B beats a 30B** on strict extraction | **99.5% vs 17.8%** | `training/train/` |
| …and loses on open-ended classification | 86.8% vs **99.0%** | `training/train/` |
| **~50 fine-tuned models on one 8 GB card** (LoRA adapters) | 18× denser than separate models | R8 |
| Cost to serve, on the box | **$0.03–0.05 per 1M tokens** | `serving/serve/R10_RESULTS.md` |
| Naive multi-model serving without an adapter-aware scheduler | **~30× slower**, 44× unfair | R9, `R11_R12_RESULTS.md` |
| A load test's "ceiling" was the Python client, not the GPU | 295 → **8,378 tok/s** native | R22c |
| GPU utilisation as an autoscaling signal | read **13% while overloaded** — use queue depth | `R24_RESULTS.md` |
| RAG vs fine-tuning for knowledge | RAG **2.6×** better | R3, `training/rag/` |
| GGUF export silently erased a fine-tune | adapter 39% ≈ base 40.5% | R7b |
| **27B ternary model (1.75-bit) served on two cards** | 5.95 GB, **~33 tok/s** | `research/r25/` |
| Its context window on two 8 GB cards | 262k full speed · **1M loads** (~1 tok/s) | `docs/frontier-models.md` |
| Speculative decoding on it | n-gram **51–53 tok/s** · draft model *slower* | `research/r28/` |
| A GPT trained from scratch on 1 vs 3 GPUs | 81,886 → 216,138 tok/s (**2.64×**) | `from-scratch/` |

Each row has a script that reproduces it. Negative results stay in: a draft model that
made decoding slower, a ternary shortcut that broke a model 7,000,000×, a prediction for
multi-GPU scaling that missed.

---

## The arc

```
Foundations   what a model is, what fits in memory, how fast it can possibly go
Part I        one card, five engines — Ollama → llama.cpp → vLLM → SGLang → our own
Part II       three cards over a slow link — tensor, pipeline, data parallelism, measured
Part III      the agent — an OpenAI-compatible endpoint, a coding agent, and its evaluation
Part IV       the walls — KV-cache quantization, long context, speculative decoding
Part V        our model — what training costs, the data, the run
Part VI       the frontier — 2026 models on hardware that should not run them
Part VII      the product — fine-tuning small models worth selling; 50 per card
Part VIII     under the hood — the source-code architecture of llama.cpp, vLLM, SGLang
Part IX       in practice — RAG, fine-tuning mechanics, agent loops, graph agents, typed decisions
R&D           28 falsifiable experiments, each with a pass/fail bar
```

The five-engine ladder is the spine of Part I:

| Level | Engine | What it teaches |
|---|---|---|
| L0 | Ollama | the baseline every later level must match |
| L1 | llama.cpp | **fitting** — quantization, CPU offload, the memory model |
| L2 | vLLM | **throughput** — paged attention, continuous batching |
| L3 | SGLang | **scheduling** — prefix caching, structured output |
| L4 | ours | **understanding** — written by hand, token-identical at temperature 0 |

The L4 exam is strict on purpose: at temperature zero the hand-written engine must produce
*identical* tokens to the reference. Not similar. Identical.
(`source/15e_engine.py`, `source/15e_exam.py`)

The long-form write-up of every part — 43 chapters plus an R&D lab notebook — lives in a
separate knowledge base; this repo holds the code and the results files behind it.

---

## The frontier track

The newest work verifies and serves a shipped 2026 model rather than trusting its card.

- **Verify** (`research/r25/`) — Ternary Bonsai 2 27B: 5.95 GB at 1.75 bits per weight,
  bit-identical pack, claims checked line by line. Its prebuilt CUDA binary crashed on this
  CPU (no AVX2), so the engine is built from source.
- **Why ternary is hard** (`research/r26/`) — naive rounding to {−1, 0, +1} collapses a model
  7,000,000×; packing is a free ~10× size win but *slower* without a fused kernel.
- **Typed decisions** (`research/r27/`) — typed output plus a confidence-and-abstain gate
  lifts accuracy on answered items from 63% to 89.5%.
- **Serve it** (`deploy/`, `docs/frontier-models.md`) — OpenAI-compatible, bearer-token
  gated, 4 concurrent slots, prefix caching (a repeated 3k-token context: time to first
  token 4.54 s → 1.07 s), exposed through a Cloudflare tunnel with no open port.
- **Chase 50 tok/s** (`research/r28/`) — batching is step-shaped on the ternary kernel;
  free n-gram speculation reaches 51–53 tok/s on code edits; a draft model loses.

---

## From scratch

`from-scratch/` builds the three kinds of generative model by hand — no model classes imported:

| Part | Model | Result on one 3060 Ti |
|---|---|---|
| A | 10.8M-parameter GPT, char-level, TinyStories | loss 4.65 → 0.92; coherent stories in ~4 min |
| B | DDPM diffusion, MNIST | recognisable digits in ~105 s |
| C | autoregressive audio | generates a real 330 Hz (E4) melody note |

Part A also runs data-parallel on all three cards: 2.64× on 3 GPUs, 88% efficiency — the
missing 12% is gradient all-reduce over PCIe.

---

## Getting started

**[docs/SETUP.md](docs/SETUP.md) has every command in order, with the traps** — the service
file Ollama's installer may not write, why hiding cards from CUDA is not enough to pin to
one GPU, and the CMake flags that keep a llama.cpp build short.

```bash
git clone https://github.com/Kmassidik/Selfhost-LLM.git selfhostllm && cd selfhostllm

# uv manages the environment (https://astral.sh/uv) — not pip
uv venv --python 3.12
uv pip install -e .

# torch is separate: the CUDA build is ~3 GB and pinned to your driver
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

export SELFHOSTLLM_ROOT=$PWD
```

Then, without downloading a single weight:

```bash
# parameters, size at three quantizations, and KV cache per token — from a config alone
python3 source/01_budget_from_config.py models/configs/*/config.json
```

```
model                       params     bf16     Q4   KV/tok  shape
Meta-Llama-3-8B      8,030,261,248    15.0G   4.1G     128K  h4096 L32 32/8 ffn14336
Mistral-7B-v0.3      7,248,023,552    13.5G   3.7G     128K  h4096 L32 32/8 ffn14336
Qwen2.5-7B           7,615,487,488    14.2G   3.9G      56K  h3584 L28 28/4 ffn18944
Qwen3-8B             8,190,726,144    15.3G   4.2G     144K  h4096 L36 32/8 ffn12288
```

The KV column is why the table exists: **Qwen2.5-7B needs less than half the conversation
memory of Llama-3-8B** (28 layers and 4 key/value heads, against 32 and 8). At 32k context
that is 1.75 GiB against 4.00 GiB — on an 8 GB card, the difference between running and not.

Next, `source/02_inspect_model.py` predicts a model's parameter count from its config, then
reads the real file and compares:

```
predicted    : 361,821,120
actual       : 361,821,120
difference   : 0   EXACT
```

Model downloads are covered in [docs/DOWNLOADING-MODELS.md](docs/DOWNLOADING-MODELS.md).

---

## What is in here

```
source/        chapter code, numbered by chapter: NN_<name>.py (01 → 30)
serving/
  serve/       serving experiments + R*_RESULTS.md (batching, prefix cache, KV quant, autoscale)
  bench/       the six-metric measuring harness (run.py); results committed
  scripts/     launch engines, drive load, measure the GPUs
training/
  train/       fine-tuning: LoRA, QLoRA, quantization, distillation
  rag/         retrieval over the guide's own pages (numpy cosine, no vector DB)
research/      frontier R&D: r25 verify · r26 ternary QAT · r27 typed decisions · r28 speed
from-scratch/  a GPT, a diffusion model, an audio model — by hand
deploy/        expose an endpoint through a Cloudflare tunnel
engines/       per-engine wrappers; third-party clones are git-ignored (see engines/SOURCES.md)
models/        configs/ committed (a few KB); weights git-ignored
agent-tasks/   the four coding tasks the agent is scored on
docs/          SETUP, model downloads, frontier-model notes
```

A few scripts that carry the spine:

| Script | Establishes |
|---|---|
| `01_budget_from_config.py` | parameter and memory budget from a config alone |
| `02_inspect_model.py` | what is inside a safetensors file, without loading it |
| `06_box_survey.py` | what this GPU actually does, versus its spec sheet |
| `08_batching.py` | where throughput comes from |
| `15e_engine.py` / `15e_exam.py` | the hand-written engine and its token-identical exam |
| `18_parallelism_measured.py` | multi-GPU, and why the prediction missed |
| `23_kv_quant.py` / `24_context.py` | the KV-cache and long-context walls |
| `26_training_cost.py` / `28_the_run.py` | the cost of training, and the run itself |

---

## Method

- **One model held fixed; the engine is the variable.** A comparison that changes both
  measures nothing.
- **Every serving run emits the same six metrics** — time to first token, single-stream
  tokens/s, concurrent tokens/s, peak VRAM per card, context ceiling, and a
  temperature-0 output hash (so "did the engines compute the same thing?" is answered,
  not assumed).
- **Predicted before measured.** Where the measurement disagrees, the gap is explained —
  the prediction is not quietly deleted.
- **Prove the harness isn't the bottleneck.** Every load result is checked against the
  engine's native benchmark; once, the "GPU ceiling" turned out to be the test client.
- **Failures are results.** Negative and null outcomes are written up with the same care.

---

## Prior work

The sibling project ran MiniMax-H3 — a 33-billion-parameter video-and-audio model — on a
single 8 GB card, by finding and chunking the one matrix multiply responsible for the memory
wall, then rendered 15 seconds of native 720p across three cards with a hand-written
sequence-parallel engine. The ring attention built there is reused here: it is attention,
and it does not care what kind of model it sits inside.
