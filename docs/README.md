# selfhostllm

**Serving and training language models on three second-hand 8 GB GPUs.**

This is a build log with the numbers left in. It documents an attempt to go from
not understanding LLM serving at all, to running one model through five different
serving engines, to writing an engine by hand, and finally to training a small
model from scratch — all on consumer hardware that cost less than a single
datacenter GPU.

Nothing here is a benchmark roundup. One model is held fixed and the *engine* is
the variable, which is the opposite of how these comparisons are usually done and
the reason the numbers mean anything.

## The hardware

    3x NVIDIA RTX 3060 Ti    8 GB each, 24 GB total, no NVLink
    2x Intel Xeon E5-2665    32 threads
    125 GB RAM               879 GB disk
    Ubuntu 24.04             driver 595.84

Used parts. The GPUs are mid-range gaming cards from 2020; the CPUs are from a
retired server generation.

## The plan

    PART I    SERVING     five engines, easiest to hardest
    PART II   THE AGENT   a coding agent, and the harness that scores it
    PART III  THE WALLS   KV quantization, long context, multi-GPU
    PART IV   OUR MODEL   pretrain a small model from scratch

### Part I — the ladder

| Level | Engine | What it teaches |
|---|---|---|
| L0 | Ollama | the baseline — a reference every later level must match |
| L1 | llama.cpp | **fitting** — quantization, CPU offload, the memory model |
| L2 | vLLM | **throughput** — paged attention, continuous batching, parallelism |
| L3 | SGLang | **scheduling** — prefix caching, structured output |
| L4 | ours | **understanding** — written by hand, token-identical to L0 at temp 0 |

The exam at L4 is strict on purpose: at temperature zero the hand-written engine
must produce *token-identical* output to the reference. Not similar. Identical.
Anything else is a bug wearing a disguise.

## How much fits in 24 GB?

The question everyone asks, answered with arithmetic rather than a rule of thumb.

**Weights** are the easy half:

    weight_bytes = params x bytes_per_param

      params            the count of numbers in the model -- the "8B" in its name
      bytes_per_param   space for ONE of them, set by the number format:
                        fp16 2.0  |  int8 1.0  |  Q4_K_M ~0.55

So 8B at fp16 = 16 GB; the same model at Q4 = ~4.5 GB. The parameter count
never changes -- only the cost of storing each one.

**The KV cache** is the half that surprises people, because it grows with the
length of the conversation:

    bytes_per_token = 2 x layers x kv_heads x head_dim x dtype_bytes

      2            keys AND values -- two things stored per token, always
      layers       the model's repeated blocks; each keeps its own cache  (8B: 32)
      kv_heads     parallel heads that store keys/values                  (8B: 8)
      head_dim     numbers each head stores per token                     (8B: 128)
      dtype_bytes  space per stored number -- 2 for 16-bit, 1 for 8-bit

Five of those six are fixed by whoever designed the model. Only dtype_bytes is
ours to change, which is why quantizing the KV cache is a real technique and not
a footnote.

For an 8B model that is **128 KB per token** — so a 32,000-token context costs
**4.2 GB**, on a card whose weights already took 4.5 GB of 7.6 GB usable.

Doubling the context costs more than doubling the parameters. Almost everything a
serving engine does cleverly — paging, sharing, quantizing, evicting — is aimed at
that one number.

## How fast can it go?

Generating text is limited by memory bandwidth, not arithmetic. Each token
requires reading the model's active weights once:

    tokens_per_sec ~= memory_bandwidth / bytes_read_per_token

      memory_bandwidth        how fast the card reads its own memory (448 GB/s here)
      bytes_read_per_token    the model's ACTIVE weights -- for a dense model that
                              is all of them; for a mixture-of-experts, far fewer

At 448 GB/s per card, an 8B model at Q4 (4.5 GB) has a ceiling near 100 tok/s.
Prefill — processing the prompt — is the opposite, limited by compute, which is
why time-to-first-token and tokens-per-second are unrelated numbers.

## Can you train your own model on hardware you own?

Yes, at a size that honest arithmetic gives rather than optimism.

    training_FLOPs ~= 6 x params x tokens

      6         the standard constant: ~2 FLOPs forward + ~4 backward, per
                parameter per token. Training costs roughly 3x inference.
      tokens    how much text the model is trained on, in total

    training_memory ~= 16 bytes/param

      2 weights (bf16) + 2 gradients (bf16) + 8 Adam state (fp32 m and v)
      + 4 master weights = 16. Note this is EIGHT times the 2 bytes/param an
      fp16 model needs just to run -- which is why a model you can serve
      comfortably may be impossible to train.

| Model | Tokens | Estimated wall-clock on 3x 3060 Ti |
|---|---|---|
| 0.1B | 2B | ~8 hours |
| 0.5B | 10B | ~8 days |
| 1B | 20B | ~33 days |
| 7B | 140B | ~4.5 years — no |

So "our own model" here means **0.1B to 1B, pretrained from scratch**. Not a
frontier model. A real one, on chosen data, scored by a harness built in Part II —
because a model you cannot measure is a model you cannot claim.

*(Those figures assume 35% utilisation, which is a guess until measured. On PCIe
with no NVLink it may be well below that. Measuring it is an early task, and these
numbers will be corrected here when it is.)*

## Method

- **One model, held fixed. The engine is the variable.** Comparisons that change
  both at once measure nothing.
- **Every run emits the same six metrics** — TTFT, single-stream tok/s, concurrent
  tok/s, peak VRAM per card, context ceiling, and an output hash at temperature 0.
- **Predicted before measured.** Each level publishes a prediction first. Where the
  measurement disagrees, the gap is explained rather than the prediction deleted.
- **A number not measured on this machine is labelled an estimate, or it is absent.**

## Status

**Scaffolded. No measurements taken yet.** Started 2026-09-11.

Every table above is currently a prediction. They are published in advance
deliberately — so that being wrong is on the record.

## Prior work

The sibling project [selfhostgenai](https://github.com/) ran MiniMax-H3, a 33B
video+audio model, on a single 8 GB card by chunking the one matmul that was
actually responsible for the memory wall, then rendered 15 seconds of native 720p
across three cards using a hand-written sequence-parallel engine. The ring
attention built there is reused in Part III — it is attention, not video, and it
does not care what kind of model it is inside.
