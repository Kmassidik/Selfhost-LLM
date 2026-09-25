# R8 — Multi-adapter serving (the mini-ixCSP core)

**Question:** how many custom SLMs can one card serve off a single shared base,
and at what cost to accuracy and latency? **Skill:** multi-LoRA serving, MaaS density.

## Method
Load base Qwen2.5-0.5B once; attach the three task adapters (extract/sql/cls) with
peft `load_adapter`; route per request with `set_adapter`. Measure VRAM, switch
latency, and retained accuracy on 100 hard-set rows per task. Safetensors/LoRA
path (the faithful one — see R7b, GGUF degrades these).

## Result

| task | routed via shared base | single-adapter baseline |
|---|---|---|
| extract | 100% EXACT | 99.5% |
| sql | 99% exec-match | 99% |
| cls | 87% macro-F1 | 86.8% |

- **Accuracy fully retained** — routing costs nothing.
- **Switch latency:** 24 ms mean (`set_adapter`).
- **Memory:** base 0.99 GB + **~35 MB per adapter**.

## Density (weight memory)

| models | multi-adapter | separate | denser |
|---|---|---|---|
| 3 | 1.09 GB | 3.0 GB | 2.7× |
| 10 | 1.34 GB | 9.9 GB | 7.4× |
| 20 | 1.69 GB | 19.8 GB | 11.7× |
| 50 | 2.75 GB | 49.4 GB | 18× |

**~50 custom SLMs fit in one 8 GB card's weight budget (2.75 GB)** where separate
models would need 49 GB — seven cards.

## Pass/fail
Bar: "≥3 adapters/base, no acc loss." **Pass decisively** — 3 adapters, zero acc
loss, projects to 50+.

## Caveats / follow-ups
- This is *weight-memory* density (route + serve). Concurrent serving adds KV
  cache per in-flight request; the card's real ceiling is weights + KV.
- 24 ms per-request `set_adapter` is fine for routed single-stream but wasteful
  under load. Production wants **batched multi-LoRA** (vLLM `--enable-lora`,
  S-LoRA) which batches same-adapter requests. That's **R9** (hot-swap latency)
  and the vLLM path.

## Product
The "sell N custom models" thesis is now concrete: one cheap card hosts ~50
metered specialists. This is ixCSP's MaaS/marketplace density, on hardware
INFINITIX would price at datacenter GPUs. Artifacts: `train/r8_multiadapter.py`.
