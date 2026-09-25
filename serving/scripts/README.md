# scripts/ — serving & benchmarking harness

Operational scripts used to launch engines, drive load, and measure the box
during the R&D program. Results they produced live in `../serve/` and `../bench/`.

## Serving launchers
- `run_lcpp.sh` — launch a llama.cpp `llama-server` with a chosen model/config
- `run_vllm.sh` — launch a vLLM OpenAI-compatible server
- `run_vllm_pp.sh` — launch vLLM with pipeline parallelism across cards

## Load drivers
- `drive_lcpp.sh` — send benchmark load to the llama.cpp server
- `drive_vllm.sh`, `drive_vllm2.sh` — send benchmark load to vLLM
- `stop_drivers.sh` — stop any running load drivers

## Measurement
- `bench_helpers.sh` — shared shell helpers (wait-for-ready, etc.)
- `measure.sh` — TTFT + decode tok/s from streaming completions
- `ctx_speed.py` — decode/prefill speed vs context depth
- `summ.py` — summarise raw benchmark output

## GPU monitoring
- `gpusample.sh` — sample GPU utilisation / memory over time
- `gpusum.sh` — summarise the GPU samples

## Model download
- `dl_empero.py` — download the Empero Qwen3.8-35B-A3B GGUF from Hugging Face
- `dl_watchdog.sh` — restart a stalled download

## Note on secrets
`ctx_speed.py` reads the API token from `/root/model-api.token` at runtime.
That token file is a secret and is intentionally NOT in this repo — create it
locally before running.
