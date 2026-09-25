#!/bin/bash
# R10 — cost per token. Measure batched throughput + actual GPU power under load
# for Llama-3-8B (a real general model that generates sustained output, so the
# tok/s number is honest and comparable to cloud API pricing). 2 cards.
cd /root/Desktop/selfhostllm
export PATH=$HOME/.local/bin:$PATH
BIN=engines/llamacpp/build/bin/llama-server
MODEL=models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

CUDA_VISIBLE_DEVICES=1,2 nohup $BIN -m "$MODEL" --host 10.0.0.20 --port 8089 -ngl 99 \
  -c 16384 --no-warmup --parallel 16 -sm layer -a m >/tmp/r10srv.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do curl -s http://10.0.0.20:8089/health 2>/dev/null | grep -q ok && break; sleep 2; done

# sample cards 1+2 power (summed) during the load
( for i in $(seq 1 120); do
    p=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 1,2 | paste -sd+ | bc)
    echo "$p"; sleep 1; done > /tmp/r10pow.csv ) &
POW=$!

echo "--- load (Llama-3-8B Q4, 2 cards, 16 slots) ---"
TPS=$(uv run python serve/loadtest.py --base-url http://10.0.0.20:8089/v1 --model m \
      --levels 16 --reqs-per-conc 8 --max-tokens 256 2>/dev/null | awk '$1==16{print $2}')
kill $POW 2>/dev/null; kill $SRV 2>/dev/null; sleep 2
AVGW=$(awk '{s+=$1;n++} END{printf "%.0f", s/n}' /tmp/r10pow.csv)
echo "measured: sys throughput ${TPS} tok/s, avg power (2 cards) ${AVGW} W"

uv run python - "$TPS" "$AVGW" <<'PY'
import sys
tps=float(sys.argv[1]); gpuW=float(sys.argv[2])
HW=1000.0; YEARS=3; KWH=0.15; HOST_W=100   # host power for a 2-card job
hw_sec=(HW*2/3)/(YEARS*365*24*3600)         # 2 of 3 cards' hardware share
watt=gpuW+HOST_W
elec_sec=watt/1000*KWH/3600
perM=lambda c: c/tps*1e6
print("\n=== cost per 1M tokens (Llama-3-8B Q4, measured on 2 cards) ===")
print(f"throughput            : {tps:.0f} tok/s (batched, 16 slots, 2 cards)")
print(f"power                 : {gpuW:.0f} W GPUs + {HOST_W} W host = {watt:.0f} W")
print(f"marginal (electricity): ${perM(elec_sec):.4f} / 1M tokens")
print(f"fully-loaded (+hw amrt): ${perM(elec_sec+hw_sec):.4f} / 1M tokens")
print(f"  (hw = ${HW:.0f} box / {YEARS}y continuous, 2/3 for two cards; ${KWH}/kWh)")
print("cloud reference — Llama-3-8B class API: ~$0.05-0.20 / 1M output tokens")
PY
echo "=== R10 done ==="
