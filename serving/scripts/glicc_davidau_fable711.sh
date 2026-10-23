#!/usr/bin/env bash
# glicc endpoint (:8081) on DavidAU Fable-Fusion-711 (Qwen3.6-27B dense, abliterated/uncensored
# lineage, Q4_K_M) with vision projector, split across cards 0, 1 AND 2. Comparison candidate,
# not the default - switch to it from the dashboard or by running this script directly.
# First-boot defaults are conservative (small ctx, 4-bit cache) because this model's dense 27B
# KV cache is heavy (~18 KB/token, derived) against ~19 GB of weights+projector already on the
# cards; raise CTX only after confirming no OOM at the default.
set -u
cd /root/Desktop/selfhostllm
TOK=$(cat glicc-api.token)
CTX=${CTX:-32768}
UB=${UB:-256}
CACHE=${CACHE:-q4_0}
TS=${TS:-}
EXTRA=${EXTRA:-}
D=models/gguf/davidau-fable711-27b

fuser -k 8081/tcp 2>/dev/null; sleep 6
for i in $(seq 1 20); do fuser 8081/tcp >/dev/null 2>&1 || break; sleep 1; done

TSFLAG=""
if [ -n "$TS" ]; then TSFLAG="--tensor-split $TS"; fi

CUDA_VISIBLE_DEVICES=0,1,2 setsid nohup engines/llamacpp/build/bin/llama-server \
  -m $D/Fable-Fusion-711-Q4_K_M.gguf --mmproj $D/mmproj-F16.gguf \
  -ngl 99 --split-mode layer -ub $UB -fa on -ctk $CACHE -ctv $CACHE -c "$CTX" --parallel 1 --cache-reuse 256 \
  $TSFLAG $EXTRA --jinja --temp 1.0 --top-p 0.95 --top-k 20 \
  --api-key "$TOK" -a davidau-fable711-27b --host 127.0.0.1 --port 8081 \
  > logs/glicc_srv.log 2>&1 < /dev/null &

for i in $(seq 1 100); do
  sleep 3
  if curl -s -m3 http://127.0.0.1:8081/health | grep -q ok; then
    echo "DAVIDAU_UP cards 0+1+2, ctx $CTX, cache $CACHE"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  pgrep -f "[l]lama-server.*8081" >/dev/null || break
done
echo "DAVIDAU_FAILED"; grep -iE "error|memory|fail" logs/glicc_srv.log | tail -10; exit 1
