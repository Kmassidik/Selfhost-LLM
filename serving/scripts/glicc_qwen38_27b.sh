#!/usr/bin/env bash
# glicc endpoint (:8081) on Qwen3.8-27B (ISTA-DASLab GSQ-RCO IQ3_S, 3.5 bits/weight) with the
# vision projector, split across CARDS 1 AND 2. Card 0 stays with the music engine.
# Replaces glicc_bonsai_card2.sh (Bonsai stopped 2026-10-12). Same port, key; model id qwen3.8-27b.
# Numbers for this setup (speed, context) are ESTIMATES until measured - see knowledge base.
set -u
cd /root/Desktop/selfhostllm
TOK=$(cat glicc-api.token)
CTX=${CTX:-32768}
UB=${UB:-256}
EXTRA=${EXTRA:-}
D=models/gguf/qwen38-27b-gsq

fuser -k 8081/tcp 2>/dev/null; sleep 6
for i in $(seq 1 20); do fuser 8081/tcp >/dev/null 2>&1 || break; sleep 1; done

CUDA_VISIBLE_DEVICES=1,2 setsid nohup engines/llamacpp/build/bin/llama-server \
  -m $D/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf --mmproj $D/mmproj-Qwen3.8-27B-BF16.gguf \
  -ngl 99 --split-mode layer -ub $UB -fa on -ctk q8_0 -ctv q8_0 -c "$CTX" --parallel 1 --cache-reuse 256 \
  $EXTRA --jinja --temp 1.0 --top-p 0.95 --top-k 20 \
  --api-key "$TOK" -a qwen3.8-27b --host 127.0.0.1 --port 8081 \
  > logs/glicc_srv.log 2>&1 < /dev/null &

for i in $(seq 1 100); do
  sleep 3
  if curl -s -m3 http://127.0.0.1:8081/health | grep -q ok; then
    echo "QWEN38_UP cards 1+2, ctx $CTX"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  pgrep -f "[l]lama-server.*8081" >/dev/null || break
done
echo "QWEN38_FAILED"; grep -iE "error|memory|fail" logs/glicc_srv.log | tail -6; exit 1
