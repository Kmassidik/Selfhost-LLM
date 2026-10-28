#!/usr/bin/env bash
# glicc endpoint :8081 — Qwen3.8-35B-A3B uncensored MoE, 256K, MTP.
# Weights: IsValorum APEX-I-MiniPlus V2.1 Abliterated (13.65 GiB)
# MTP: separate Q4_0 draft head, max 2 draft tokens.
set -u
cd /root/Desktop/selfhostllm
TOK=$(cat glicc-api.token)
D=models/gguf/qwen38-35b-a3b
CTX=${CTX:-256000}
UB=${UB:-32}
TS=${TS:-1,1,0.9}

fuser -k 8081/tcp 2>/dev/null; sleep 6
for i in $(seq 1 20); do fuser 8081/tcp >/dev/null 2>&1 || break; sleep 1; done

TSFLAG=""
if [ -n "$TS" ]; then TSFLAG="--tensor-split $TS"; fi

CUDA_VISIBLE_DEVICES=0,1,2 setsid nohup /root/Desktop/competitors/llama-upstream/build/bin/llama-server \
  -m $D/Qwen3.8-35B-A3B-Distill.APEX-I-MiniPlus-V2.1-Abliterated.gguf \
  -md $D/mtp-Qwen3.8-35B-A3B-Distill-Q4_0.gguf \
  --mmproj $D/mmproj-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 --split-mode layer $TSFLAG \
  -ub "$UB" -fa on \
  -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 \
  -c "$CTX" --parallel 1 -t 16 --cache-reuse 256 \
  --no-mmproj-offload --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --api-key "$TOK" -a qwen3.8-moe --host 127.0.0.1 --port 8081 \
  > logs/glicc_qwen38moe.log 2>&1 < /dev/null &

for i in $(seq 1 100); do
  sleep 3
  if curl -s -m3 http://127.0.0.1:8081/health | grep -q ok; then
    echo "QWEN38_MOE_UP ctx $CTX, ub $UB, ts $TS"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  pgrep -f "[l]lama-server.*8081" >/dev/null || break
done
echo "QWEN38_MOE_FAILED"; grep -iE "error|memory|fail" logs/glicc_qwen38moe.log | tail -10; exit 1
