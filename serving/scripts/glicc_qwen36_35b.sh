#!/usr/bin/env bash
# glicc endpoint (:8081) on Qwen3.6-35B-A3B (unsloth UD-IQ3_S, MoE 35B/3B active) with the
# vision projector, split across cards 0, 1 AND 2 (image/music services disabled on 2026-10-23,
# all three cards free). Replaces glicc_qwen38_27b.sh for this A/B comparison round.
# Numbers for this setup (speed, context) are being measured fresh in R31 - see knowledge base.
set -u
cd /root/Desktop/selfhostllm
TOK=$(cat glicc-api.token)
CTX=${CTX:-262144}
UB=${UB:-256}
CACHE=${CACHE:-q4_0}
TS=${TS:-}
EXTRA=${EXTRA:-}
D=models/gguf/qwen36-35b-a3b

fuser -k 8081/tcp 2>/dev/null; sleep 6
for i in $(seq 1 20); do fuser 8081/tcp >/dev/null 2>&1 || break; sleep 1; done

TSFLAG=""
if [ -n "$TS" ]; then TSFLAG="--tensor-split $TS"; fi

CUDA_VISIBLE_DEVICES=0,1,2 setsid nohup engines/llamacpp/build/bin/llama-server \
  -m $D/Qwen3.6-35B-A3B-UD-IQ3_S.gguf --mmproj $D/mmproj-F16.gguf \
  -ngl 99 --split-mode layer -ub $UB -fa on -ctk $CACHE -ctv $CACHE -c "$CTX" --parallel 1 --cache-reuse 256 \
  $TSFLAG $EXTRA --jinja --temp 1.0 --top-p 0.95 --top-k 20 \
  --api-key "$TOK" -a qwen3.6-35b-a3b --host 127.0.0.1 --port 8081 \
  > logs/glicc_srv.log 2>&1 < /dev/null &

for i in $(seq 1 100); do
  sleep 3
  if curl -s -m3 http://127.0.0.1:8081/health | grep -q ok; then
    echo "QWEN36_UP cards 0+1+2, ctx $CTX, cache $CACHE"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  pgrep -f "[l]lama-server.*8081" >/dev/null || break
done
echo "QWEN36_FAILED"; grep -iE "error|memory|fail" logs/glicc_srv.log | tail -10; exit 1
