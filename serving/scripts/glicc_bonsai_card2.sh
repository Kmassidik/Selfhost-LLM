#!/usr/bin/env bash
# glicc endpoint (:8081) on Ternary Bonsai 2 27B, CARD 2 ONLY.
# Since 2026-10-01 the box is shared three ways: card 0 = music (audiocpp), card 1 = image AI
# (selfhostimage-worker.service), card 2 = this. One card holds the 5.95 GB pack + a 64k window
# (R25: 8 GB card, 64k ceiling, ~31 tok/s). Same port, key and model id as before.
# Caveat: the music worker's MiniMax runner uses all three cards; while this holds card 2,
# such a song can run out of memory there.
set -u
cd /root/Desktop/selfhostllm
TOK=$(cat glicc-api.token)
CTX=${CTX:-65536}

fuser -k 8081/tcp 2>/dev/null
for i in $(seq 1 20); do fuser 8081/tcp >/dev/null 2>&1 || break; sleep 1; done

CUDA_VISIBLE_DEVICES=2 setsid nohup engines/prismml-llamacpp/build/bin/llama-server \
  -m models/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -fa on \
  -ctk q4_0 -ctv q4_0 -c "$CTX" --parallel 1 --cache-reuse 256 \
  --api-key "$TOK" -a glicc-model-testing --host 127.0.0.1 --port 8081 \
  > logs/glicc_srv.log 2>&1 < /dev/null &

for i in $(seq 1 60); do
  sleep 3
  if curl -s -m3 http://127.0.0.1:8081/health | grep -q ok; then
    echo "BONSAI_UP card 2, ctx $CTX"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  grep -qiE "out of memory|failed to allocate" logs/glicc_srv.log && break
done
echo "BONSAI_FAILED"; grep -iE "error|memory|fail" logs/glicc_srv.log | tail -4; exit 1
