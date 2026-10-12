#!/usr/bin/env bash
# Put the glicc endpoint (:8081, cards 1+2) back on Ternary Bonsai 2 27B via llama.cpp.
# Exact production command from docs/frontier-models.md. Card 0 (music) is never touched.
set -u
cd /root/Desktop/selfhostllm
TOK=$(cat glicc-api.token)

fuser -k 8081/tcp 2>/dev/null
for i in $(seq 1 20); do fuser 8081/tcp >/dev/null 2>&1 || break; sleep 1; done
sleep 3   # let the old process release VRAM

CUDA_VISIBLE_DEVICES=1,2 setsid nohup engines/prismml-llamacpp/build/bin/llama-server \
  -m models/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 99 -sm layer -fa on \
  -ctk q4_0 -ctv q4_0 -c 131072 --parallel 4 --cache-reuse 256 \
  --api-key "$TOK" -a glicc-model-testing --host 127.0.0.1 --port 8081 \
  > logs/glicc_srv.log 2>&1 < /dev/null &

for i in $(seq 1 60); do
  sleep 3
  curl -s -m3 http://127.0.0.1:8081/health | grep -q ok && { echo "BONSAI_UP on :8081"; exit 0; }
done
echo "BONSAI_FAILED - see logs/glicc_srv.log"; exit 1
