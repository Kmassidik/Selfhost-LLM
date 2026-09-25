#!/bin/bash
cd /root/Desktop/selfhostllm
for pid in $(ss -tlnp 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
  c=$(ps -p $pid -o comm= 2>/dev/null); [ "$c" = "llama-server" -o "$c" = "python" ] && kill -9 $pid 2>/dev/null
done
sleep 2; rm -f /tmp/agent_l3.log
setsid bash -c "CUDA_VISIBLE_DEVICES=0 engines/llamacpp/build/bin/llama-server \
  -m models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --host 10.0.0.20 --port 8086 \
  -ngl 99 -c 8192 --no-warmup > /tmp/agent_l3.log 2>&1" < /dev/null > /dev/null 2>&1 &
for i in $(seq 1 120); do
  sleep 1
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 4 -H 'Content-Type: application/json' \
    -d '{"model":"l","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
    http://10.0.0.20:8086/v1/chat/completions 2>/dev/null)
  [ "$code" = "200" ] && { echo "ready after ${i}s"; exit 0; }
done
echo "not ready"; tail -4 /tmp/agent_l3.log
