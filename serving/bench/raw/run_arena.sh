#!/bin/bash
cd /root/Desktop/selfhostllm
for pid in $(ss -tlnp 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
  c=$(ps -p $pid -o comm= 2>/dev/null)
  [ "$c" = "llama-server" -o "$c" = "python" ] && kill -9 $pid 2>/dev/null
done
sleep 2; rm -f /tmp/arena.log
setsid bash -c ".venv/bin/python -u serve/arena.py --host 10.0.0.20 --port 8090 \
  > /tmp/arena.log 2>&1" < /dev/null > /dev/null 2>&1 &
for i in $(seq 1 30); do
  sleep 1
  curl -s --max-time 3 -o /dev/null -w '' http://10.0.0.20:8090/api/models 2>/dev/null && { echo "arena up after ${i}s"; exit 0; }
done
echo "did not start"; tail -5 /tmp/arena.log
