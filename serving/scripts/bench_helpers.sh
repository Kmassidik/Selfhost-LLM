#!/bin/bash
R=/root/Desktop/selfhostllm
GGUF=$R/models/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
HF=$R/models/hf/SmolLM2-360M-Instruct
LSRV=$R/engines/llamacpp/build/bin/llama-server
PY=$R/.venv/bin/python
VPY=$R/engines/vllm/.venv/bin/python

# kill engines by PID, never matching our own process tree
kill_engines() {
  local self=$$
  local mine=$(ps -eo pid,ppid --no-headers | awk -v s=$self "{print \$1\" \"\$2}")
  for p in $(ps -eo pid,args --no-headers | grep -E "build/bin/lla|api_server|VLLM::|EngineCore" | grep -v grep | awk "{print \$1}"); do
    [ "$p" = "$self" ] && continue
    kill -9 "$p" 2>/dev/null
  done
  sleep 4
}

wait_ready() {
  local t=0
  while [ $t -lt ${2:-240} ]; do
    code=$(curl -s -o /tmp/ready.out -w "%{http_code}" -m 20 -X POST http://10.0.0.20:$1/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"${3:-m}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"temperature\":0}" 2>/dev/null)
    if [ "$code" = "200" ]; then echo "READY after ${t}s"; return 0; fi
    if [ -n "$LOG" ] && [ -f "$LOG" ] && grep -qE "Engine core initialization failed|exiting due to model loading error|WorkerProc failed to start|error loading model|ValidationError|must be divisible" "$LOG"; then
      echo "FATAL in $LOG after ${t}s"; return 1; fi
    sleep 3; t=$((t+3))
  done
  echo "TIMEOUT after ${t}s"; return 1
}

gpu_snap() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; }
