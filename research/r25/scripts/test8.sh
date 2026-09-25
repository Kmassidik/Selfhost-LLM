cd /root/r25b
BIN=./fork/build/bin; M=PTQ1_0.gguf; PORT=8091
start(){ fuser -k $PORT/tcp 2>/dev/null; sleep 2
  CUDA_VISIBLE_DEVICES=0 $BIN/llama-server -m $M -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -c $1 --port $PORT --host 127.0.0.1 > srv.log 2>&1 &
  for i in $(seq 1 60); do sleep 2
    curl -s http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && return 0
    grep -qiE "out of memory|failed to allocate|cudaMalloc" srv.log && return 1
  done; return 1; }
depth(){ F=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*($1//9))")
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$F Reply in one short sentence.\"}],\"max_tokens\":48,\"temperature\":0}" \
  | python3 -c "import sys,json;t=json.load(sys.stdin).get(\"timings\",{});print(f\"  depth {t.get(\"prompt_n\",0):>6} tok  ->  decode {t.get(\"predicted_per_second\",0):5.1f} tok/s  prefill {t.get(\"prompt_per_second\",0):4.0f} tok/s\")"; }
echo "=== 8GB context ceiling (q4 KV, flash attn, 1 card) ==="
CEIL=0
for C in 131072 98304 65536 49152; do
  if start $C; then V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); echo "  -c $C : FITS, resident ${V} MiB / 8192"; CEIL=$C; break
  else echo "  -c $C : OOM"; fi
done
echo "=== speed by depth at -c $CEIL ==="
for D in 2000 8000 35000 64000; do [ $D -lt $CEIL ] && depth $D; done
fuser -k $PORT/tcp 2>/dev/null
echo TEST8_DONE
