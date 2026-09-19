cd /root/r25b
BIN=./fork/build/bin; M=PTQ1_0.gguf; PORT=8091
fuser -k $PORT/tcp 2>/dev/null; sleep 2
CUDA_VISIBLE_DEVICES=0 $BIN/llama-server -m $M -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -c 65536 --port $PORT --host 127.0.0.1 > srv.log 2>&1 &
for i in $(seq 1 60); do sleep 2; curl -s http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break; done
V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); echo "loaded -c 65536, resident ${V} MiB / 8192"
echo "=== speed by depth (8GB card, PTQ1_0, q4 KV, flash attn) ==="
for D in 2000 8000 35000 60000; do
  python3 -c "import json;f='The quick brown fox jumps over the lazy dog. '*($D//9);json.dump({'messages':[{'role':'user','content':f+' Reply in one short sentence.'}],'max_tokens':48,'temperature':0},open('/tmp/req.json','w'))"
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/req.json \
   | python3 -c "import sys,json;t=json.load(sys.stdin).get('timings',{});print(f'  depth {t.get(\"prompt_n\",0):>6} tok  decode {t.get(\"predicted_per_second\",0):5.1f} tok/s  prefill {t.get(\"prompt_per_second\",0):4.0f} tok/s  TTFT {t.get(\"prompt_ms\",0)/1000:.1f}s')"
done
fuser -k $PORT/tcp 2>/dev/null; echo DEPTH_DONE
