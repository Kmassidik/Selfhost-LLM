#!/bin/bash
# R7 (re-run, eval only) — GGUFs already built. Fix: -c 8192 --parallel 4 =
# 2048 ctx/slot, so the SQL schema prompt is not truncated.
cd /root/Desktop/selfhostllm
export PATH=$HOME/.local/bin:$PATH
BIN=engines/llamacpp/build/bin/llama-server

serve () {
  CUDA_VISIBLE_DEVICES=1 nohup $BIN -m "$1" --host 10.0.0.20 --port 8087 -ngl 99 \
    -c 8192 --no-warmup --parallel 4 -a q >/tmp/r7.log 2>&1 &
  echo $!
  for i in $(seq 1 40); do curl -s http://10.0.0.20:8087/health 2>/dev/null | grep -q ok && break; sleep 2; done
}

echo "task quant  metric"
for t in f16 Q4_K_M Q2_K; do
  SRV=$(serve train/2_sql/quant/sql-$t.gguf)
  m=$(uv run python train/2_sql/eval_sql.py --base-url http://10.0.0.20:8087/v1 --model q \
        --data train/2_sql/data/hard.jsonl 2>/dev/null | grep EXEC_MATCH | grep -oE '[0-9.]+%')
  printf "sql  %-7s %s (exec-match)\n" "$t" "$m"; kill $SRV 2>/dev/null; sleep 3
done
for t in f16 Q4_K_M Q2_K; do
  SRV=$(serve train/3_cls/quant/cls-$t.gguf)
  m=$(uv run python train/3_cls/eval_cls.py --base-url http://10.0.0.20:8087/v1 --model q \
        --data train/3_cls/data/hard.jsonl 2>/dev/null | grep MACRO_F1 | grep -oE '[0-9.]+%')
  printf "cls  %-7s %s (macro-F1)\n" "$t" "$m"; kill $SRV 2>/dev/null; sleep 3
done
echo "=== R7 done ==="
