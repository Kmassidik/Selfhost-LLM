#!/bin/bash
# R7 — cross-task quant sensitivity. Does quantization break harder tasks (SQL,
# classification) where it left extraction untouched? Merge + quantize each tuned
# model, sweep f16 / Q4_K_M / Q2_K, eval with that task's own metric.
cd /root/Desktop/selfhostllm
export PATH=$HOME/.local/bin:$PATH
BIN=engines/llamacpp/build/bin/llama-server
Q=engines/llamacpp/build/bin/llama-quantize

prep () {  # $1=name  $2=adapter  $3=outdir
  uv run python train/1_extract/merge_export.py --adapter "$2" --out models/hf/$1-merged >/dev/null 2>&1
  mkdir -p "$3"
  uv run python engines/llamacpp/convert_hf_to_gguf.py models/hf/$1-merged --outtype f16 --outfile "$3/$1-f16.gguf" >/dev/null 2>&1
  for t in Q4_K_M Q2_K; do $Q "$3/$1-f16.gguf" "$3/$1-$t.gguf" $t >/dev/null 2>&1; done
}

serve () {  # $1=gguf ; starts server, waits health, echoes PID
  CUDA_VISIBLE_DEVICES=1 nohup $BIN -m "$1" --host 10.0.0.20 --port 8087 -ngl 99 \
    -c 2048 --no-warmup --parallel 8 -a q >/tmp/r7.log 2>&1 &
  echo $!
  for i in $(seq 1 40); do curl -s http://10.0.0.20:8087/health 2>/dev/null | grep -q ok && break; sleep 2; done
}

echo "prep sql + cls (merge, convert, quantize)..."
prep sql train/2_sql/adapter train/2_sql/quant
prep cls train/3_cls/adapter train/3_cls/quant

echo ""; echo "task quant  metric"
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
