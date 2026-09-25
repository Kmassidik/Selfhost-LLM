#!/bin/bash
# R7 (faithful) — cross-task quant sensitivity on the safetensors path.
# bitsandbytes int8 / nf4 vs bf16, adapter on top (QLoRA-style inference), so the
# tuning is preserved (unlike the GGUF path, which degraded SQL/CLS).
cd /root/Desktop/selfhostllm
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
echo "task     quant  metric"
run () {  # $1=label $2=script $3=adapter $4=data $5=grepkey
  for q in "" "--quant int8" "--quant nf4"; do
    qn=$(echo "$q" | grep -oE 'int8|nf4'); qn=${qn:-bf16}
    m=$(uv run python "$2" --adapter "$3" --data "$4" $q 2>/dev/null | grep "$5" | grep -oE '[0-9.]+%' | head -1)
    printf "%-8s %-6s %s\n" "$1" "$qn" "$m"
  done
}
run extract train/1_extract/eval.py     train/1_extract/adapter train/1_extract/data/hard.jsonl EXACT
run sql      train/2_sql/eval_sql.py     train/2_sql/adapter     train/2_sql/data/hard.jsonl     EXEC_MATCH
run cls      train/3_cls/eval_cls.py     train/3_cls/adapter     train/3_cls/data/hard.jsonl     MACRO_F1
echo "=== R7-bnb done ==="
