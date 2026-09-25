#!/bin/bash
cd /Users/mac/Desktop/R25-orcabonsai/runtime
source ../.venv/bin/activate
args=()
while IFS= read -r line; do [ -n "$line" ] && args+=("$line"); done < ../eval/benign_overrefusal.txt
echo "loaded ${#args[@]} prompts"
echo "=== ALPHA 0 (ablation OFF / original model) ==="
python run.py --pack ../pack --alpha 0 --max-new 110 --temp 0 "${args[@]}"
echo
echo "=== ALPHA 1 (ablation ON) ==="
python run.py --pack ../pack --alpha 1 --max-new 110 --temp 0 "${args[@]}"
echo AB_DONE
