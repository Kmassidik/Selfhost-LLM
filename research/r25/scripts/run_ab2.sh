#!/bin/bash
cd /Users/mac/Desktop/R25-orcabonsai/runtime
source ../.venv/bin/activate
args=()
while IFS= read -r line; do [ -n "$line" ] && args+=("$line"); done < ../eval/overrefused_safe.txt
echo "loaded ${#args[@]} prompts"
echo "=== ALPHA 0 (original) ==="
python run.py --pack ../pack --alpha 0 --max-new 90 --temp 0 "${args[@]}"
echo "=== ALPHA 1 (ablated) ==="
python run.py --pack ../pack --alpha 1 --max-new 90 --temp 0 "${args[@]}"
echo AB2_DONE
