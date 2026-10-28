#!/usr/bin/env bash
# glicc endpoint (:8081) on the Heretic/MPOA decensored Qwen3.6-35B-A3B (Q3_K_M, 3B active MoE).
# Same architecture as glicc_qwen36_35b.sh, so KV cost and speed class are identical; the weights
# are the same model with the refusal direction ablated (20/20 MTP tensors preserved).
# Tuned production: 250K, q4 KV, ub32, MTP3, split 1,1,0.8.
set -u
cd /root/Desktop/selfhostllm
TOK=$(cat glicc-api.token)
CTX=${CTX:-256000}
UB=${UB:-32}
CACHE=${CACHE:-q4_0}
TS=${TS:-1,1,0.8}
EXTRA=${EXTRA:-}
SPEC=${SPEC:-"--spec-type draft-mtp --spec-draft-n-max 3"}
D=models/gguf/qwen36-35b-heretic
MM=${MMPROJ-$D/mmproj-BF16.gguf}
MP=""
[ -n "$MM" ] && MP="--mmproj $MM"

fuser -k 8081/tcp 2>/dev/null; sleep 6
for i in $(seq 1 20); do fuser 8081/tcp >/dev/null 2>&1 || break; sleep 1; done

TSFLAG=""
if [ -n "$TS" ]; then TSFLAG="--tensor-split $TS"; fi

CUDA_VISIBLE_DEVICES=0,1,2 setsid nohup /root/Desktop/competitors/llama-upstream/build/bin/llama-server \
  -m $D/heretic-Q3_K_M.gguf $MP \
  -ngl 99 --split-mode layer -ub $UB -fa on -ctk $CACHE -ctv $CACHE -ctkd $CACHE -ctvd $CACHE -c "$CTX" \
  --parallel 1 --cache-reuse 256 $SPEC \
  $TSFLAG $EXTRA --jinja --temp 1.0 --top-p 0.95 --top-k 20 \
  --api-key "$TOK" -a heretic-qwen36-35b --host 127.0.0.1 --port 8081 \
  > logs/glicc_srv.log 2>&1 < /dev/null &

for i in $(seq 1 100); do
  sleep 3
  if curl -s -m3 http://127.0.0.1:8081/health | grep -q ok; then
    echo "HERETIC_UP cards 0+1+2, ctx $CTX, ub $UB, cache $CACHE, ts $TS"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  pgrep -f "[l]lama-server.*8081" >/dev/null || break
done
echo "HERETIC_FAILED"; grep -iE "error|memory|fail" logs/glicc_srv.log | tail -10; exit 1
