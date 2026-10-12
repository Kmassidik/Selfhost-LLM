#!/usr/bin/env bash
# Put the glicc endpoint (:8081, cards 1+2) on OrcaSAQ-2-27B (EXL3) via TabbyAPI/exllamav3.
# Same port, key and model id as Bonsai. If Orca does not come up, reverts to Bonsai.
#   MAXLEN=131072 CACHE=Q8 DRAFT=disabled bash research/r29/swap_to_orca.sh
set -u
cd /root/Desktop/selfhostllm
MAXLEN=${MAXLEN:-131072}; CACHE=${CACHE:-Q8}; DRAFT=${DRAFT:-disabled}
T=engines/tabbyapi

grep -q R29_DOWNLOAD_DONE logs/r29_download.log 2>/dev/null || { echo "download not finished"; exit 2; }
[ -x $T/.venv/bin/python ] || { echo "tabbyapi venv missing"; exit 2; }

# model id = folder name, so expose the checkpoint as glicc-model-testing
ln -sfn OrcaSAQ-2-27B models/hf/glicc-model-testing

# config with this run's knobs; key file from the same token as Bonsai (git-ignored dir)
sed -e "s/__MAXLEN__/$MAXLEN/g" -e "s/__CACHE__/$CACHE/" -e "s/__DRAFT__/$DRAFT/" \
  research/r29/tabby_config.yml > $T/config.yml
( umask 077; printf "api_key: %s\nadmin_key: %s\n" "$(cat glicc-api.token)" \
  "$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')" > $T/api_tokens.yml )

fuser -k 8081/tcp 2>/dev/null
for i in $(seq 1 20); do fuser 8081/tcp >/dev/null 2>&1 || break; sleep 1; done
sleep 3

( cd $T && CUDA_VISIBLE_DEVICES=1,2 setsid nohup .venv/bin/python main.py \
    > ../../logs/glicc_orca.log 2>&1 < /dev/null & echo $! > ../../logs/glicc_orca.pid )
PID=$(cat logs/glicc_orca.pid)
echo "tabbyapi pid $PID: maxlen=$MAXLEN cache=$CACHE draft=$DRAFT"

AUTH="Authorization: Bearer $(cat glicc-api.token)"
for i in $(seq 1 120); do
  sleep 5
  if curl -s -m3 http://127.0.0.1:8081/v1/model -H "$AUTH" | grep -q glicc-model-testing; then
    echo "ORCA_UP on :8081"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    exit 0
  fi
  kill -0 "$PID" 2>/dev/null || break
done
echo "ORCA_FAILED - last log lines:"; grep -iE "error|memory|fail|Traceback" logs/glicc_orca.log | tail -5
kill "$PID" 2>/dev/null
echo "reverting to Bonsai..."; bash research/r29/swap_to_bonsai.sh
exit 1
