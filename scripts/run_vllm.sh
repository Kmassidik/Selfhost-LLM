#!/bin/bash
# $1 = tp size (1|3)  $2 = label  $3 = tag  $4 = extra env
source /root/bench_helpers.sh
TP=$1; LABEL=$2; TAG=$3; EXTRAENV=$4
kill_engines
LOG=/tmp/vllm_$TAG.log
export PATH="$R/engines/vllm/.venv/bin:$PATH"
if [ "$TP" = "1" ]; then DEV="CUDA_VISIBLE_DEVICES=0"; else DEV="CUDA_VISIBLE_DEVICES=0,1,2"; fi
CMD="env $DEV VLLM_USE_FLASHINFER_SAMPLER=0 PATH=$R/engines/vllm/.venv/bin:$PATH $EXTRAENV $R/engines/vllm/.venv/bin/vllm serve $HF --served-model-name SmolLM2-360M-Instruct --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.85 --tensor-parallel-size $TP --host 10.0.0.20 --port 8081"
echo "CMD: $CMD"
setsid bash -c "$CMD > $LOG 2>&1" < /dev/null > /dev/null 2>&1 &
sleep 5
wait_ready 8081 420 SmolLM2-360M-Instruct || { echo "=== SERVER FAILED TO START ==="; echo "--- last 60 log lines ---"; tail -60 $LOG; exit 1; }
echo "--- GPU after load ---"; gpu_snap
bash /root/gpusample.sh > /tmp/gpusample_$TAG.txt 2>&1 &
SAMP=$!
cd $R && $PY bench/run.py --endpoint http://10.0.0.20:8081/v1 --model SmolLM2-360M-Instruct \
   --label "$LABEL" --level L2 --quant F16 --note "vllm 0.29.0, TP=$TP, 3x RTX 3060 Ti box $EXTRAENV"
kill $SAMP 2>/dev/null
echo "--- GPU during run ---"
bash /root/gpusum.sh /tmp/gpusample_$TAG.txt
