#!/bin/bash
# $1 = mode (1card|layer|row|tensor)  $2 = label  $3 = tag
source /root/bench_helpers.sh
MODE=$1; LABEL=$2; TAG=$3
kill_engines
LOG=/tmp/lcpp_$TAG.log
case $MODE in
  1card) ENVV="CUDA_VISIBLE_DEVICES=0"; SM="none" ;;
  *)     ENVV="CUDA_DEVICE_ORDER=PCI_BUS_ID"; SM="$MODE" ;;
esac
setsid bash -c "env $ENVV $LSRV -m $GGUF --host 10.0.0.20 --port 8080 -ngl 99 -c 4096 -sm $SM > $LOG 2>&1" < /dev/null > /dev/null 2>&1 &
sleep 3
wait_ready 8080 240 || { echo "=== SERVER FAILED TO START ==="; tail -40 $LOG; exit 1; }
echo "--- devices llama.cpp reports ---"
grep -iE "using device|CUDA[0-9]|assigned to device|buffer size|split mode|offload" $LOG | head -40
echo "--- GPU after load ---"; gpu_snap
# sample GPU utilisation throughout the benchmark
bash /root/gpusample.sh > /tmp/gpusample_$TAG.txt 2>&1 &
SAMP=$!
cd $R && $PY bench/run.py --endpoint http://10.0.0.20:8080/v1 --model llama-3-8b-q4km \
   --label "$LABEL" --level L1 --quant Q4_K_M --note "llama.cpp -sm $SM, $MODE, 3x RTX 3060 Ti box"
kill $SAMP 2>/dev/null
echo "--- GPU during run ---"
bash /root/gpusum.sh /tmp/gpusample_$TAG.txt
