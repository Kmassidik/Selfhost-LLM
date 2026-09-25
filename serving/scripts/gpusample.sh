#!/bin/bash
# $1 = tag ; samples all cards until killed
for i in $(seq 1 2000); do
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  sleep 0.4
done
