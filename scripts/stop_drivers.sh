#!/bin/bash
for p in $(ps -eo pid,args --no-headers | grep -E "drive_vllm|run_vllm|gpusample" | grep -v grep | awk "{print \$1}"); do kill -9 $p 2>/dev/null; done
source /root/bench_helpers.sh
kill_engines
