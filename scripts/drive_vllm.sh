#!/bin/bash
bash /root/run_vllm.sh 1 "L2 vLLM 1-card"   v1  ""                     > /tmp/out_v1.txt  2>&1
bash /root/run_vllm.sh 3 "L2 vLLM TP=3"     v3  ""                     > /tmp/out_v3.txt  2>&1
bash /root/run_vllm.sh 3 "L2 vLLM TP=3 noP2P" v3n "NCCL_P2P_DISABLE=1" > /tmp/out_v3n.txt 2>&1
echo ALLDONE > /tmp/vllm_alldone
