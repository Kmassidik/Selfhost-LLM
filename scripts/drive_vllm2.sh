#!/bin/bash
bash /root/run_vllm.sh    3 "L2 vLLM TP=3 NCCL_P2P_DISABLE=1" v3n "NCCL_P2P_DISABLE=1" > /tmp/out_v3n.txt 2>&1
bash /root/run_vllm.sh    2 "L2 vLLM TP=2"                    v2  ""                   > /tmp/out_v2.txt  2>&1
bash /root/run_vllm_pp.sh 3 "L2 vLLM PP=3 (3-card pipeline)"  p3  ""                   > /tmp/out_p3.txt  2>&1
bash /root/run_vllm_pp.sh 3 "L2 vLLM PP=3 noP2P"              p3n "NCCL_P2P_DISABLE=1" > /tmp/out_p3n.txt 2>&1
echo ALLDONE > /tmp/vllm2_alldone
