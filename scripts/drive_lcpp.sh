#!/bin/bash
bash /root/run_lcpp.sh 1card  "L1 llama.cpp 1-card"        c1     > /tmp/out_c1.txt      2>&1
bash /root/run_lcpp.sh layer  "L1 llama.cpp 3-card layer"  c3layer> /tmp/out_c3layer.txt 2>&1
bash /root/run_lcpp.sh row    "L1 llama.cpp 3-card row"    c3row  > /tmp/out_c3row.txt   2>&1
bash /root/run_lcpp.sh tensor "L1 llama.cpp 3-card tensor" c3tens > /tmp/out_c3tens.txt  2>&1
echo ALLDONE > /tmp/lcpp_alldone
