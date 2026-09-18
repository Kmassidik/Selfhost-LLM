#!/bin/bash
cd /root/Desktop/selfhostllm/models/gguf || exit 1
S2T=49956780160; S3T=40964890464
F2=DeepSeek-V4-Flash-0731-UD-IQ2_M-00002-of-00003.gguf
F3=DeepSeek-V4-Flash-0731-UD-IQ2_M-00003-of-00003.gguf
LOG=/root/dl_watchdog.log
while true; do
  s2=$(stat -c%s "$F2" 2>/dev/null || echo 0)
  s3=$(stat -c%s "$F3" 2>/dev/null || echo 0)
  if [ "$s2" -ge "$S2T" ] && [ "$s3" -ge "$S3T" ]; then
    echo "$(date +%H:%M:%S) COMPLETE s2=$s2 s3=$s3" >> "$LOG"; break
  fi
  if ! pgrep -x curl >/dev/null; then
    echo "$(date +%H:%M:%S) RESUMING s2=$s2 s3=$s3" >> "$LOG"
    cd /root/Desktop/selfhostllm
    setsid bash planning/fetch-split.sh "unsloth/DeepSeek-V4-Flash-0731-GGUF" UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00002-of-00003.gguf UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00003-of-00003.gguf </dev/null >/dev/null 2>&1
    cd /root/Desktop/selfhostllm/models/gguf
  else
    echo "$(date +%H:%M:%S) alive s2=$s2 s3=$s3" >> "$LOG"
  fi
  sleep 30
done
