#!/bin/bash
# Resumable model fetch. curl -C - picks up where it left off, so a dropped
# connection costs the last chunk and not the whole file.
repo="$1"; file="$2"
cd /root/Desktop/selfhostllm/models/gguf || exit 1
setsid nohup curl -fL --retry 8 --retry-delay 5 -C - -o "$file" \
  "https://huggingface.co/$repo/resolve/main/$file" \
  </dev/null > "/tmp/fetch.log" 2>&1 &
disown
echo "fetching $file"
