#!/bin/bash
# Fetch a split GGUF, one shard after another, resumable. Uses the HF token at
# ~/.cache/huggingface/token when present — public repos work without it, but
# an authenticated request gets a higher rate limit, which matters for a
# multi-shard 90 GB pull that HF may otherwise throttle.
repo="$1"; shift
cd /root/Desktop/selfhostllm/models/gguf || exit 1
AUTH=()
TOK=~/.cache/huggingface/token
[ -f "$TOK" ] && AUTH=(-H "Authorization: Bearer $(cat "$TOK")")
{
  for f in "$@"; do
    out="$(basename "$f")"
    echo "=== $out ==="
    curl -fL --retry 8 --retry-delay 5 -C - "${AUTH[@]}" -o "$out" \
      "https://huggingface.co/$repo/resolve/main/$f" 2>&1 | tail -2
  done
  echo "=== all shards done ==="
} > /tmp/fetch-deepseek.log 2>&1 &
disown
echo started
