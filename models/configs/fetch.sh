#!/usr/bin/env bash
# Fetch one model's config.json (and generation_config.json if present).
# Usage: ./fetch.sh Qwen/Qwen3-8B
set -e
[ -z "$1" ] && { echo "usage: $0 <org>/<model>"; exit 1; }
DIR="$(cd "$(dirname "$0")" && pwd)/$(basename "$1")"
mkdir -p "$DIR"
for f in config.json generation_config.json; do
  curl -sfL --http1.1 --retry 3 -o "$DIR/$f" \
    "https://huggingface.co/$1/raw/main/$f" || rm -f "$DIR/$f"
done
echo "$1 -> $DIR"
ls -la "$DIR" | grep -v '^total\|^d'
