#!/usr/bin/env bash
# R28 speculative-decoding trials on Ternary Bonsai 2 27B, cards 1+2 (card 0 = music, untouched).
# Frees cards 1+2 by stopping the glicc endpoint, runs each config on :8082,
# and ALWAYS restores the production endpoint on :8081 on exit (trap).
cd /root/Desktop/selfhostllm
TOK=$(cat glicc-api.token); BIN=engines/prismml-llamacpp/build/bin/llama-server
M=models/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf; D=models/gguf/Qwen3.5-0.8B-Q8_0.gguf
OUT=research/r28/results_spec.json

restore(){
  fuser -k 8082/tcp 2>/dev/null; sleep 2
  if ! curl -s -m3 http://127.0.0.1:8081/health | grep -q ok; then
    CUDA_VISIBLE_DEVICES=1,2 setsid nohup $BIN -m $M -ngl 99 -sm layer -fa on -ctk q4_0 -ctv q4_0 \
      -c 131072 --parallel 4 --cache-reuse 256 --api-key "$TOK" -a glicc-model-testing \
      --host 127.0.0.1 --port 8081 > logs/glicc_srv.log 2>&1 < /dev/null &
    for i in $(seq 1 60); do sleep 3; curl -s -m3 http://127.0.0.1:8081/health | grep -q ok && break; done
  fi
  echo "RESTORED :8081 -> $(curl -s -m3 http://127.0.0.1:8081/health)"
}
trap restore EXIT

run(){
  label=$1; shift
  fuser -k 8082/tcp 2>/dev/null; sleep 2
  CUDA_VISIBLE_DEVICES=1,2 $BIN -m $M -ngl 99 -sm layer -fa on -ctk q4_0 -ctv q4_0 \
    -c 16384 --parallel 1 --api-key "$TOK" -a glicc-model-testing \
    --host 127.0.0.1 --port 8082 "$@" > logs/r28_$label.log 2>&1 &
  pid=$!
  for i in $(seq 1 60); do
    sleep 3
    curl -s -m3 http://127.0.0.1:8082/health | grep -q ok && break
    if ! kill -0 $pid 2>/dev/null; then
      echo "=== $label: SERVER EXITED ==="; grep -iE "error|fail|not supported|unsupported" logs/r28_$label.log | tail -3
      return
    fi
  done
  echo "=== $label ==="
  python3 research/r28/bench.py spec http://127.0.0.1:8082 "$TOK" $OUT $label
  grep -iE "speculative|draft" logs/r28_$label.log | grep -vE "^\s*$" | tail -3
  fuser -k 8082/tcp 2>/dev/null; sleep 3
}

fuser -k 8081/tcp 2>/dev/null; sleep 4
# engine-level control: does batching scale at all, with no HTTP/Python in the path?
echo "=== batched-bench (native) ==="
CUDA_VISIBLE_DEVICES=1,2 engines/prismml-llamacpp/build/bin/llama-batched-bench -m $M -ngl 99 -sm layer \
  -fa on -ctk q4_0 -ctv q4_0 -c 8192 -npp 64 -ntg 128 -npl 1,2,4,8 2>&1 \
  | grep -E "^\|" | tee research/r28/batched_bench.txt
run base
run ngram-simple --spec-type ngram-simple
run ngram-mod    --spec-type ngram-mod
run draft3       -md $D -ngld 99 --spec-type draft-simple
run draft8       -md $D -ngld 99 --spec-type draft-simple --spec-draft-n-max 8
echo R28_SPEC_DONE
