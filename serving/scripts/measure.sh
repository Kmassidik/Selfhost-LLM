#!/bin/bash
# Measure decode tok/s and TTFT over streamed generations
URL=http://127.0.0.1:8087/v1/chat/completions
PROMPT="Explain how a GPU renders a triangle, step by step, in about 120 words."
for run in 1 2 3; do
  REQ=$(cat <<JSON
{"model":"m","messages":[{"role":"user","content":"$PROMPT"}],"max_tokens":128,"stream":true,"stream_options":{"include_usage":true}}
JSON
)
  t_start=$(date +%s.%N)
  t_first=""
  comp_tokens=""
  # stream; capture first-token time and usage
  while IFS= read -r line; do
    case "$line" in
      data:*)
        now=$(date +%s.%N)
        payload=${line#data: }
        if [ -z "$t_first" ] && echo "$payload" | grep -q '"content"'; then
          # only count as first token if there is actual content or reasoning delta
          t_first=$now
        fi
        if echo "$payload" | grep -q 'completion_tokens'; then
          comp_tokens=$(echo "$payload" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*')
        fi
        ;;
    esac
  done < <(curl -sN --max-time 600 -X POST "$URL" -H "Content-Type: application/json" -d "$REQ")
  t_end=$(date +%s.%N)
  if [ -z "$t_first" ]; then t_first=$t_end; fi
  ttft=$(awk "BEGIN{printf \"%.3f\", $t_first-$t_start}")
  decode_span=$(awk "BEGIN{printf \"%.3f\", $t_end-$t_first}")
  if [ -n "$comp_tokens" ] && [ "$comp_tokens" -gt 1 ]; then
    # decode tok/s = (tokens-1) / (last-first)
    toks=$(awk "BEGIN{printf \"%.2f\", ($comp_tokens-1)/($t_end-$t_first)}")
  else
    toks="NA"
  fi
  echo "run$run: completion_tokens=$comp_tokens TTFT=${ttft}s decode_span=${decode_span}s decode_toks_per_s=$toks"
done
