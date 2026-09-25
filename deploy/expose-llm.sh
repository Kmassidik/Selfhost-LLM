#!/usr/bin/env bash
# One command: publish the glicc LLM endpoint at llm.glicc.id (merges, keeps music).
cd "$(dirname "$0")/.."
HOST=llm SERVICE=http://127.0.0.1:8081 ./deploy/cf-add-hostname.sh
