#!/bin/bash
# Fetch the datasets for the From Scratch track (data is NOT committed to git).
# Part A: TinyStories (text) · Part B: MNIST (images) · Part C: self-synthesized, no download.
set -e
cd "$(dirname "$0")"
echo "== Part A: TinyStories (~20 MB) =="
curl -sL "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" | head -c 20000000 > corpus.txt
echo "  corpus.txt $(du -h corpus.txt | cut -f1)"
echo "== Part B: MNIST (~10 MB) =="
mkdir -p mnist
curl -sL "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz" | gunzip > mnist/train-images-idx3-ubyte
echo "  mnist/train-images-idx3-ubyte $(du -h mnist/train-images-idx3-ubyte | cut -f1)"
echo "== Part C: no download — audio.py synthesizes its melody corpus =="
echo "done. paths match gpt.py / diffusion.py defaults."
