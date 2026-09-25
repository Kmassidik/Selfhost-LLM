# From Scratch, Part A — a GPT by hand

A 10.8M-param GPT (transformer + attention written out, not imported), char-level, trained on
TinyStories. Writes coherent English in ~4 min on one RTX 3060 Ti. KB page:
`knowledge-base/from-scratch/a-language-model.html`.

## Results (measured, 2500 steps)
- loss 4.65 -> val 0.92; gibberish -> grammar -> little stories.
- 1 GPU: 81,886 tok/s.  3 GPU (DDP): 216,138 tok/s = 2.64x (88% efficiency; the ~12% is the
  gradient all-reduce over PCIe — no NVLink). Data parallelism = replicate the model + split
  the data (opposite of inference layer-split in R25b).

## Run
```
# corpus (~20 MB, not vendored):
curl -sL "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" | head -c 20000000 > corpus.txt
# 1 GPU:
STEPS=2500 CUDA_VISIBLE_DEVICES=0 python gpt.py
# 3 GPU (data parallel):
STEPS=2500 CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --standalone gpt.py
```

## Files
`gpt.py` (model + training + DDP + sampling), `run.sh` (1-GPU then 3-GPU runner).

## Part B — diffusion (image) from scratch
`diffusion.py` — a 0.27M CNN denoiser on MNIST; learn to predict noise, sample digits from
pure noise (DDPM). MSE 1.05 -> 0.056; recognisable digits in ~105 s on one 3060 Ti. Renders
samples as ASCII. KB: `knowledge-base/from-scratch/b-image-diffusion.html`.
```
# MNIST (~10 MB, not vendored):
mkdir -p mnist && curl -sL "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz" | gunzip > mnist/train-images-idx3-ubyte
STEPS=4000 CUDA_VISIBLE_DEVICES=0 python diffusion.py
```

## Part C — audio from scratch
`audio.py` — mu-law encode a waveform to 256 tokens, train the SAME GPT as Part A to predict
the next sample. Loss 5.73 -> 0.064; generated audio peaks at 330 Hz (E4, a real melody note)
in ~138 s. Honest limit: locks onto one note, does not sequence the full melody (small model +
autoregressive attractor). KB: `knowledge-base/from-scratch/c-audio.html`.
```
STEPS=1500 CUDA_VISIBLE_DEVICES=0 python audio.py   # self-contained (synthesizes its corpus)
```
