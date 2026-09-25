# Setup — every command, as actually run

The exact commands used to bring this box up, in order, with the reasons and the
traps. Nothing here is idealised: where something did not work the first time,
the failure is recorded next to the fix.

Target: **Ubuntu 24.04**, NVIDIA driver already installed, 3× RTX 3060 Ti.

---

## 0 · Where things live

```
/root/Desktop/selfhostllm/
  source/        the code — NN_<name>.py, numbered by knowledge-base chapter
  bench/         the measurement harness. Every engine measured by the same script
  models/
    hf/          safetensors weights (gitignored)
    gguf/        llama.cpp format (gitignored)
    configs/     architecture only, a few KB — committed
  engines/       one serving stack per level, each self-contained
  docs/          this
```

---

## 1 · Python, with uv — never pip

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

cd /root/Desktop/selfhostllm
uv venv --python 3.12
uv pip install -e .                      # reads pyproject.toml
```

torch is deliberately **not** a plain dependency — the CUDA build is about 3 GB and
is tied to the installed driver:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check it:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# 2.11.0+cu128 3
```

> **Why uv rather than pip.** The environment is declared in `pyproject.toml`, not
> remembered. A dependency that exists only in somebody's shell history is a
> dependency that breaks the next clone.

---

## 2 · Surveying the machine

```bash
python3 source/06_box_survey.py
```

Prints every hardware fact the knowledge base claims — cards, interconnect,
processors, memory, disk, network paths, tooling. **If a line disagrees with
chapter 06, the chapter is stale.**

The line that matters most:

```
GPU0   X    SYS  SYS
GPU1  SYS    X   PHB
GPU2  SYS   PHB   X
```

`SYS` means traffic crosses between PCIe root complexes; `PHB` means the cards share
a bridge. **Card 0 is not a peer of cards 1 and 2.** There is no NVLink.

---

## 3 · Getting a model

```bash
mkdir -p models/hf/SmolLM2-360M-Instruct && cd $_
M=HuggingFaceTB/SmolLM2-360M-Instruct
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json; do
  curl -sL --http1.1 --retry 3 -o $f "https://huggingface.co/$M/resolve/main/$f"
done
curl -L --http1.1 --retry 3 -C - -o model.safetensors \
  "https://huggingface.co/$M/resolve/main/model.safetensors"
```

`--http1.1` is not decoration: HTTP/2 gets throttled on large downloads and stalls.
`-C -` resumes rather than restarting.

Architecture only, for models you want to reason about but not run:

```bash
./models/configs/fetch.sh Qwen/Qwen2.5-7B     # a few KB, no weights
python3 source/01_budget_from_config.py models/configs/*/config.json
```

---

## 4 · L0 · Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Trap 1 — the installer may not create the service.** If `systemctl status ollama`
says the unit does not exist, write it:

```ini
# /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=default.target
```

**Trap 2 — bind address.** `OLLAMA_HOST=0.0.0.0:11434`, not localhost, or nothing
outside the box can reach it. This is the most common "works on the server, not from
my laptop" problem and it is one line.

**Trap 3 — pinning to one card is harder than it looks.** For single-GPU work:

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="GGML_VK_VISIBLE_DEVICES=0"
Environment="OLLAMA_VULKAN=0"
```

`CUDA_VISIBLE_DEVICES` alone is **not enough**. Ollama also discovers cards through
Vulkan and will use ones CUDA cannot see. Both interfaces have to be shut off, and
the only way to confirm is to check every card's memory afterwards.

```bash
systemctl daemon-reload && systemctl enable --now ollama
ollama pull smollm2:360m
ollama show smollm2:360m          # confirm: quantization F16, parameters 361.82M
curl -s http://10.0.0.20:11434/api/version
```

---

## 5 · L1 · llama.cpp, built with CUDA

Needs a CUDA compiler, which the driver alone does not provide:

```bash
apt-get install -y cmake ccache nvidia-cuda-toolkit     # ~3 GB, several minutes
nvcc --version                                          # release 12.0
```

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp engines/llamacpp
cd engines/llamacpp
export PATH=/usr/local/cuda/bin:$PATH

cmake -B build -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc) \
      --target llama-server llama-bench llama-cli
```

`CMAKE_CUDA_ARCHITECTURES=86` is this card's compute capability. Without it the build
compiles for a dozen architectures and takes far longer for no benefit.

Confirm the backend and that it sees one card:

```bash
ls build/bin/ | grep cuda            # libggml-cuda.so
CUDA_VISIBLE_DEVICES=0 build/bin/llama-cli --list-devices
# CUDA0: NVIDIA GeForce RTX 3060 Ti (7840 MiB, 7668 MiB free)
```

### Running the same file both engines use

Ollama stores its weights as content-addressed blobs. That blob **is** a GGUF file,
so link it rather than downloading a second copy — same bytes, same inode, no chance
of the two engines silently running different weights:

```bash
ln -f /usr/share/ollama/.ollama/models/blobs/sha256-<hash> \
      models/gguf/SmolLM2-360M-Instruct-F16.gguf
```

Find the hash with `ls -S /usr/share/ollama/.ollama/models/blobs/ | head -1`; the
large one is the weights, and `head -c 4` on it prints `GGUF`.

```bash
CUDA_VISIBLE_DEVICES=0 build/bin/llama-server \
  -m /root/Desktop/selfhostllm/models/gguf/SmolLM2-360M-Instruct-F16.gguf \
  -ngl 99 -c 8192 --host 0.0.0.0 --port 8080
```

| Flag | Meaning |
|---|---|
| `-ngl 99` | how many layers go on the card. 99 means all of them |
| `-c 8192` | context length in tokens |
| `--host 0.0.0.0` | reachable from other machines — same trap as Ollama |

---

## 5b · L2 · vLLM

**Its own environment, not the project one.** vLLM installs `torch 2.13.0+cu130`; the
project venv has `2.11.0+cu128`. Sharing one breaks both.

```bash
cd engines/vllm
uv venv --python 3.12
uv pip install vllm ninja        # ~7.6 GB
```

Three failures worth knowing about, in the order they happen:

**`FileNotFoundError: 'ninja'`** — vLLM compiles kernels at startup and shells out to
`ninja`. Installing it into the venv is not enough; the venv's `bin` must be on `PATH`,
because the compile runs as a subprocess:

```bash
export PATH="$PWD/.venv/bin:$PATH"
```

**`nvcc fatal : Unknown option '--compress-mode=size'`** — vLLM's flashinfer library
compiles with a flag that needs **CUDA 12.8 or newer**. `apt` provides **12.0**. The
failing kernels are for *sampling*, not attention, and vLLM has its own sampler:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

One environment variable instead of a multi-gigabyte toolkit upgrade. Read *which*
component failed before reaching for the large fix.

**Silent detachment failures** — `nohup ... &` over SSH did not survive here. Use
`setsid` and redirect stdin:

```bash
setsid bash -c "./serve.sh > /tmp/vllm.log 2>&1" < /dev/null > /dev/null 2>&1 &
```

Then:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve models/hf/SmolLM2-360M-Instruct \
  --served-model-name SmolLM2-360M-Instruct \
  --dtype float16 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000
```

| Flag | Why |
|---|---|
| `--dtype float16` | **not optional.** The weights are bf16; the GGUF engines read F16. Without this the comparison measures number formats as well as engines |
| `--gpu-memory-utilization 0.85` | vLLM claims this share of the card at startup and rations it. 0.85 of 8 GB leaves 5.15 GiB for the cache |
| `--max-model-len 8192` | context limit; larger needs more reserved cache |

Startup takes a minute or two — it profiles the card and captures execution graphs.
Watch for `Application startup complete.`

---

## 5c · L3 · SGLang

```bash
cd engines/sglang
uv venv --python 3.12
uv pip install "sglang[all]" ninja      # ~8.9 GB
```

**Same CUDA problem as vLLM, but with no optional component to disable.** vLLM's
failure was in the sampler; SGLang's is in *attention*, which it cannot do without.
The fix is a different attention implementation that does not compile through `nvcc`:

```bash
--attention-backend triton --sampling-backend pytorch
```

⚠️ **This means you are not measuring SGLang as designed.** Its default backend is
flashinfer. Anything measured on the Triton fallback is a real measurement of a real
configuration and *not* a fair statement of the engine's performance. Say so.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m sglang.launch_server \
  --model-path models/hf/SmolLM2-360M-Instruct \
  --served-model-name SmolLM2-360M-Instruct \
  --dtype float16 --context-length 8192 \
  --mem-fraction-static 0.85 \
  --attention-backend triton --sampling-backend pytorch \
  --host 0.0.0.0 --port 30000
```

**Stop the previous engine first.** SGLang failed with:

```
ValueError: Loaded weights leave no GPU memory for the KV cache
            under --mem-fraction-static=0.85
```

which reads like the model is too large and was nothing of the sort — vLLM was still
running and holding 6.8 GB. **Check `nvidia-smi` before believing a memory error.**

### Watch the CUDA graph line at startup

```
Capture target decode CUDA graph begin. bs=[1, 2, 4, 8], avail mem=0.64 GB
```

That list is the batch sizes with a recorded command sequence. **Past the largest one,
throughput falls off a cliff** — measured here at 8.6× between batch 8 and batch 10.
Each captured graph costs memory, so lowering `--mem-fraction-static` buys more of
them at the cost of context length.

---

## 6 · Measuring

Every engine is measured by the same script, with the same frozen prompts:

```bash
python3 bench/run.py --endpoint http://10.0.0.20:11434/v1 \
        --model smollm2:360m --level L0 --quant F16 \
        --label "L0 · Ollama · SmolLM2-360M F16"

python3 bench/run.py --endpoint http://10.0.0.20:8080/v1 \
        --model SmolLM2-360M-Instruct-F16 --level L1 --quant F16 \
        --label "L1 · llama.cpp · SmolLM2-360M F16"

python3 bench/table.py            # rebuild the comparison from results/
```

Results land in `bench/results/<run-id>.json`, one per run, and the table is generated
from those files only — so a number that was never measured cannot appear in it.

---

## Rules worth keeping

1. **Never stream a long job over SSH.** Launch detached, return, poll:
   `nohup bash -c '...; echo done > /tmp/x.rc' >/dev/null 2>&1 &`
   A dropped connection has cost this project hours before.
2. **Bind servers to `0.0.0.0`**, never localhost.
3. **Check every card's memory after a run**, not just the one you meant to use.
4. **Use uv, not pip.** Declare dependencies in `pyproject.toml`.
5. **A number not measured on this machine is an estimate**, and must say so.
