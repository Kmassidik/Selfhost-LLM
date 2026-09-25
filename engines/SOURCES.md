# Engine sources

The engine folders are third-party clones, git-ignored and not part of this repo.
Recorded 2026-09-26, before their `.git` folders were removed. Re-fetch with
`git clone <url> <dir> && git -C <dir> checkout <commit>`.

| dir | source | branch | commit |
|---|---|---|---|
| `engines/llamacpp` | https://github.com/ggml-org/llama.cpp | master | 37b3a9e0ccba261d1cc245a971deae0b18c201ab |
| `engines/prismml-llamacpp` | https://github.com/PrismML-Eng/llama.cpp | prism | 842b1880415d6f508f03b789e5ce70194def7bfd |
| `engines/vllm-src` | https://github.com/vllm-project/vllm | main | 5a6ccc589281b4302791cc301e258adcbcc5fead |
| `engines/sglang-src` | https://github.com/sgl-project/sglang | main | 46ae84df159e261fa67d5cce3647f3342fe365a7 |

The PrismML fork must be built from source on this box (no AVX2; prebuilt binaries crash).
