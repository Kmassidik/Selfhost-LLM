# R26 — Reproducing ternary QAT on the box (how a ternary model is built)

**Question:** does quantization-aware training beat naive rounding at ternary, and can a
cheap 8 GB box reproduce it? **Model:** Qwen2.5-0.5B-Instruct. **Verdict:** the box shows
the *walls*; the *recovery* needs the cluster — which is the honest lesson.

## Step 1 — naive rounding collapses the model
Round all 168 Linear layers to ternary `{−1,0,+1}` (BitNet absmean scalar):
- fp16 baseline ppl **1.85** → naive ternary ppl **~17M** = **7,000,000× worse**. Gibberish.
- 33% of weights round to exactly 0.

## Step 2 — QAT on toy data → memorization, not recovery
BitLinear (ternary fake-quant + straight-through estimator), fp32/bf16 master weights,
fine-tune 300 steps on ~1 KB of text:
- train loss crashed to **~0.02**, but held-out ppl **exploded ~1000×**.
- Classic **memorization + catastrophic forgetting** — with tiny data the model memorizes
  the training set and forgets general language. Real QAT (BitNet) uses *billions* of tokens
  precisely so it can't cheat this way. (SGD lr 1e-3 first diverged; bf16 AdamW + grad-clip
  fixed the instability but not the data problem.)

## Step 3 — can scaling granularity rescue post-training ternary? No.
Ternary with different scale granularity, no training (fp16 baseline 2.48):
| scaling | ternary ppl | × baseline |
|---|---|---|
| whole-matrix scalar | 17.5M | 7,000,000× |
| per-row | 17.9M | 7,200,000× |
| **group-128** (Bonsai's choice) | **857K** | 345,000× |
| group-32 | 2.5M | (noisy — model fully broken) |

Group-128 helps **~20×** over a single scalar — a real lever — but post-training ternary is
**still 345,000× broken**. No scaling saves it.

## The lesson
You cannot quantize your way to ternary. You must **train** the model to be ternary (QAT, at
scale, on real data) — that is the expensive part PrismML paid for. The 8 GB box measures the
collapse and the scaling lever, but not the recovery. This is why Bonsai's 98.2% retention is
real engineering (QAT at scale + Hadamard rotations + custom kernels), not a trick.

## R26b — the deployment half: size vs speed (does packing buy both?)
Measured on one 3060 Ti, synthetic 4096×4096 layer.

**Size (packing is trivial):**
| storage | size | bit/w |
|---|---|---|
| bf16 | 33.6 MB | 16 |
| int8 ternary | 16.8 MB | 8 (2×) |
| 2-bit packed | 4.19 MB | 2 (8×) |
| 1.58-bit dense trit | 3.31 MB | 1.58 (~10×) |

**Speed (needs a FUSED kernel):** naive "store int8 → dequant → matmul" vs bf16 cuBLAS:
- decode (M=1, memory-bound): **6.3× SLOWER** (540 µs vs 86 µs)
- prefill (M=512, compute-bound): 1.6× slower (1043 µs vs 648 µs)

Storing weights small does **not** speed the matmul in stock torch — you pay a dequant back to
bf16 and still run the matmul at bf16 cost. **The size win is free; the speed win requires a
fused kernel that multiplies on packed weights directly.** That fused kernel is the specialist
moat — exactly PrismML's fork, measured in R25b at 5.95 GB @ 32 tok/s. (A competitive
hand-written Triton/CUDA ternary GEMV would be its own experiment, R26c.)

## Scripts
`r26_naive.py` (collapse), `r26_qat.py` (QAT + STE), `r26_groups.py` (scaling sweep),
`r26b.py` (size + matmul benchmark) — archived in the box repo under `r26/`.
