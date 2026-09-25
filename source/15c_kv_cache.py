#!/usr/bin/env python3
"""
15c_kv_cache.py — L4c of the serving exam (ch.15).

Our own KV cache, and the first time this engine GENERATES rather than scores.

Chapter 05 made the argument with index cards: without a cache every new token
replays the whole conversation; with one, the work already done is kept. L4b's
forward pass had no cache, so it was the replay-the-tape version. This is the
index cards.

It is also the test of the prediction recorded in chapter 15 before it was
written: that generation would diverge from the reference on at least one of the
four frozen prompts, with longctx (12 of 37 positions at risk) most likely and
short (0 of 5) least. That prediction is scored at the bottom, held or not.

    uv run source/15c_kv_cache.py
"""
import json, math, os, sys, time, importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fwd", os.path.join(HERE, "15b_forward_pass.py"))
fwd = importlib.util.module_from_spec(spec); spec.loader.exec_module(fwd)
MODEL = fwd.MODEL


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------
class KVCache:
    """One stack of index cards per layer.

    Each layer keeps every key and value it has already computed. A new token
    appends one card to each stack rather than rebuilding the pile.
    """

    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def append(self, i, k, v):
        """Add this step's keys and values, return everything the layer can see."""
        if self.k[i] is None:
            self.k[i], self.v[i] = k, v
        else:
            self.k[i] = torch.cat([self.k[i], k], dim=2)   # grow along sequence
            self.v[i] = torch.cat([self.v[i], v], dim=2)
        return self.k[i], self.v[i]

    def __len__(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


def attention_cached(x, w, cfg, cos, sin, cache, layer, past):
    """Attention that reads the cache instead of recomputing the past.

    `past` is passed in rather than read from the cache: every layer must see the
    same value, and by the time layer 1 runs, layer 0 has already appended.
    """
    b, s, _ = x.shape
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg["hidden_size"] // nh

    q = (x @ w["q"].T).view(b, s, nh,  hd).transpose(1, 2)
    k = (x @ w["k"].T).view(b, s, nkv, hd).transpose(1, 2)
    v = (x @ w["v"].T).view(b, s, nkv, hd).transpose(1, 2)

    # RoPE sees the TRUE position, which is the cache length plus the offset in
    # this step. Getting this wrong is silent: the model keeps producing fluent
    # text, it just thinks every token is at position 0.
    q, k = fwd.apply_rope(q, k, cos, sin)

    k, v = cache.append(layer, k, v)                 # the whole past, including now
    k = fwd.repeat_kv(k, nh // nkv)
    v = fwd.repeat_kv(v, nh // nkv)

    # Torch's fused attention, not our own softmax written out. Identical
    # mathematics; measured in 15c_kernel_test.py to be bit-identical to the
    # reference where the hand-written version carried up to 0.88 of logit error.
    # is_causal handles the prefill mask; during decode (s == 1) the single query
    # may read every cached key, all of which are in its past by construction.
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=(s > 1))
    out = out.transpose(1, 2).reshape(b, s, nh * hd)
    return out @ w["o"].T


def forward_cached(tokens, W, cfg, device, cache):
    eps = cfg["rms_norm_eps"]
    nh = cfg["num_attention_heads"]
    hd = cfg["hidden_size"] // nh
    past = len(cache)
    s = tokens.shape[1]

    x = W["embed"][tokens]
    # rotation tables for positions [past, past + s), not [0, s)
    full_cos, full_sin = fwd.rope_tables(hd, past + s, cfg["rope_theta"], device, x.dtype)
    cos, sin = full_cos[past:past + s], full_sin[past:past + s]

    for i in range(cfg["num_hidden_layers"]):
        L = W["layers"][i]
        x = x + attention_cached(fwd.rms_norm(x, L["ln1"], eps), L, cfg, cos, sin, cache, i, past)
        x = x + fwd.mlp(fwd.rms_norm(x, L["ln2"], eps), L)

    x = fwd.rms_norm(x, W["norm"], eps)
    return x @ W["embed"].T


def generate(ids, W, cfg, device, max_new, cache=True, eos=None):
    """Greedy generation. temperature 0, so no randomness anywhere.

    Stops at end-of-sequence. Without that check the engine keeps predicting
    past the end of its own answer, which reads as a divergence from any
    reference that does stop.
    """
    out = ids
    if cache:
        kv = KVCache(cfg["num_hidden_layers"])
        logits = forward_cached(ids, W, cfg, device, kv)
        for _ in range(max_new):
            nxt = logits[0, -1].argmax().view(1, 1)
            out = torch.cat([out, nxt], dim=1)
            if eos is not None and nxt.item() == eos:
                break
            logits = forward_cached(nxt, W, cfg, device, kv)
        return out, kv
    for _ in range(max_new):                       # the replay-the-tape version
        logits = fwd.forward(out, W, cfg, device)
        nxt = logits[0, -1].argmax().view(1, 1)
        out = torch.cat([out, nxt], dim=1)
        if eos is not None and nxt.item() == eos:
            break
    return out, None


# --------------------------------------------------------------------------
def main():
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    W, cfg = fwd.load(dev)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()

    def expand(text):
        """REPEAT:n:phrase -> phrase repeated n times. Copied from bench/run.py
        so this test sends exactly what the four measured engines were sent."""
        if not text.startswith("REPEAT:"):
            return text
        _, n, rest = text.split(":", 2)
        filler, tail = rest.split("\n\n", 1)
        return filler * int(n) + "\n\n" + tail

    def as_chat(text):
        """The harness posts to /chat/completions, which applies this template.
        Comparing against raw text would be comparing against a different input."""
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)

    prompts = json.load(open("/root/Desktop/selfhostllm/bench/prompts.json"))
    cases = [(p["id"], as_chat(expand(p["text"])), p["max_tokens"])
             for p in prompts["prompts"]]

    # ---- 1 · does the cache change the answer? It must not. ---------------
    print("=" * 72)
    print("1 · CACHE CORRECTNESS — cached and uncached must agree with each other")
    print("=" * 72)
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        a, kv = generate(ids, W, cfg, dev, 24, cache=True)
        b, _ = generate(ids, W, cfg, dev, 24, cache=False)
    same = torch.equal(a, b)
    print(f"  cached   {tok.decode(a[0], skip_special_tokens=True)!r}")
    print(f"  uncached {tok.decode(b[0], skip_special_tokens=True)!r}")
    print(f"  identical: {same}   {'PASS' if same else 'FAIL — the cache changed the answer'}")
    if not same:
        sys.exit(1)

    # ---- 2 · what the cache is worth, across context lengths -----------
    print()
    print("=" * 72)
    print("2 · WHAT THE CACHE IS WORTH — and at what context length")
    print("=" * 72)
    print(f"  {'context':>9}{'new':>5}{'cached':>10}{'uncached':>11}{'speedup':>10}"
          f"{'cache':>10}{'tok/s cached':>14}")
    print("  " + "-" * 67)
    filler = "Self-hosting a language model means understanding every layer. "
    for ctx_words in (1, 20, 100, 400):
        text = filler * ctx_words
        pids = tok(text, return_tensors="pt").input_ids.to(dev)
        n_new = 16
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            _, kv = generate(pids, W, cfg, dev, n_new, cache=True)
        torch.cuda.synchronize(); t_c = time.perf_counter() - t0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            generate(pids, W, cfg, dev, n_new, cache=False)
        torch.cuda.synchronize(); t_u = time.perf_counter() - t0
        print(f"  {pids.shape[1]:>9}{n_new:>5}{t_c:>9.2f}s{t_u:>10.2f}s"
              f"{t_u/t_c:>9.1f}x{kv.bytes()/1e6:>8.1f} MB{n_new/t_c:>13.1f}")
    print()
    print("  Chapter 05 found the same thing and kept it: at a short context there is")
    print("  nothing to skip, so the cache cannot pay. It pays where there is a past.")

    # ---- 3 · THE EXAM: generated output vs the reference -------------------
    # The reference is compared in BOTH of its modes. At long context it does not
    # always agree with itself, and a single-mode comparison would report our
    # engine as wrong for reproducing one of the reference's own two answers.
    print()
    print("=" * 72)
    print("3 · THE EXAM — generated tokens against the reference, temperature 0")
    print("=" * 72)
    print(f"  {'prompt':<10}{'in':>6}{'new':>5}{'vs ref cached':>16}{'vs ref uncached':>18}{'verdict':>13}")
    print("  " + "-" * 68)
    results = {}
    for name, text, max_new in cases:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        with torch.no_grad():
            ours, _ = generate(ids, W, cfg, dev, max_new, cache=True, eos=tok.eos_token_id)
            refs = {}
            for mode in (True, False):
                refs[mode] = ref_model.generate(
                    ids, max_new_tokens=max_new, do_sample=False, use_cache=mode,
                    pad_token_id=tok.eos_token_id)[0, ids.shape[1]:]
        o = ours[0, ids.shape[1]:]
        hits = {m: torch.equal(o, r) for m, r in refs.items()}
        self_consistent = torch.equal(refs[True], refs[False])
        ok = hits[True] or hits[False]
        results[name] = (ok, hits, self_consistent)
        mark = lambda b: "IDENTICAL" if b else "differs"
        verdict = ("IDENTICAL" if hits[True] and hits[False]
                   else "MATCHES ONE" if ok else "DIVERGES")
        print(f"  {name:<10}{ids.shape[1]:>6}{len(o):>5}{mark(hits[True]):>16}"
              f"{mark(hits[False]):>18}{verdict:>13}")
        if not self_consistent:
            print(f"      NOTE: the reference does not agree with ITSELF here —")
            print(f"      ours         {tok.decode(o, skip_special_tokens=True)!r}")
            print(f"      ref cached   {tok.decode(refs[True], skip_special_tokens=True)!r}")
            print(f"      ref uncached {tok.decode(refs[False], skip_special_tokens=True)!r}")
        elif not ok:
            n = min(len(o), len(refs[True]))
            i = 0
            while i < n and o[i].item() == refs[True][i].item():
                i += 1
            print(f"      first difference at new token {i}:"
                  f"  ours {o[i].item()} {tok.decode([o[i].item()])!r}"
                  f"  vs ref {refs[True][i].item()} {tok.decode([refs[True][i].item()])!r}")

    passed = sum(1 for v in results.values() if v[0])
    print()
    print(f"  {passed} of {len(results)} prompts reproduce the reference exactly.")

    # ---- 4 · score the prediction -----------------------------------------
    print()
    print("=" * 72)
    print("4 · THE PREDICTION, SCORED")
    print("=" * 72)
    print('  Recorded in ch.15 before this was written:')
    print('    "Generation will diverge from the reference on at least one of the four')
    print('     frozen prompts. The short prompt, with zero at-risk positions, is the')
    print('     most likely to survive; longctx, at 12 of 37, is the least."')
    print()
    diverged = [k for k, (ok, _, _) in results.items() if not ok]
    print(f"  diverged on: {diverged if diverged else 'nothing — all four identical'}")
    claim1 = len(diverged) >= 1
    print(f"  claim 1 · at least one diverges          {'HELD' if claim1 else 'WRONG'}")
    if claim1:
        short_ok = results['short'][0]
        long_ok = results['longctx'][0]
        print(f"  claim 2 · short survives                 "
              f"{'HELD' if short_ok else 'WRONG'}")
        print(f"  claim 3 · longctx does not               "
              f"{'HELD' if not long_ok else 'WRONG'}")
    else:
        print("  claims 2 and 3 · not testable — nothing diverged")
        print()
        print("  The at-risk analysis said 18 of 83 positions were coin-flips. Every")
        print("  one of those coins landed the same way as the reference. That is a")
        print("  real result and it is not the same as the risk being imaginary.")


if __name__ == "__main__":
    main()
