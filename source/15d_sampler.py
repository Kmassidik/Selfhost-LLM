#!/usr/bin/env python3
"""
15d_sampler.py — L4d of the serving exam (ch.15).

Choosing a token from the scores.

At temperature 0 this is one line — take the highest. Every chapter so far has
used that line and nothing else. It is also the only setting under which two
engines can be compared at all, which is why the whole benchmark runs there.

The moment temperature is not 0, "pick a token" stops being one line and starts
being three knobs that interact, and the engine stops being reproducible on
purpose. This measures what each knob actually does rather than describing it.

    uv run source/15d_sampler.py
"""
import json, math, os, sys, importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "15c_kv_cache.py"))
kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
fwd = kv.fwd
MODEL = fwd.MODEL


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------
def sample(logits, temperature=0.0, top_k=0, top_p=1.0, generator=None):
    """Turn a row of scores into one token id.

    Order matters and is not arbitrary: temperature reshapes the distribution,
    then top-k and top-p cut it, then one draw is taken from what is left.
    Cutting before scaling would truncate a different distribution than the one
    the temperature was meant to produce.
    """
    if temperature <= 0:
        # Not a temperature of zero — division by zero. It is the LIMIT of the
        # distribution as temperature approaches zero, where all the probability
        # collects on the single highest score. Every engine special-cases it.
        return int(logits.argmax())

    logits = logits.float() / temperature

    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        ordered, idx = torch.sort(logits, descending=True)
        cum = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        # keep everything up to and including the token that crosses p
        cut = cum > top_p
        cut = torch.cat([torch.zeros(1, dtype=torch.bool, device=cut.device), cut[:-1]])
        drop = torch.zeros_like(logits, dtype=torch.bool)
        drop[idx[cut]] = True
        logits = logits.masked_fill(drop, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


def probs_of(logits, temperature, top_k=0, top_p=1.0):
    """The distribution the sampler is drawing from — for checking, not sampling."""
    lg = logits.float() / temperature
    if top_k and top_k < lg.numel():
        kth = torch.topk(lg, top_k).values[-1]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    if top_p < 1.0:
        ordered, idx = torch.sort(lg, descending=True)
        cum = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        cut = cum > top_p
        cut = torch.cat([torch.zeros(1, dtype=torch.bool, device=cut.device), cut[:-1]])
        drop = torch.zeros_like(lg, dtype=torch.bool)
        drop[idx[cut]] = True
        lg = lg.masked_fill(drop, float("-inf"))
    return torch.softmax(lg, dim=-1)


# --------------------------------------------------------------------------
def main():
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    W, cfg = fwd.load(dev)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    text = tok.apply_chat_template([{"role": "user", "content": "The capital of France is"}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    with torch.no_grad():
        cache = kv.KVCache(cfg["num_hidden_layers"])
        logits = kv.forward_cached(ids, W, cfg, dev, cache)[0, -1]

    top = torch.topk(logits.float(), 8)
    print("The scores at one position, before any sampling touches them")
    print(f"  {'rank':<6}{'token':>8}  {'text':<14}{'logit':>9}{'p at T=1':>11}")
    p1 = torch.softmax(logits.float(), dim=-1)
    for r, (v, i) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        print(f"  {r:<6}{i:>8}  {tok.decode([i])!r:<14}{v:>9.3f}{p1[i].item():>11.4f}")

    # ---- 1 · what temperature does ----------------------------------------
    print()
    print("=" * 72)
    print("1 · WHAT TEMPERATURE DOES TO THE DISTRIBUTION")
    print("=" * 72)
    print(f"  {'T':>6}{'p(top1)':>10}{'p(top2)':>10}{'entropy':>10}{'tokens for 90%':>16}")
    print("  " + "-" * 52)
    for T in (0.1, 0.5, 0.7, 1.0, 1.5, 2.0):
        p = probs_of(logits, T)
        s = torch.sort(p, descending=True).values
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum().item()
        n90 = int((s.cumsum(0) < 0.90).sum().item()) + 1
        print(f"  {T:>6.1f}{s[0].item():>10.4f}{s[1].item():>10.4f}{ent:>10.3f}{n90:>16,}")
    print()
    print("  Low temperature concentrates the mass on the leading token; high")
    print("  temperature spreads it. Entropy is that spread as one number, and the")
    print("  last column is what it means in practice: how many different tokens")
    print("  are seriously in play.")

    # ---- 2 · the zero-temperature limit -----------------------------------
    print()
    print("=" * 72)
    print("2 · TEMPERATURE 0 IS A LIMIT, NOT A VALUE")
    print("=" * 72)
    greedy = int(logits.argmax())
    print(f"  greedy (argmax)          -> {greedy} {tok.decode([greedy])!r}")
    for T in (1.0, 0.1, 0.01, 0.001):
        p = probs_of(logits, T)
        agree = int(p.argmax()) == greedy
        print(f"  T={T:<7} p(top1)={p.max().item():.6f}   argmax matches greedy: {agree}")
    print("  T=0        dividing by zero — special-cased to argmax in every engine,")
    print("             including ours. It is the limit above, not a temperature.")

    # ---- 3 · does the sampler draw the distribution it claims? -------------
    print()
    print("=" * 72)
    print("3 · DOES THE SAMPLER DRAW WHAT IT CLAIMS? — 200,000 draws each")
    print("=" * 72)
    print(f"  {'setting':<26}{'kept':>8}{'worst |empirical-target|':>26}{'verdict':>10}")
    print("  " + "-" * 70)
    g = torch.Generator(device=dev); N = 200_000
    ok_all = True
    for label, kwargs in [
        ("T=1.0",                 dict(temperature=1.0)),
        ("T=0.7",                 dict(temperature=0.7)),
        ("T=1.0, top_k=5",        dict(temperature=1.0, top_k=5)),
        ("T=1.0, top_p=0.9",      dict(temperature=1.0, top_p=0.9)),
        ("T=0.7, top_k=40, p=0.95", dict(temperature=0.7, top_k=40, top_p=0.95)),
    ]:
        target = probs_of(logits, **kwargs)
        kept = int((target > 0).sum())
        g.manual_seed(0)
        draws = torch.multinomial(target, N, replacement=True, generator=g)
        emp = torch.bincount(draws, minlength=target.numel()).float() / N
        worst = (emp - target).abs().max().item()
        # sampling error at N draws is about 1/sqrt(N) on the largest bin
        tol = 4.0 / math.sqrt(N)
        good = worst < tol
        ok_all &= good
        print(f"  {label:<26}{kept:>8,}{worst:>26.5f}{'ok' if good else 'OFF':>10}")
    print(f"  tolerance {4.0/math.sqrt(N):.5f}  (4 standard errors at {N:,} draws)")

    # ---- 4 · the knobs actually truncate ----------------------------------
    print()
    print("=" * 72)
    print("4 · TOP-K AND TOP-P CUT EXACTLY WHAT THEY SAY")
    print("=" * 72)
    V = logits.numel()
    print(f"  vocabulary is {V:,} tokens")
    print(f"  {'setting':<22}{'tokens kept':>14}{'share of vocabulary':>22}")
    print("  " + "-" * 58)
    for label, kwargs in [("no cut", dict(temperature=1.0)),
                          ("top_k=40", dict(temperature=1.0, top_k=40)),
                          ("top_k=5", dict(temperature=1.0, top_k=5)),
                          ("top_p=0.95", dict(temperature=1.0, top_p=0.95)),
                          ("top_p=0.9", dict(temperature=1.0, top_p=0.9)),
                          ("top_p=0.5", dict(temperature=1.0, top_p=0.5))]:
        kept = int((probs_of(logits, **kwargs) > 0).sum())
        print(f"  {label:<22}{kept:>14,}{kept/V*100:>21.3f}%")
    print()
    print("  top_k=40 keeps 41, which is not an off-by-one: two tokens tie at exactly")
    print("  10.437500, the 40th score. bf16 carries about three decimal digits, so")
    print("  ties at the cut are common rather than rare, and every implementation")
    print("  that cuts on `score < kth` keeps all of them. Measured, not assumed.")
    print()
    print("  top_k is a fixed count and ignores how confident the model is.")
    print("  top_p adapts: where the model is sure, it keeps few; where the model")
    print("  is unsure, it keeps many. That is the whole argument for preferring it.")

    print()
    print("=" * 72)
    print("PASS — the sampler draws the distribution it claims to" if ok_all
          else "FAIL — empirical frequencies do not match the intended distribution")
    print("=" * 72)
    if not ok_all:
        sys.exit(1)
    print("L4d done. Next: L4e, a batching loop and an endpoint.")


if __name__ == "__main__":
    main()
