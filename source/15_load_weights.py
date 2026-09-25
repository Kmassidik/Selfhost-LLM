#!/usr/bin/env python3
"""
15_load_weights.py — L4a of the serving exam (ch.15).

Load the model ourselves. No `from_pretrained`, no serving framework: open the
safetensors file, parse its header, and turn raw bytes into tensors on the card.

Chapter 02 established what the file is — a 32,672-byte JSON header followed by
361,821,120 numbers at 2 bytes each. This turns that reading into a loader, then
checks it against the reference loader tensor by tensor. Anything less than an
exact match is a bug, not a rounding difference.

    uv run source/15_load_weights.py
"""
import json, mmap, os, struct, sys, time
import torch

MODEL = "/root/Desktop/selfhostllm/models/hf/SmolLM2-360M-Instruct"
SAFES = os.path.join(MODEL, "model.safetensors")

# safetensors dtype strings -> (torch dtype, bytes per element)
DTYPES = {
    "F64": (torch.float64, 8), "F32": (torch.float32, 4),
    "F16": (torch.float16, 2), "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8),   "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),   "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),    "BOOL": (torch.bool, 1),
}


def read_header(path):
    """The first 8 bytes are a little-endian u64: the length of the JSON header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    # every data_offsets pair is relative to the end of the header
    return header, 8 + n


def load_by_hand(path, device):
    """Parse the header, map the file, build a tensor per entry. No framework."""
    header, data_start = read_header(path)
    meta = header.pop("__metadata__", {})

    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    tensors, total_elems = {}, 0
    for name, spec in header.items():
        dtype, itemsize = DTYPES[spec["dtype"]]
        shape = spec["shape"]
        lo, hi = spec["data_offsets"]
        nbytes = hi - lo

        n = 1
        for d in shape:
            n *= d
        assert n * itemsize == nbytes, f"{name}: shape says {n*itemsize} B, offsets say {nbytes}"

        # a tensor IS a flat run of bytes plus a shape — ch.02's point, made executable
        buf = mm[data_start + lo : data_start + hi]
        t = torch.frombuffer(bytearray(buf), dtype=dtype).reshape(shape)
        tensors[name] = t.to(device, non_blocking=True)
        total_elems += n

    mm.close(); f.close()
    return tensors, meta, total_elems, data_start


def role(name):
    """Group a tensor by the job it does, so the inventory teaches the shape."""
    if "embed_tokens" in name:                     return "embedding"
    if name.startswith("lm_head"):                 return "output projection"
    if ".self_attn.q_proj" in name:                return "attention · query"
    if ".self_attn.k_proj" in name:                return "attention · key"
    if ".self_attn.v_proj" in name:                return "attention · value"
    if ".self_attn.o_proj" in name:                return "attention · output"
    if ".mlp.gate_proj" in name:                   return "feed-forward · gate"
    if ".mlp.up_proj" in name:                     return "feed-forward · up"
    if ".mlp.down_proj" in name:                   return "feed-forward · down"
    if "norm" in name:                             return "normalisation"
    return "other"


def main():
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)          # force init before asking for stats
    torch.cuda.reset_peak_memory_stats(dev)

    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    print(f"model   {os.path.basename(MODEL)}")
    print(f"config  {cfg['num_hidden_layers']} layers · hidden {cfg['hidden_size']}"
          f" · {cfg['num_attention_heads']} heads / {cfg['num_key_value_heads']} kv"
          f" · vocab {cfg['vocab_size']}")
    print(f"card    {torch.cuda.get_device_name(dev)}\n")

    t0 = time.perf_counter()
    ours, meta, n_elems, data_start = load_by_hand(SAFES, dev)
    torch.cuda.synchronize()
    t_ours = time.perf_counter() - t0

    filesize = os.path.getsize(SAFES)
    print(f"header      {data_start:,} bytes (8-byte length + JSON)")
    print(f"tensors     {len(ours):,}")
    print(f"parameters  {n_elems:,}")
    print(f"file        {filesize:,} bytes"
          f"  =  params x 2 + {filesize - n_elems*2:,} header\n")

    # ---- inventory by role: where the parameters actually live -------------
    by_role = {}
    for name, t in ours.items():
        by_role.setdefault(role(name), [0, 0])
        by_role[role(name)][0] += 1
        by_role[role(name)][1] += t.numel()
    print(f"{'role':<24}{'tensors':>9}{'params':>15}{'share':>8}")
    print("-" * 56)
    for r, (c, n) in sorted(by_role.items(), key=lambda kv: -kv[1][1]):
        print(f"{r:<24}{c:>9,}{n:>15,}{n/n_elems*100:>7.1f}%")
    print("-" * 56)
    print(f"{'total':<24}{len(ours):>9,}{n_elems:>15,}{100.0:>7.1f}%\n")

    # ---- the check: exact, or it is a bug ---------------------------------
    from safetensors.torch import load_file
    t0 = time.perf_counter()
    ref = load_file(SAFES, device="cuda:0")
    torch.cuda.synchronize()
    t_ref = time.perf_counter() - t0

    assert set(ref) == set(ours), "tensor names differ from the reference loader"
    mismatched = []
    for name in ref:
        a, b = ref[name], ours[name]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            mismatched.append(name)

    print(f"ours        {t_ours*1000:8.1f} ms")
    print(f"reference   {t_ref*1000:8.1f} ms")
    print(f"peak card   {torch.cuda.max_memory_allocated(dev)/1e6:8.1f} MB"
          f"   (weights alone: {n_elems*2/1e6:.1f} MB)\n")

    if mismatched:
        print(f"FAIL  {len(mismatched)} tensor(s) differ from the reference:")
        for n in mismatched[:10]:
            print(f"        {n}")
        sys.exit(1)
    print(f"PASS  all {len(ref):,} tensors bit-identical to the reference loader")
    print("      L4a done — the weights are ours. Next: the forward pass.")


if __name__ == "__main__":
    main()
