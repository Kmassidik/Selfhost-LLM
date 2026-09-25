#!/usr/bin/env python3
"""
arena.py — the thing the knowledge base was research for.

One service on the box. It owns the llama-server process, so it can load a
model, unload it, and tell you exactly what the cards are holding while it does.
Chat with whatever is loaded; run the coding battle against it; compare models
on a board that persists.

WHY UNLOADING IS A PROCESS KILL. llama.cpp has no way to release weights in
place. The only reliable way to get the memory back is for the process to exit,
so switching models means stopping one server and starting another. That costs
a reload — a few seconds for a small model, longer for a large one — and it is
the honest trade for never having two models resident at once on 8 GB cards.

    uv run serve/arena.py --host 10.0.0.20 --port 8090
"""
import argparse, html, json, os, re, shutil, signal, struct, subprocess, sys, threading, time
import urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/root/Desktop/selfhostllm"
GGUF = f"{ROOT}/models/gguf"
HF = f"{ROOT}/models/hf"
BIN = f"{ROOT}/engines/llamacpp/build/bin/llama-server"
BACKEND_PORT = 8086
BOARD = f"{ROOT}/bench/results/arena-board.json"
CONV = f"{ROOT}/serve/conversations"        # one JSON per conversation
HERE = os.path.dirname(os.path.abspath(__file__))

state = {"model": None, "proc": None, "ready": False, "loading": False,
         "cards": 1, "engine": "llama.cpp", "note": ""}
lock = threading.Lock()


# ── the cards ─────────────────────────────────────────────────────────
_vram_cache = {"t": 0.0, "rows": []}
_vram_lock = threading.Lock()


def vram(max_age=0.8):
    """nvidia-smi is a subprocess; spawning one per request meant that when
    several clients (and the readiness probe) hit /api/status at once, twenty
    nvidia-smi processes piled up under load, each request timed out, and the
    pile starved the server. Sample at most every max_age seconds and share the
    result. The number is a poll display, not an audit — 0.8s stale is fine."""
    now = time.time()
    if now - _vram_cache["t"] < max_age:
        return _vram_cache["rows"]
    with _vram_lock:
        if time.time() - _vram_cache["t"] < max_age:   # another thread just did it
            return _vram_cache["rows"]
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            rows = []
            for line in out.strip().splitlines():
                u, t, g = [int(x.strip()) for x in line.split(",")]
                rows.append({"used": u, "total": t, "util": g})
        except Exception:
            rows = _vram_cache["rows"]
        _vram_cache.update(t=time.time(), rows=rows)
        return rows


def _gguf_kv(path):
    """Every key-value pair in a GGUF header, or {} if it cannot be read."""
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return {}
            struct.unpack("<I", fh.read(4))
            _, n_kv = struct.unpack("<QQ", fh.read(16))

            def _s():
                n, = struct.unpack("<Q", fh.read(8))
                return fh.read(n).decode("utf-8", "replace")

            def _v(t):
                fixed = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                         6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
                if t in fixed:
                    return struct.unpack(fixed[t], fh.read(struct.calcsize(fixed[t])))[0]
                if t == 8:
                    return _s()
                if t == 9:
                    et, n = struct.unpack("<IQ", fh.read(12))
                    # A tokenizer's merge table is hundreds of thousands of
                    # strings and nothing here wants it. Llama-3 has ~280k and
                    # an earlier cap of 200k made this function return nothing
                    # for that model, so its active-parameter count came back
                    # empty while smaller vocabularies worked. Walk past a big
                    # array instead of refusing it or building it.
                    if n > 8192:
                        for _ in range(n):
                            _v(et)
                        return n
                    return [_v(et) for _ in range(n)]
                raise ValueError("kv type %d" % t)

            out = {}
            for _ in range(n_kv):
                k = _s()
                t, = struct.unpack("<I", fh.read(4))
                out[k] = _v(t)
            return out
    except Exception:
        return {}


_ACTIVE_CACHE = {}


def active_bytes_per_token(path):
    """Bytes of weights one token actually pulls through memory.

    The header carries everything needed: layer count, width, head counts, and
    for a mixture of experts how many experts exist and how many run. Counting
    the matrices a token touches gives active parameters; file size over total
    parameters gives bytes per parameter; the product is what the roofline
    needs.

    This replaced a guess. The panel used to assume a mixture of experts read
    "8 to 20 percent" of itself and printed a range that wide. Computed
    instead, Qwen3-Coder-30B-A3B comes out at 3.04 B active of 30.22 B, which
    is exactly what the A3B in its own name claims -- so the arithmetic checks
    itself against the model's own label.

    Returns (bytes_per_token, active_params, total_params), or None.
    """
    try:
        st = os.stat(path)
        key = (path, st.st_size, int(st.st_mtime))
    except OSError:
        return None
    if key in _ACTIVE_CACHE:
        return _ACTIVE_CACHE[key]

    m = _gguf_kv(path)
    pre = next((k.split(".")[0] for k in m if k.endswith(".block_count")), None)
    if not pre:
        _ACTIVE_CACHE[key] = None
        return None

    def g(k, d=0):
        return m.get(pre + "." + k, d)

    L, E = g("block_count"), g("embedding_length")
    H, KV = g("attention.head_count"), g("attention.head_count_kv")
    hd = g("attention.key_length") or (E // H if H else 0)
    ff = g("feed_forward_length")
    n_exp, n_used = g("expert_count", 0), g("expert_used_count", 0)
    exp_ff = g("expert_feed_forward_length", 0)
    toks = m.get("tokenizer.ggml.tokens")
    V = g("vocab_size", 0) or (toks if isinstance(toks, int) else len(toks or []))
    if not (L and E and H and hd):
        _ACTIVE_CACHE[key] = None
        return None

    attn = E * H * hd + 2 * E * KV * hd + H * hd * E          # q, k, v, o
    if n_exp and n_used and exp_ff:
        ffn_live = n_used * 3 * E * exp_ff                    # gate, up, down
        ffn_all = n_exp * 3 * E * exp_ff
        router = E * n_exp
    else:
        ffn_live = ffn_all = 3 * E * ff
        router = 0
    head = V * E
    active = L * (attn + ffn_live + router) + head
    total = L * (attn + ffn_all + router) + head
    if total <= 0:
        _ACTIVE_CACHE[key] = None
        return None
    out = (active * (st.st_size / total), active, total)
    _ACTIVE_CACHE[key] = out
    return out


def _gguf_short_by(path):
    """How many bytes a .gguf is short of holding its own tensors, or 0.

    A download still in flight is a valid-looking file of the right name and
    the wrong length, and offering it in the model list is a trap: it loads
    for a while and then fails somewhere inside the engine. The header says
    where the tensor data starts and where each tensor sits inside it, so the
    file cannot be shorter than the start plus the largest offset. That is a
    lower bound, not the exact size -- it needs no table of quantisation block
    sizes and it catches any partial download by a mile.

    Returns 0 for a file that passes and -1 for one this parser cannot read.
    Those are kept apart deliberately and only a positive result hides a
    model: a file the parser does not understand is far more likely to be a
    GGUF variant newer than this code than a broken download, and refusing to
    load a model because of that would be a worse bug than the one being
    fixed.
    """
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return -1
            ver, = struct.unpack("<I", fh.read(4))
            n_tensor, n_kv = struct.unpack("<QQ", fh.read(16))
            if not (1 <= ver <= 3) or n_tensor > 1 << 20 or n_kv > 1 << 20:
                return -1

            def _str():
                n, = struct.unpack("<Q", fh.read(8))
                return fh.read(n)

            def _val(t):
                # 0..7 u8 i8 u16 i16 u32 i32 f32 bool, 8 string, 9 array,
                # 10..12 u64 i64 f64
                fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                         10: 8, 11: 8, 12: 8}
                if t in fixed:
                    fh.seek(fixed[t], 1)
                elif t == 8:
                    _str()
                elif t == 9:
                    et, n = struct.unpack("<IQ", fh.read(12))
                    for _ in range(n):
                        _val(et)
                else:
                    raise ValueError("kv type %d" % t)

            align = 32
            for _ in range(n_kv):
                key = _str()
                t, = struct.unpack("<I", fh.read(4))
                if key == b"general.alignment" and t == 4:
                    align, = struct.unpack("<I", fh.read(4))
                else:
                    _val(t)

            biggest = 0
            for _ in range(n_tensor):
                _str()
                nd, = struct.unpack("<I", fh.read(4))
                fh.seek(8 * nd, 1)
                fh.seek(4, 1)                       # ggml type
                off, = struct.unpack("<Q", fh.read(8))
                biggest = max(biggest, off)

            start = fh.tell()
            if align > 1:
                start += (-start) % align
            need = start + biggest
            have = os.path.getsize(path)
            return 0 if have > need else need - have
    except Exception:
        return -1


# Kept for reproducing the measurements in parts I to V, not for new tests.
BASELINE_MODELS = ("smollm2", "meta-llama-3-8b")


def models():
    out = []
    files = sorted(os.listdir(GGUF))
    for f in files:
        if not f.endswith(".gguf"):
            continue
        # A split GGUF is many files; llama.cpp loads them all from the first
        # shard, so list only "-00001-of-" and hide the rest. A single-file
        # model has no "-of-" and is listed as itself.
        if "-of-" in f and "-00001-of-" not in f:
            continue
        p = os.path.join(GGUF, f)
        if "-00001-of-" in f:
            import re as _re
            stem = _re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", f)
            gb = sum(os.path.getsize(os.path.join(GGUF, x)) / 1e9
                     for x in files if x.startswith(stem) and x.endswith(".gguf"))
            # a shard still downloading makes the whole model not ready
            short = max((_gguf_short_by(os.path.join(GGUF, x))
                         for x in files if x.startswith(stem) and x.endswith(".gguf")),
                        default=0)
        else:
            gb = os.path.getsize(p) / 1e9
            short = _gguf_short_by(p)
        # a model bigger than one card's usable memory gets split across three
        # v2.0 tests the best models of 2026. The v1.0 models stay on disk
        # and stay listed: twenty-eight chapters cite them, and a measurement
        # whose subject has been deleted cannot be reproduced. They are
        # labelled rather than removed, so the default is the current one.
        baseline = any(k in f.lower() for k in BASELINE_MODELS)
        got = active_bytes_per_token(p)
        out.append({"file": f, "gb": round(gb, 1),
                    "cards": 3 if gb > 6.0 else 1,
                    "split": "-of-" in f,
                    "moe": bool(got and got[1] < got[2] * 0.95),
                    "active_b": round(got[1] / 1e9, 2) if got else None,
                    "total_b": round(got[2] / 1e9, 2) if got else None,
                    "baseline": baseline, "hf": False,
                    "partial": short > 0})
    # HF (safetensors) models — the format vLLM and SGLang need. A directory
    # with a config.json and a .safetensors is a loadable model.
    import json as _json
    if os.path.isdir(HF):
        for d in sorted(os.listdir(HF)):
            md = os.path.join(HF, d)
            if not os.path.isdir(md) or not os.path.exists(os.path.join(md, "config.json")):
                continue
            sts = [x for x in os.listdir(md) if x.endswith(".safetensors")]
            if not sts:
                continue
            gb = sum(os.path.getsize(os.path.join(md, x)) for x in sts) / 1e9
            tot = None
            try:
                c = _json.load(open(os.path.join(md, "config.json")))
                c = c.get("text_config", c)
                L, E, V, ff = (c.get("num_hidden_layers"), c.get("hidden_size"),
                               c.get("vocab_size"), c.get("intermediate_size", 0))
                if L and E:
                    tot = (L * (4 * E * E + 3 * E * ff) + 2 * (V or 0) * E) / 1e9
            except Exception:
                pass
            out.append({"file": d, "gb": round(gb, 1), "cards": 1, "moe": False,
                        "active_b": round(tot, 2) if tot else None,
                        "total_b": round(tot, 2) if tot else None,
                        "baseline": False, "split": False, "hf": True,
                        "partial": False})
    return out


def _status():
    """What is actually true right now, not what was true when load() returned.

    An engine can die under the app: a 30B split across three cards took an
    Xid 31 MMU fault mid-battle and the process vanished, while this endpoint
    went on reporting ready with the cards sitting at 33/15/15 MB. A status
    that can be stale in the direction of "everything is fine" is worse than
    no status, so the child is checked on every call and a death is reported
    as one.
    """
    p = state.get("proc")
    if state["ready"] and p is not None and p.poll() is not None:
        state.update(ready=False, note=f"engine exited ({p.returncode}) "
                                       f"- check dmesg for Xid faults")
    return {"model": state["model"], "ready": state["ready"],
            "loading": state["loading"], "cards": state["cards"],
            "engine": state["engine"],
            "note": state["note"], "vram": vram(),
            "battle": {"running": battle["running"], "log": battle["log"]},
            "bench": {"running": bench["running"], "phase": bench["phase"],
                      "result": bench["result"], "error": bench["error"]}}


# ── loading and unloading ─────────────────────────────────────────────
def _wait_free(timeout=60, floor=600):
    """Block until the driver has actually taken the memory back.

    A killed process is not a freed card. This used to sleep a flat two
    seconds, which is fine for a 5 GB model and not remotely enough for 18.6 GB
    spread over three cards: the next load raced the teardown and died with
    "unable to allocate CUDA0 buffer", leaving the app with no engine and no
    explanation. Waiting on the number the driver reports costs nothing when
    there was little to free.
    """
    end = time.time() + timeout
    while time.time() < end:
        rows = vram()
        if not rows:
            break
        if max(r["used"] for r in rows) <= floor:
            return True
        time.sleep(0.5)
    time.sleep(2)
    return False


def _kill():
    p = state.get("proc")
    if p and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait(timeout=15)
        except Exception:
            pass
    state.update(proc=None, model=None, ready=False)
    _wait_free()


def _reap_orphans():
    """Kill any llama-server this program left behind on an earlier run.

    _kill() only knows the child in state["proc"], so restarting arena
    orphaned the engine: several GB stayed occupied on the cards while the
    UI truthfully reported no model loaded, and the next Load would have
    found the backend port already taken. Match on the binary path rather
    than the name, so an unrelated llama-server is left alone.
    """
    me, killed = os.getpid(), []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except Exception:
            continue
        if cmd and cmd[0].decode("utf-8", "replace") == BIN:
            try:
                os.killpg(os.getpgid(int(d)), signal.SIGKILL)
            except Exception:
                try:
                    os.kill(int(d), signal.SIGKILL)
                except Exception:
                    continue
            killed.append(d)
    if killed:
        _wait_free()
    return killed


def _ready(timeout=240):
    """A 200 on /v1/models is not readiness — llama-server answers that while
    still loading, then refuses completions. Only a real completion proves it."""
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 1}).encode()
    for _ in range(timeout):
        time.sleep(1)
        p = state.get("proc")
        if p and p.poll() is not None:
            return False
        try:
            r = urllib.request.Request(f"http://127.0.0.1:{BACKEND_PORT}/v1/chat/completions",
                                       body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=6) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
    return False


# ── engines ───────────────────────────────────────────────────────────
# Three serving engines are installed. They do not all serve the same files,
# and pretending they do is how you get a dropdown that lies. llama.cpp reads
# GGUF natively; this SGLang build has a GGUF loader; this vLLM build does not,
# and wants a safetensors model this box has not downloaded. The registry says
# who can serve what, and the UI greys out the rest.
LLAMA_BIN = f"{ROOT}/engines/llamacpp/build/bin/llama-server"
VLLM_BIN  = f"{ROOT}/engines/vllm/.venv/bin/vllm"
SGLANG_PY = f"{ROOT}/engines/sglang/.venv/bin/python"


def engines():
    """Installed engines, each with what it can and cannot do here."""
    out = []
    if os.path.exists(LLAMA_BIN):
        out.append({"id": "llama.cpp", "label": "llama.cpp",
                    "note": "native GGUF · CPU offload for oversized models"})
    if os.path.isdir(f"{ROOT}/engines/sglang/.venv"):
        out.append({"id": "sglang", "label": "SGLang",
                    "note": "needs an HF model (safetensors), not a bare GGUF"})
    if os.path.exists(VLLM_BIN):
        out.append({"id": "vllm", "label": "vLLM",
                    "note": "needs an HF model (safetensors), not a bare GGUF"})
    return out


def engine_can_serve(engine, info):
    """(can, reason). The reason is shown when it cannot."""
    gguf = not info.get("hf")
    if engine == "llama.cpp":
        return (gguf, "" if gguf else "llama.cpp serves GGUF files")
    if engine == "sglang":
        # SGLang loads GGUF weights but its tokenizer path wants an HF repo,
        # not a standalone .gguf — tested, it aborts on exactly that. So it
        # needs the model in HF (safetensors) form, tokenizer and all.
        if gguf:
            return (False, "SGLang needs an HF model — it can't read the tokenizer out of a bare .gguf")
        return (True, "")
    if engine == "vllm":
        if gguf:
            return (False, "this vLLM build has no GGUF loader — needs a safetensors model")
        return (True, "")
    return (False, "unknown engine")


def _gpu_layers(path, cards):
    """How many layers to put on the cards.

    -ngl 99 (everything) OOMs a model bigger than the cards hold — DeepSeek at
    90 GB wanted ~25 GB on 24 GB and died. So for an oversized model, fit as
    many whole layers as the graphics memory allows and leave the rest on the
    CPU. The budget per card is deliberately conservative (~6.2 GB of the 8):
    a layer costs its weights plus its share of the KV cache and compute
    buffers, and overshooting means a hard out-of-memory instead of a slow but
    working load. Measured against DeepSeek this lands on 9 of 43, which is the
    value that loaded where 10 did not.
    """
    m = _gguf_kv(path)
    pre = next((k.split(".")[0] for k in m if k.endswith(".block_count")), None)
    layers = m.get(pre + ".block_count") if pre else None
    try:
        size_gb = os.path.getsize(path) / 1e9
        # for a split model, path is the tiny first shard; sum the set
        if "-of-" in path:
            import re as _re, glob as _glob
            stem = _re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", path)
            size_gb = sum(os.path.getsize(x) / 1e9 for x in _glob.glob(stem + "-*-of-*.gguf"))
    except OSError:
        return 99
    if not layers or size_gb < 7.0 * cards:
        return 99                                   # fits — put it all on GPU
    per_layer = size_gb / layers
    fit = int(cards * 6.2 / per_layer)
    return max(1, min(layers, fit))


def _engine_args(engine, path, n, ctx=8192):
    """The launch command for each engine, serving OpenAI-compatible on BACKEND_PORT."""
    if engine == "sglang":
        # triton attention + pytorch sampler + no cuda graph: all to avoid the
        # flashinfer JIT compile this box cannot do (no nvcc). Measured working.
        return [SGLANG_PY, "-m", "sglang.launch_server", "--model-path", path,
                "--served-model-name", "m",
                "--host", "0.0.0.0", "--port", str(BACKEND_PORT),
                "--tp-size", str(n), "--context-length", str(ctx),
                "--attention-backend", "triton", "--sampling-backend", "pytorch",
                "--disable-cuda-graph", "--mem-fraction-static", "0.5"]
    if engine == "vllm":
        # eager mode skips cuda-graph capture (which needs flashinfer/nvcc);
        # the TORCH_SDPA env in load() supplies the attention backend.
        a = [VLLM_BIN, "serve", path, "--served-model-name", "m",
             "--host", "0.0.0.0", "--port", str(BACKEND_PORT),
             "--max-model-len", str(ctx), "--enforce-eager",
             "--gpu-memory-utilization", "0.85"]
        if n > 1:
            a += ["--pipeline-parallel-size", str(n)]
        return a
    # llama.cpp — the default
    ngl = _gpu_layers(path, n)
    a = [LLAMA_BIN, "-m", path, "--host", "0.0.0.0", "--port", str(BACKEND_PORT),
         "-ngl", str(ngl), "-c", str(ctx), "--no-warmup",
         "-ctk", "q8_0", "-ctv", "q8_0", "-sps", "0",
         "--slot-save-path", "/tmp/arena-slots"]
    if n > 1:
        a += ["-sm", "layer"]
    return a


def load(name, cards=None, engine="llama.cpp"):
    """Load `name` on `engine` across `cards` graphics cards.

    The card count is a choice, not a consequence — Part II measured pipeline
    split at 0.98x of one card and tensor split at 1.07x. The engine is a
    choice too, within what it can actually serve: see engine_can_serve.
    """
    with lock:
        if state["loading"]:
            return False, "already loading"
        state["loading"] = True
    try:
        _kill()
        info = next((m for m in models() if m["file"] == name), None)
        if not info:
            return False, "no such model"
        if info["partial"]:
            return False, f"{name} is still downloading"
        if engine not in {e["id"] for e in engines()}:
            return False, f"{engine} is not installed"
        can, why = engine_can_serve(engine, info)
        if not can:
            return False, why
        n = int(cards or info["cards"])
        n = max(1, min(3, n))
        # llama.cpp keeps the whole hard-won flag set (no similarity slot
        # reuse, slot-erase for the benchmark); the others get their own launch
        # line. -sm layer for llama, pipeline-parallel for the rest, because a
        # tensor split needs the head count to divide by the card count and 3
        # rarely does.
        root_dir = HF if info.get("hf") else GGUF
        args = _engine_args(engine, os.path.join(root_dir, name), n)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n))
        if engine == "vllm":
            # flashinfer wants to JIT a kernel and this box has no nvcc, so use
            # the pure-torch attention and sampler, and put the venv's ninja on
            # PATH for anything that still shells out to it.
            env["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
            env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            env["PATH"] = f"{ROOT}/engines/vllm/.venv/bin:" + env.get("PATH", "")
        state["proc"] = subprocess.Popen(args, stdout=open("/tmp/arena_backend.log", "w"),
                                         stderr=subprocess.STDOUT, env=env,
                                         preexec_fn=os.setsid)
        fits = info["gb"] < 7.2 * n
        state.update(model=name, cards=n, engine=engine, ready=False,
                     note=engine + " · " + f"{n} card{'s' if n > 1 else ''}"
                          + (", mixture of experts" if info["moe"] else "")
                          + ("" if fits else " — larger than the cards hold, "
                             "layers will spill to host memory and it will be slow"))
        ok = _ready(timeout=600)
        state["ready"] = ok
        if not ok:
            tail = open("/tmp/arena_backend.log").read()[-400:]
            _kill()
            return False, "failed to become ready: " + tail[-200:]
        return True, "ready"
    finally:
        state["loading"] = False


# ── the coding battle ─────────────────────────────────────────────────
# ── the inference benchmark ───────────────────────────────────────────
# Vendor figures, NOT measured here, and used only as the ceiling a measured
# rate is compared against. An RTX 3060 Ti is GDDR6 on a 256-bit bus.
CARD_GBPS = {"3060 Ti": 448.0}

bench = {"running": False, "phase": "", "result": None, "error": None}


class Sampler(threading.Thread):
    """nvidia-smi in a loop. Utilisation is the number that separates 'this
    card is saturated' from 'this card is waiting for another one'."""

    def __init__(self, hz=5):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.peak, self.util = {}, {}

    def run(self):
        while not self.stop.is_set():
            for i, r in enumerate(vram()):
                self.peak[i] = max(self.peak.get(i, 0), r["used"])
                self.util.setdefault(i, []).append(r["util"])
            self.stop.wait(0.2)

    def summary(self):
        return [{"card": i,
                 "peak_mb": self.peak.get(i, 0),
                 "util_mean": round(sum(v) / len(v), 1) if (v := self.util.get(i)) else 0,
                 "util_max": max(self.util.get(i) or [0])}
                for i in sorted(self.peak)]


def _stream_once(prompt, max_tokens, temperature=0.0):
    """One streamed completion, timed. Token counts come from the engine --
    counting characters was wrong by 1.43x on the first engine tested."""
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature,
                       "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{BACKEND_PORT}/v1/chat/completions",
                                 body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n = think = answer = 0
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except Exception:
                continue
            if o.get("usage"):
                usage = o["usage"]
            ch = o.get("choices") or []
            delta = (ch[0].get("delta") or {}) if ch else {}
            # A reasoning model puts its thinking in reasoning_content and its
            # answer in content. Counting only the second made this benchmark
            # blind to a model that was working perfectly: three of four
            # prompts spent their whole budget thinking, produced no content,
            # and were dropped as if nothing had happened. The fourth had its
            # decode timed over the sliver after the thinking stopped, which
            # reported 305 tok/s and 381% of the cards' rated bandwidth.
            got = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if got or reasoning:
                if first is None:
                    first = time.perf_counter()
                n += 1
                answer += bool(got)
                think += bool(reasoning)
    end = time.perf_counter()
    out = (usage or {}).get("completion_tokens") or n
    return {"ttft_s": (first - t0) if first else None,
            "decode_s": (end - first) if first else None,
            "out_tokens": out,
            "think_tokens": think, "answer_tokens": answer,
            "in_tokens": (usage or {}).get("prompt_tokens"),
            "source": "engine" if (usage or {}).get("completion_tokens") else "chunks"}


def _bottleneck(path, cards, tps, util):
    """Name what is holding the decode back, from measured numbers.

    Decoding a token reads the weights it needs and does very little arithmetic
    with them, so the ceiling here is memory bandwidth, not compute. Measured
    rate times bytes-per-token is the bandwidth reached; against the cards'
    rated bandwidth that is a percentage, and the percentage is the diagnosis.

    `tps` MUST be a single-stream rate. With several streams in flight the
    weights are read once per batch instead of once per token and this
    arithmetic stops meaning anything: four streams of Llama-3-8B measured
    158.9 tok/s here, which at 4.9 GB each would be 778 GB/s out of memory
    rated at 448.
    """
    got = active_bytes_per_token(path)
    if not got:
        return None
    per_token, active, total = got
    name = "3060 Ti"
    # A pipeline split does NOT give three cards' bandwidth. The layers run in
    # order, so exactly one card is reading weights at any instant and the
    # other two wait. Summing their bandwidth flattered every multi-card result
    # by the card count: the dense 27B measured 380 GB/s, which read as 28% of
    # a 1344 GB/s ceiling that does not exist, when it is 85% of the 448 GB/s
    # ceiling that does — and 85% is saturated. The mean card utilisations say
    # the same thing from the other side: three cards near 28% each is one card
    # working and two idle, not three cards a quarter busy.
    roof = CARD_GBPS[name]          # one card at a time, by design of -sm layer
    split = "pipeline" if cards > 1 else "single card"
    gbps = per_token / 1e9 * tps
    pct = gbps / roof * 100

    busy = [u["util_mean"] for u in util] or [0]
    spread = max(busy) - min(busy)
    if pct >= 55:
        verdict = ("Memory bandwidth. The card doing the work is reading weights "
                   "about as fast as its memory allows"
                   + (". The other cards hold layers but cannot help: a layer "
                      "split runs them in order." if cards > 1 else "")
                   + " A faster card, or fewer bytes per token, is what would "
                     "move this.")
    elif cards > 1 and spread > 25:
        verdict = (f"Pipeline bubble. Card utilisation differs by {spread:.0f} "
                   "points, so cards idle while another works. A layer split runs "
                   "one card at a time by design.")
    elif max(busy) < 45:
        verdict = ("Not the cards. They are idle most of the time, so the limit is "
                   "ahead of them: per-token overhead, the HTTP hop, or a request "
                   "too small to fill a batch. Send more at once.")
    else:
        verdict = ("Mixed. The cards are working but nowhere near the bandwidth "
                   "ceiling, which usually means kernel launch overhead at batch "
                   "size one.")

    notes = []
    if active < total * 0.95:
        notes.append("A mixture of experts: %.2f B parameters of %.2f B run per "
                     "token, %.0f%% of the file. Counted from the header, not "
                     "assumed." % (active / 1e9, total / 1e9, active / total * 100))
    return {"reads_per_token": "%.2f GB" % (per_token / 1e9),
            "achieved_gbps": "%.0f" % gbps, "roof_gbps": round(roof, 0),
            "split": split,
            "pct_of_roof": round(pct, 1), "card": name, "verdict": verdict,
            "notes": notes, "active_b": round(active / 1e9, 2),
            "total_b": round(total / 1e9, 2), "util_spread": round(spread, 1)}


def run_bench():
    """Wrapper so a crash in here is visible.

    The first version of this died on a KeyError in its own thread. The panel
    showed an empty result and a button that had gone back to saying "Measure",
    which is indistinguishable from a benchmark that ran and found nothing.
    """
    try:
        _run_bench()
    except Exception as e:
        bench.update(running=False, phase="",
                     error=f"{type(e).__name__}: {e}")
        raise


def _run_bench():
    ps = json.load(open(f"{ROOT}/bench/prompts.json"))
    info = next((m for m in models() if m["file"] == state["model"]), None) or {}
    bench.update(running=True, phase="starting", result=None, error=None)
    out = {"model": state["model"], "cards": state["cards"],
           "when": time.strftime("%Y-%m-%d %H:%M"), "prompts": [], "concurrency": []}
    sam = Sampler()
    sam.start()
    try:
        for spec in ps["prompts"]:
            bench["phase"] = "prompt: " + spec["id"]
            # the frozen set calls it "text"; REPEAT:n:phrase is how the long
            # prompt is stored without putting 2,000 words in a JSON file
            text = spec["text"]
            if text.startswith("REPEAT:"):
                _, n, rest = text.split(":", 2)
                text = rest * int(n)
            runs = [_stream_once(text, spec["max_tokens"], ps["temperature"])
                    for _ in range(3)]
            runs = [r for r in runs if r["decode_s"]]
            if not runs:
                # Say so. A prompt that produced nothing used to vanish from
                # the table, which looks identical to a prompt that was never
                # asked for.
                out["prompts"].append({"id": spec["id"], "in_tokens": None,
                                       "out_tokens": 0, "ttft_ms": 0,
                                       "decode_tps": 0, "prefill_tps": None,
                                       "note": "produced no tokens"})
                continue
            runs.sort(key=lambda r: r["decode_s"])
            mid = runs[len(runs) // 2]
            out["prompts"].append({
                "id": spec["id"],
                "in_tokens": mid["in_tokens"],
                "out_tokens": mid["out_tokens"],
                "ttft_ms": round(mid["ttft_s"] * 1000, 1),
                "decode_tps": round(mid["out_tokens"] / mid["decode_s"], 1),
                "think_tokens": mid["think_tokens"],
                "answer_tokens": mid["answer_tokens"],
                # prefill rate: the prompt's tokens divided by the wait for the
                # first one. This is the compute-bound half of the work.
                "prefill_tps": (round(mid["in_tokens"] / mid["ttft_s"], 0)
                                if mid["in_tokens"] and mid["ttft_s"] else None)})

        for n in (1, 4, 16):
            bench["phase"] = f"concurrency {n}"
            res = [None] * n
            def one(i):
                try:
                    res[i] = _stream_once("Write a haiku about a graphics card.", 128)
                except Exception:
                    res[i] = None
            ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            wall = time.perf_counter() - t0
            got = [r for r in res if r]
            tok = sum(r["out_tokens"] for r in got)
            out["concurrency"].append({
                "n": n, "total_tps": round(tok / wall, 1),
                "per_stream_tps": round(tok / wall / max(len(got), 1), 1),
                "tokens": tok, "wall_s": round(wall, 2)})
    finally:
        sam.stop.set()
        sam.join(timeout=3)
        out["gpus"] = sam.summary()
        best = max((p["decode_tps"] for p in out["prompts"]), default=0)
        if best and state["model"]:
            out["bottleneck"] = _bottleneck(os.path.join(GGUF, state["model"]),
                                            state["cards"], best, out["gpus"])
        os.makedirs(f"{ROOT}/bench/results", exist_ok=True)
        json.dump(out, open(f"{ROOT}/bench/results/arena-bench-"
                            f"{int(time.time())}.json", "w"), indent=1)
        bench.update(running=False, phase="", result=out)


battle = {"running": False, "log": [], "rows": []}


def run_battle():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ag", f"{ROOT}/source/30_agent.py")
    ag = importlib.util.module_from_spec(spec); spec.loader.exec_module(ag)
    tasks = sorted(d for d in os.listdir(f"{ROOT}/agent-tasks")
                   if os.path.isdir(f"{ROOT}/agent-tasks/{d}"))
    battle.update(running=True, log=[], rows=[])
    ep = f"http://127.0.0.1:{BACKEND_PORT}/v1"
    try:
        for t in tasks:
            battle["log"].append({"task": t, "state": "running"})
            r = ag.solve(ep, "m", f"{ROOT}/agent-tasks/{t}", max_steps=14, verbose=False)
            r["task"] = t
            battle["rows"].append(r)
            battle["log"][-1] = {"task": t, "state": "solved" if r["solved"] else "failed",
                                 "steps": r["steps"], "why": r["why"],
                                 "wall": round(r["wall"], 1)}
        solved = sum(1 for r in battle["rows"] if r["solved"])
        board = json.load(open(BOARD)) if os.path.exists(BOARD) else {}
        key = f"{state['model'] or '?'} · {state['cards']} card" + ("s" if state["cards"] > 1 else "")
        board[key] = {
            "solved": solved, "of": len(tasks),
            "steps": sum(r["steps"] for r in battle["rows"]),
            "sent": sum(r["sent"] for r in battle["rows"]),
            "wall": round(sum(r["wall"] for r in battle["rows"]), 1),
            "cards": state["cards"], "when": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(board, open(BOARD, "w"), indent=1)
    finally:
        battle["running"] = False


# ── conversations, kept as one file each ──────────────────────────────
def _conv_path(cid):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", cid or ""):
        return None
    return os.path.join(CONV, cid + ".json")


def conv_list():
    os.makedirs(CONV, exist_ok=True)
    out = []
    for f in os.listdir(CONV):
        if not f.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(CONV, f)))
            out.append({"id": d["id"], "title": d.get("title") or "untitled",
                        "model": d.get("model"), "updated": d.get("updated", 0),
                        "n": len(d.get("messages", []))})
        except Exception:
            pass
    return sorted(out, key=lambda c: -c["updated"])


def conv_get(cid):
    p = _conv_path(cid)
    if not p or not os.path.exists(p):
        return None
    return json.load(open(p))


def conv_save(d):
    os.makedirs(CONV, exist_ok=True)
    p = _conv_path(d.get("id", ""))
    if not p:
        return False
    d["updated"] = time.time()
    if not d.get("title"):
        first = next((m["content"] for m in d.get("messages", [])
                      if m["role"] == "user"), "")
        d["title"] = (first[:46] + "…") if len(first) > 46 else (first or "untitled")
    json.dump(d, open(p, "w"), indent=1)
    return True


def conv_delete(cid):
    p = _conv_path(cid)
    if p and os.path.exists(p):
        os.remove(p)
        return True
    return False


# ── http ──────────────────────────────────────────────────────────────
def make_handler():
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *a): pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Static assets live beside this file — vendored libraries, fonts,
            # app scripts and styles — so the page works on a network with no
            # way out, which is most of the point of self-hosting. Serve any
            # file under serve/, by extension, and never above it.
            if self.path != "/" and "?" not in self.path and ".." not in self.path:
                rel = self.path.lstrip("/").split("?")[0]
                fp = os.path.normpath(os.path.join(HERE, rel))
                if fp.startswith(HERE) and os.path.isfile(fp) and rel != "arena.py":
                    ext = os.path.splitext(fp)[1].lower()
                    types = {".js": "text/javascript", ".css": "text/css",
                             ".woff2": "font/woff2", ".woff": "font/woff",
                             ".html": "text/html; charset=utf-8",
                             ".svg": "image/svg+xml", ".json": "application/json",
                             ".png": "image/png", ".ico": "image/x-icon"}
                    ct = types.get(ext, "application/octet-stream")
                    binary = ext in (".woff2", ".woff", ".png", ".ico")
                    data = open(fp, "rb").read() if binary else open(fp, encoding="utf-8").read()
                    return self._send(200, data, ct)
            if self.path in ("/", "/index.html"):
                return self._send(200, open(os.path.join(HERE, "arena.html")).read(),
                                  "text/html; charset=utf-8")
            if self.path == "/api/status":
                return self._send(200, _status())
            if self.path == "/api/engines":
                return self._send(200, engines())
            if self.path == "/api/models":
                return self._send(200, models())
            if self.path == "/api/conversations":
                return self._send(200, conv_list())
            if self.path.startswith("/api/conversation/"):
                d = conv_get(self.path.rsplit("/", 1)[-1])
                return self._send(200 if d else 404, d or {"error": "not found"})
            if self.path == "/api/board":
                b = json.load(open(BOARD)) if os.path.exists(BOARD) else {}
                return self._send(200, b)
            return self._send(404, {"error": "not found"})

        def do_DELETE(self):
            if self.path.startswith("/api/conversation/"):
                ok = conv_delete(self.path.rsplit("/", 1)[-1])
                return self._send(200 if ok else 404, {"ok": ok})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or "{}")
            if self.path == "/api/load":
                ok, msg = load(body.get("model", ""), body.get("cards"), body.get("engine", "llama.cpp"))
                return self._send(200 if ok else 500, {"ok": ok, "message": msg})
            if self.path == "/api/unload":
                _kill()
                return self._send(200, {"ok": True})
            if self.path == "/api/battle":
                if battle["running"]:
                    return self._send(409, {"error": "already running"})
                if not state["ready"]:
                    return self._send(409, {"error": "load a model first"})
                threading.Thread(target=run_battle, daemon=True).start()
                return self._send(200, {"ok": True})
            if self.path == "/api/bench":
                if bench["running"] or battle["running"]:
                    return self._send(409, {"error": "something is already running"})
                if not state["ready"]:
                    return self._send(409, {"error": "load a model first"})
                threading.Thread(target=run_bench, daemon=True).start()
                return self._send(200, {"ok": True})
            if self.path == "/api/conversation":
                ok = conv_save(body)
                return self._send(200 if ok else 400, {"ok": ok})
            if self.path == "/api/chat":
                if not state["ready"]:
                    return self._send(409, {"error": "no model loaded"})
                return self._chat(body)
            return self._send(404, {"error": "not found"})

        def _chat(self, body):
            # The browser keeps bookkeeping on each message -- why a reply
            # stopped, what the token cap was when it did. That belongs in the
            # saved conversation, not in a request to the engine, which is
            # entitled to reject a message object carrying keys it does not
            # know. Send it role and content and nothing else.
            msgs = [{"role": m.get("role"), "content": m.get("content", "")}
                    for m in body.get("messages", []) if m.get("role")]
            payload = json.dumps({"model": "m", "messages": msgs,
                                  "max_tokens": int(body.get("max_tokens", 2048)),
                                  "temperature": float(body.get("temperature", 0.2)),
                                  "stream": True,
                                  "stream_options": {"include_usage": True}}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{BACKEND_PORT}/v1/chat/completions", payload,
                {"Content-Type": "application/json"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    for raw in r:
                        self.wfile.write(raw)
                        self.wfile.flush()
            except Exception as e:
                try:
                    self.wfile.write(f"data: {json.dumps({'error': str(e)})}\n\n".encode())
                except Exception:
                    pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.0.0.20")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit(f"llama-server not found at {BIN}")
    orphans = _reap_orphans()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler())
    if orphans:
        print(f"reaped {len(orphans)} orphaned llama-server: {' '.join(orphans)}", flush=True)
    print(f"arena on http://{a.host}:{a.port}   models: {len(models())}", flush=True)
    try:
        srv.serve_forever()
    finally:
        _kill()


if __name__ == "__main__":
    main()
