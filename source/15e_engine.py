#!/usr/bin/env python3
"""
15e_engine.py — L4e of the serving exam (ch.15). The engine itself.

A batching loop and an OpenAI-compatible endpoint, so this thing can be pointed
at bench/run.py exactly like the four engines in chapters 10 to 13 and earn a
row in the same table rather than a paragraph of excuses.

WHAT THIS DOES AND DOES NOT DO
  Requests that arrive together are served together, sharing one forward pass
  per step. That is STATIC batching: the batch is fixed when it starts and the
  whole batch finishes before the next one begins. It is not the CONTINUOUS
  batching of chapters 12 and 13, where a finished sequence is evicted mid-flight
  and a waiting one takes its slot immediately. The difference is measurable and
  is the point of comparing against them.

Sequences are LEFT-padded so every one of them ends at the same column, which
is what makes a single shared cache tensor possible. The real position of each
token is then its column minus that sequence's padding, and RoPE must be given
that real position or the model silently believes every prompt starts at zero.

    uv run source/15e_engine.py --host 10.0.0.20 --port 8085
"""
import argparse, json, math, os, queue, sys, threading, time, uuid, importlib.util
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_NEW = 256            # cache headroom for generated tokens
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("kv", os.path.join(HERE, "15c_kv_cache.py"))
kv = importlib.util.module_from_spec(_s); _s.loader.exec_module(kv)
_s2 = importlib.util.spec_from_file_location("smp", os.path.join(HERE, "15d_sampler.py"))
smp = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(smp)
fwd = kv.fwd
MODEL = fwd.MODEL


# --------------------------------------------------------------------------
# batched generation
# --------------------------------------------------------------------------
class Batch:
    """One wave of requests, generated together."""

    def __init__(self, reqs, W, cfg, device, pad_id):
        self.reqs, self.W, self.cfg, self.dev = reqs, W, cfg, device
        self.n = len(reqs)
        lens = [len(r.ids) for r in reqs]
        self.maxlen = max(lens)
        self.pad = [self.maxlen - L for L in lens]

        # left-pad so every sequence ends in the same column
        ids = torch.full((self.n, self.maxlen), pad_id, dtype=torch.long, device=device)
        for i, r in enumerate(reqs):
            ids[i, self.pad[i]:] = torch.tensor(r.ids, device=device)
        self.ids = ids
        # real position of each column, per sequence; padding sits at position 0
        self.pos = torch.stack([
            torch.clamp(torch.arange(self.maxlen, device=device) - p, min=0)
            for p in self.pad])
        # Allocated once at full size and written in place. Growing it with
        # torch.cat would move it on every step, and a recorded graph cannot
        # follow a tensor that moves.
        L = cfg["num_hidden_layers"]
        nkv = cfg["num_key_value_heads"]
        hd = cfg["hidden_size"] // cfg["num_attention_heads"]
        self.cap = self.maxlen + MAX_NEW
        self.k = [torch.zeros(self.n, nkv, self.cap, hd, dtype=torch.bfloat16,
                              device=device) for _ in range(L)]
        self.v = [torch.zeros(self.n, nkv, self.cap, hd, dtype=torch.bfloat16,
                              device=device) for _ in range(L)]
        self.filled = 0            # how many cache columns hold real keys
        # the decode mask, at a fixed address so a recording can read it
        self._bias = torch.zeros(self.n, 1, 1, self.cap,
                                 dtype=torch.bfloat16, device=device)
        self.graph = None
        self._nograph = False
        self._gtok = self._gpos = self._gout = None
        self._wpos = torch.zeros(1, dtype=torch.long, device=device)

        self.done = [False] * self.n
        self.out = [[] for _ in range(self.n)]

    def _rope(self, pos):
        hd = self.cfg["hidden_size"] // self.cfg["num_attention_heads"]
        inv = 1.0 / (self.cfg["rope_theta"] **
                     (torch.arange(0, hd, 2, device=self.dev).float() / hd))
        f = pos.float().unsqueeze(-1) * inv            # (n, s, hd/2)
        emb = torch.cat((f, f), dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def forward(self, tokens, pos, first):
        cfg, W = self.cfg, self.W
        eps = cfg["rms_norm_eps"]
        nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hd = cfg["hidden_size"] // nh
        n, s = tokens.shape
        past = 0 if first else self.filled
        write = past

        x = W["embed"][tokens]
        cos, sin = self._rope(pos)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)   # (n, 1, s, hd)

        # a column is readable if it is real (not padding) and not in the future
        # Decode reads the mask buffer that _refresh_mask fills before each
        # step. Rebuilding it here would be captured into the recording, and
        # the replay would then overwrite the fresh mask with a stale one.
        if not first:
            bias = self._bias
        else:
            total = self.cap
            keep = torch.zeros(n, total, dtype=torch.bool, device=self.dev)
            keep[:, :past + s] = True
            for i, p in enumerate(self.pad):
                keep[i, :p] = False
            bias = torch.zeros(n, 1, s, total, device=self.dev, dtype=torch.bfloat16)
            bias.masked_fill_(~keep[:, None, None, :], float("-inf"))
            causal = torch.ones(s, total, dtype=torch.bool, device=self.dev).triu(past + 1)
            bias.masked_fill_(causal[None, None], float("-inf"))

        for i in range(cfg["num_hidden_layers"]):
            L = W["layers"][i]
            h = fwd.rms_norm(x, L["ln1"], eps)
            q = (h @ L["q"].T).view(n, s, nh,  hd).transpose(1, 2)
            k = (h @ L["k"].T).view(n, s, nkv, hd).transpose(1, 2)
            v = (h @ L["v"].T).view(n, s, nkv, hd).transpose(1, 2)
            q = q * cos + fwd.rotate_half(q) * sin
            k = k * cos + fwd.rotate_half(k) * sin
            if s == 1:
                # decode: the column comes from a device tensor, so a recording
                # writes wherever _wpos points at replay time rather than where
                # it pointed at capture time.
                self.k[i].index_copy_(2, self._wpos, k)
                self.v[i].index_copy_(2, self._wpos, v)
            else:
                self.k[i][:, :, write:write + s] = k
                self.v[i][:, :, write:write + s] = v
            kk = fwd.repeat_kv(self.k[i], nh // nkv)
            vv = fwd.repeat_kv(self.v[i], nh // nkv)
            o = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, attn_mask=bias)
            x = x + o.transpose(1, 2).reshape(n, s, nh * hd) @ L["o"].T
            x = x + fwd.mlp(fwd.rms_norm(x, L["ln2"], eps), L)

        self.filled = past + s      # the cache now holds this many columns
        return fwd.rms_norm(x, W["norm"], eps) @ W["embed"].T

    def _refresh_mask(self):
        keep = torch.zeros(self.n, self.cap, dtype=torch.bool, device=self.dev)
        keep[:, :self.filled + 1] = True
        for i, p in enumerate(self.pad):
            keep[i, :p] = False
        b = torch.zeros(self.n, 1, 1, self.cap, dtype=torch.bfloat16, device=self.dev)
        b.masked_fill_(~keep[:, None, None, :], float("-inf"))
        self._bias.copy_(b)

    def _capture(self):
        """Record one decode step and replay it for every token after.

        Only safe because the cache, the mask and the token and position buffers
        all live at fixed addresses now. Capture is attempted once per batch and
        abandoned on any error — a slower engine beats a wrong one.
        """
        if self.graph is not None or self._nograph:
            return
        try:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            saved = self.filled
            self._refresh_mask()
            with torch.cuda.stream(st):
                for _ in range(2):
                    self.forward(self._gtok, self._gpos, False)
                    self.filled = saved
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._gout = self.forward(self._gtok, self._gpos, False)
            self.filled = saved
            torch.cuda.synchronize()
            self.graph = g
        except Exception:
            self._nograph = True

    def step(self, first=False):
        """One shared forward pass; returns the token chosen for each sequence."""
        if first:
            logits = self.forward(self.ids, self.pos, True)[:, -1]
            self.cur = self.pos[:, -1]
            self._gtok = torch.zeros(self.n, 1, dtype=torch.long, device=self.dev)
            self._gpos = torch.zeros(self.n, 1, dtype=torch.long, device=self.dev)
        else:
            self.cur = self.cur + 1
            self._gtok.copy_(self.next_tok)
            self._gpos.copy_(self.cur.unsqueeze(1))
            self._wpos.fill_(self.filled)
            self._capture()
            if self.graph is not None:
                # the mask depends on how much of the cache is filled, and the
                # recording reads it from a fixed address — so refresh it here
                self._refresh_mask()
                self.graph.replay()
                logits = self._gout[:, -1]
                self.filled += 1
            else:
                self._refresh_mask()
                logits = self.forward(self._gtok, self._gpos, False)[:, -1]
        toks = []
        for i, r in enumerate(self.reqs):
            t = smp.sample(logits[i], temperature=r.temperature,
                           top_k=r.top_k, top_p=r.top_p, generator=r.gen)
            toks.append(t)
        self.next_tok = torch.tensor(toks, device=self.dev).view(self.n, 1)
        return toks


# --------------------------------------------------------------------------
class Request:
    def __init__(self, ids, params):
        self.ids = ids
        self.temperature = float(params.get("temperature", 0.0) or 0.0)
        self.top_k = int(params.get("top_k", 0) or 0)
        self.top_p = float(params.get("top_p", 1.0) or 1.0)
        self.max_tokens = int(params.get("max_tokens", 128) or 128)
        self.stream = bool(params.get("stream", False))
        self.gen = None
        self.q = queue.Queue()
        self.prompt_tokens = len(ids)
        self.completion_tokens = 0


class Engine:
    """One worker thread owns the card. Requests queue; waves are batched."""

    def __init__(self, device, max_batch=16, wait_ms=6):
        self.dev = device
        self.W, self.cfg = fwd.load(device)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        self.eos = self.tok.eos_token_id
        self.inbox = queue.Queue()
        self.max_batch, self.wait = max_batch, wait_ms / 1000.0
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, req):
        self.inbox.put(req)
        return req

    def _collect(self):
        first = self.inbox.get()
        reqs = [first]
        deadline = time.perf_counter() + self.wait
        while len(reqs) < self.max_batch:
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            try:
                reqs.append(self.inbox.get(timeout=left))
            except queue.Empty:
                break
        return reqs

    @torch.no_grad()
    def _loop(self):
        while True:
            reqs = self._collect()
            b = Batch(reqs, self.W, self.cfg, self.dev, self.pad_id)
            toks = b.step(first=True)
            budget = max(r.max_tokens for r in reqs)
            for i, t in enumerate(toks):
                self._emit(b, i, t)
            for _ in range(budget - 1):
                if all(b.done):
                    break
                toks = b.step()
                for i, t in enumerate(toks):
                    self._emit(b, i, t)
            for i, r in enumerate(reqs):
                if not b.done[i]:
                    b.done[i] = True
                    r.q.put(None)

    def _emit(self, b, i, t):
        r = b.reqs[i]
        if b.done[i]:
            return
        if t == self.eos or r.completion_tokens >= r.max_tokens:
            b.done[i] = True
            r.q.put(None)
            return
        r.completion_tokens += 1
        b.out[i].append(t)
        r.q.put(t)
        if r.completion_tokens >= r.max_tokens:
            b.done[i] = True
            r.q.put(None)


# --------------------------------------------------------------------------
# the OpenAI-compatible surface bench/run.py already speaks
# --------------------------------------------------------------------------
def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._json(200, {"object": "list", "data": [
                    {"id": "ours-smollm2-360m", "object": "model", "owned_by": "l4e"}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._json(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)))
            msgs = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]
            text = engine.tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
            ids = engine.tok(text, add_special_tokens=False).input_ids
            req = Request(ids, body)
            engine.submit(req)
            rid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            want_usage = bool((body.get("stream_options") or {}).get("include_usage"))

            if not req.stream:
                toks = []
                while True:
                    t = req.q.get()
                    if t is None:
                        break
                    toks.append(t)
                return self._json(200, {
                    "id": rid, "object": "chat.completion", "created": created,
                    "model": "ours-smollm2-360m",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant",
                        "content": engine.tok.decode(toks, skip_special_tokens=True)}}],
                    "usage": {"prompt_tokens": req.prompt_tokens,
                              "completion_tokens": len(toks),
                              "total_tokens": req.prompt_tokens + len(toks)}})

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def sse(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                 "model": "ours-smollm2-360m",
                 "choices": [{"index": 0, "delta": {"role": "assistant"},
                              "finish_reason": None}]})
            n = 0
            while True:
                t = req.q.get()
                if t is None:
                    break
                n += 1
                sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": "ours-smollm2-360m",
                     "choices": [{"index": 0,
                                  "delta": {"content": engine.tok.decode([t])},
                                  "finish_reason": None}]})
            sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                 "model": "ours-smollm2-360m",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if want_usage:
                sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                     "model": "ours-smollm2-360m", "choices": [],
                     "usage": {"prompt_tokens": req.prompt_tokens,
                               "completion_tokens": n,
                               "total_tokens": req.prompt_tokens + n}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.0.0.20")   # the LAN, never localhost
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--max-batch", type=int, default=16)
    a = ap.parse_args()

    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    eng = Engine(dev, max_batch=a.max_batch)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(eng))
    print(f"L4e listening on http://{a.host}:{a.port}/v1  "
          f"(max batch {a.max_batch}, static batching)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
