#!/usr/bin/env python3
"""R11 — TaaS metering proxy. Sits in front of the OpenAI-compatible backend,
forwards each request, reads token usage from the response, and accumulates a
per-user ledger (tokens + cost). GET /usage returns the ledger. This is the
Token-as-a-Service layer: pay-per-inference, metered per API key."""
import http.server, json, threading, requests
from collections import defaultdict

BACKEND = "http://10.0.0.20:8089/v1/chat/completions"
RATE = 0.05 / 1e6           # $/token — R10 fully-loaded ~$0.05 / 1M
ledger = defaultdict(lambda: {"reqs": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
lock = threading.Lock()

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/usage":
            with lock:
                body = json.dumps({k: dict(v, cost_usd=round(v["cost_usd"], 8)) for k, v in ledger.items()}, indent=2).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); raw = self.rfile.read(n)
        user = self.headers.get("Authorization", "Bearer anon").split()[-1]
        try:
            r = requests.post(BACKEND, data=raw, headers={"Content-Type": "application/json"}, timeout=120)
            j = r.json()
        except Exception as e:
            self.send_response(502); self.end_headers(); self.wfile.write(str(e).encode()); return
        u = j.get("usage", {}) or {}
        pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        with lock:
            e = ledger[user]; e["reqs"] += 1; e["prompt_tokens"] += pt
            e["completion_tokens"] += ct; e["cost_usd"] += (pt + ct) * RATE
        out = json.dumps(j).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)

print("metering proxy on 10.0.0.20:8091 -> backend", BACKEND, flush=True)
http.server.ThreadingHTTPServer(("10.0.0.20", 8091), H).serve_forever()
