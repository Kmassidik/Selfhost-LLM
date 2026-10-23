#!/usr/bin/env python3
"""Tiny model-switch dashboard for the glicc endpoint.

No third-party dependencies on purpose - stdlib http.server only, so it needs nothing
beyond python3 already on the box. Protected by HTTP Basic Auth; credentials come from
a local .env file (never committed) read at startup, not hardcoded here.

Run directly for testing:  python3 app.py
Normally run via the selfhostllm-dashboard.service systemd unit.
"""
import base64, json, os, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/root/Desktop/selfhostllm"
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8082

# ── credentials: read from .env next to this file, never from the repo ──────────
def load_env(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out

ENV = load_env(os.path.join(HERE, ".env"))
DASH_USER = ENV.get("DASH_USER", "admin")
DASH_PASS = ENV.get("DASH_PASS", "")

# ── the models this dashboard knows how to switch to ────────────────────────────
# "ready" is computed at request time from file size, not cached, so a download that
# finishes while the dashboard is already running is picked up immediately.
MODELS = {
    "qwen36": {
        "label": "Qwen3.6-35B-A3B",
        "sublabel": "MoE 35B/3B active · vision · live default",
        "model_id": "qwen3.6-35b-a3b",
        "script": "serving/scripts/glicc_qwen36_35b.sh",
        "weight_file": "models/gguf/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
        "min_bytes": 13_000_000_000,
    },
    "davidau": {
        "label": "DavidAU Fable-Fusion-711",
        "sublabel": "dense 27B · abliterated · comparison candidate",
        "model_id": "davidau-fable711-27b",
        "script": "serving/scripts/glicc_davidau_fable711.sh",
        "weight_file": "models/gguf/davidau-fable711-27b/Fable-Fusion-711-Q4_K_M.gguf",
        "min_bytes": 16_000_000_000,
    },
}


def model_ready(key):
    m = MODELS[key]
    p = os.path.join(ROOT, m["weight_file"])
    return os.path.exists(p) and os.path.getsize(p) >= m["min_bytes"]


def health_check():
    try:
        out = subprocess.run(
            ["curl", "-s", "-m", "3", "http://127.0.0.1:8081/health"],
            capture_output=True, text=True, timeout=5,
        )
        return "ok" in out.stdout
    except Exception:
        return False


def current_model_id():
    """Best-effort: ask the running server what it thinks its model id is, via /v1/models."""
    try:
        tok = open(os.path.join(ROOT, "glicc-api.token")).read().strip()
        out = subprocess.run(
            ["curl", "-s", "-m", "3", "-H", "Authorization: Bearer " + tok,
             "http://127.0.0.1:8081/v1/models"],
            capture_output=True, text=True, timeout=5,
        )
        data = json.loads(out.stdout)
        return data["data"][0]["id"]
    except Exception:
        return None


INDEX_HTML = open(os.path.join(HERE, "index.html")).read()


class Handler(BaseHTTPRequestHandler):
    server_version = "glicc-dashboard/1.0"

    def log_message(self, fmt, *args):
        pass  # keep systemd journal quiet; curl errors still show in exceptions

    def _authed(self):
        if not DASH_PASS:
            return False  # refuse to serve with no password set, rather than fail open
        hdr = self.headers.get("Authorization", "")
        if not hdr.startswith("Basic "):
            return False
        try:
            raw = base64.b64decode(hdr[6:]).decode()
            user, pw = raw.split(":", 1)
        except Exception:
            return False
        return user == DASH_USER and pw == DASH_PASS

    def _require_auth(self):
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="glicc dashboard"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._authed():
            return self._require_auth()
        if self.path == "/" or self.path == "/index.html":
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            self._json(200, {
                "healthy": health_check(),
                "model_id": current_model_id(),
                "models": {k: {"label": v["label"], "sublabel": v["sublabel"],
                               "ready": model_ready(k)} for k, v in MODELS.items()},
            })
        elif self.path == "/api/token":
            # Served only behind this same Basic Auth, never written into a static file.
            try:
                tok = open(os.path.join(ROOT, "glicc-api.token")).read().strip()
            except Exception:
                tok = None
            self._json(200, {"token": tok})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authed():
            return self._require_auth()
        if self.path != "/api/switch":
            return self._json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad json"})
        key = body.get("model")
        if key not in MODELS:
            return self._json(400, {"error": "unknown model %r" % key})
        if not model_ready(key):
            return self._json(409, {"error": "%s is not fully downloaded yet" % MODELS[key]["label"]})
        script = os.path.join(ROOT, MODELS[key]["script"])
        t0 = time.time()
        try:
            out = subprocess.run(["bash", script], cwd=ROOT, capture_output=True,
                                 text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return self._json(504, {"error": "launcher timed out after 180s"})
        ok = out.returncode == 0 and health_check()
        self._json(200 if ok else 500, {
            "ok": ok,
            "seconds": round(time.time() - t0, 1),
            "stdout_tail": out.stdout[-800:],
            "stderr_tail": out.stderr[-800:],
        })


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("dashboard listening on 0.0.0.0:%d (user=%s)" % (PORT, DASH_USER))
    srv.serve_forever()
