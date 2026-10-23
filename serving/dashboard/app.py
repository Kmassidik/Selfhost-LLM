#!/usr/bin/env python3
"""Model-switch dashboard for the glicc endpoint - FastAPI + signed session cookies.

Replaces the earlier stdlib/Basic-Auth version: a real login page instead of the
browser's native auth popup, a session cookie instead of re-sending credentials on
every request, and JSON error responses the frontend turns into toasts instead of
alert(). Credentials and the session-signing secret come from a local .env file,
never from this source file and never from git.

Run directly for testing:  .venv/bin/python app.py
Normally run via the selfhostllm-dashboard.service systemd unit, which points at
this venv's uvicorn.
"""
import json, os, secrets, subprocess, time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

ROOT = "/root/Desktop/selfhostllm"
HERE = os.path.dirname(os.path.abspath(__file__))


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


ENV_PATH = os.path.join(HERE, ".env")
ENV = load_env(ENV_PATH)
DASH_USER = ENV.get("DASH_USER", "admin")
DASH_PASS = ENV.get("DASH_PASS", "")
SESSION_SECRET = ENV.get("SESSION_SECRET", "")
if not SESSION_SECRET:
    # First boot with the old .env: mint one and persist it, rather than using a
    # secret that changes (and invalidates every session) on every restart.
    SESSION_SECRET = secrets.token_hex(32)
    with open(ENV_PATH, "a") as f:
        f.write("\nSESSION_SECRET=%s\n" % SESSION_SECRET)

MODELS = {
    "qwen36": {
        "label": "Qwen3.6-35B-A3B",
        "sublabel": "MoE 35B/3B active · vision · live default",
        "script": "serving/scripts/glicc_qwen36_35b.sh",
        "weight_file": "models/gguf/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
        "min_bytes": 13_000_000_000,
    },
    "davidau": {
        "label": "DavidAU Fable-Fusion-711",
        "sublabel": "dense 27B · abliterated · comparison candidate",
        "script": "serving/scripts/glicc_davidau_fable711.sh",
        "weight_file": "models/gguf/davidau-fable711-27b/Fable-Fusion-711-Q4_K_M.gguf",
        "min_bytes": 16_000_000_000,
    },
}

# ── simple in-memory lockout: 5 failed logins -> 5 minute lock, per source IP ──
# Resets on process restart; this app is small enough that that's an acceptable
# tradeoff for not adding a database just for login attempts.
_FAILS = {}  # ip -> [timestamps]
LOCKOUT_AFTER = 5
LOCKOUT_SECONDS = 300


def is_locked(ip):
    now = time.time()
    _FAILS[ip] = [t for t in _FAILS.get(ip, []) if now - t < LOCKOUT_SECONDS]
    return len(_FAILS[ip]) >= LOCKOUT_AFTER


def record_fail(ip):
    _FAILS.setdefault(ip, []).append(time.time())


def record_success(ip):
    _FAILS.pop(ip, None)


def model_ready(key):
    m = MODELS[key]
    p = os.path.join(ROOT, m["weight_file"])
    return os.path.exists(p) and os.path.getsize(p) >= m["min_bytes"]


def health_check():
    try:
        out = subprocess.run(["curl", "-s", "-m", "3", "http://127.0.0.1:8081/health"],
                             capture_output=True, text=True, timeout=5)
        return "ok" in out.stdout
    except Exception:
        return False


def current_model_id():
    try:
        tok = open(os.path.join(ROOT, "glicc-api.token")).read().strip()
        out = subprocess.run(
            ["curl", "-s", "-m", "3", "-H", "Authorization: Bearer " + tok,
             "http://127.0.0.1:8081/v1/models"],
            capture_output=True, text=True, timeout=5)
        return json.loads(out.stdout)["data"][0]["id"]
    except Exception:
        return None


app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="glicc_dash_session",
    max_age=7 * 24 * 3600,   # a week; "Sign out" clears it sooner if wanted
    https_only=False,        # works over plain-HTTP Tailscale too, not just the
                              # public HTTPS domain - Tailscale's own encryption
                              # covers that path regardless of the cookie flag
    same_site="lax",
)

LOGIN_HTML = open(os.path.join(HERE, "login.html")).read()
INDEX_HTML = open(os.path.join(HERE, "index.html")).read()


def authed(request: Request) -> bool:
    return bool(request.session.get("authed"))


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if authed(request):
        return RedirectResponse("/")
    return LOGIN_HTML


@app.post("/api/login")
async def api_login(request: Request):
    ip = request.client.host if request.client else "unknown"
    if is_locked(ip):
        return JSONResponse({"error": "Too many attempts. Try again in a few minutes."}, status_code=429)
    body = await request.json()
    if not DASH_PASS:
        return JSONResponse({"error": "Server has no password configured."}, status_code=500)
    if body.get("username") == DASH_USER and body.get("password") == DASH_PASS:
        record_success(ip)
        request.session["authed"] = True
        return {"ok": True}
    record_fail(ip)
    return JSONResponse({"error": "Wrong username or password."}, status_code=401)


@app.post("/api/logout")
def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not authed(request):
        return RedirectResponse("/login")
    return INDEX_HTML


@app.get("/api/status")
def api_status(request: Request):
    if not authed(request):
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    return {
        "healthy": health_check(),
        "model_id": current_model_id(),
        "models": {k: {"label": v["label"], "sublabel": v["sublabel"], "ready": model_ready(k)}
                   for k, v in MODELS.items()},
    }


@app.get("/api/token")
def api_token(request: Request):
    if not authed(request):
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    try:
        tok = open(os.path.join(ROOT, "glicc-api.token")).read().strip()
    except Exception:
        tok = None
    return {"token": tok}


@app.post("/api/switch")
async def api_switch(request: Request):
    if not authed(request):
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    body = await request.json()
    key = body.get("model")
    if key not in MODELS:
        return JSONResponse({"error": "unknown model %r" % key}, status_code=400)
    if not model_ready(key):
        return JSONResponse({"error": "%s is not fully downloaded yet" % MODELS[key]["label"]}, status_code=409)
    script = os.path.join(ROOT, MODELS[key]["script"])
    t0 = time.time()
    try:
        out = subprocess.run(["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "launcher timed out after 180s"}, status_code=504)
    ok = out.returncode == 0 and health_check()
    return JSONResponse({
        "ok": ok,
        "seconds": round(time.time() - t0, 1),
        "stdout_tail": out.stdout[-800:],
        "stderr_tail": out.stderr[-800:],
    }, status_code=200 if ok else 500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
