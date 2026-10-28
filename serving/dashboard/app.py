#!/usr/bin/env python3
"""Live operations dashboard for the self-hosted Glicc endpoint."""
import html
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

ROOT = "/root/Desktop/selfhostllm"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "http://127.0.0.1:8081"


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
SESSION_SECRET = ENV.get("SESSION_SECRET", "")
if not SESSION_SECRET:
    import secrets
    SESSION_SECRET = secrets.token_hex(32)
    with open(os.path.join(HERE, ".env"), "a") as f:
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

_FAILS = {}
LOCKOUT_AFTER = 5
LOCKOUT_SECONDS = 300
_CACHE_LOCK = threading.Lock()
_LOG_PATH = None
_LOG_INODE = None
_LOG_OFFSET = 0
_LOG_TASKS = set()


def is_locked(ip):
    now = time.time()
    _FAILS[ip] = [t for t in _FAILS.get(ip, []) if now - t < LOCKOUT_SECONDS]
    return len(_FAILS[ip]) >= LOCKOUT_AFTER


def record_fail(ip):
    _FAILS.setdefault(ip, []).append(time.time())


def record_success(ip):
    _FAILS.pop(ip, None)


def token():
    try:
        return open(os.path.join(ROOT, "glicc-api.token")).read().strip()
    except OSError:
        return ""


def llama_get(path, raw=False):
    req = urllib.request.Request(
        BASE + path, headers={"Authorization": "Bearer " + token()}
    )
    with urllib.request.urlopen(req, timeout=3) as r:
        data = r.read().decode()
    return data if raw else json.loads(data)


def metrics_value(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        key, value = line.rsplit(" ", 1)
        try:
            values[key] = float(value)
        except ValueError:
            pass
    return values


def request_count(path):
    global _LOG_PATH, _LOG_INODE, _LOG_OFFSET, _LOG_TASKS
    if not path or not os.path.exists(path):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    with _CACHE_LOCK:
        if path != _LOG_PATH or st.st_ino != _LOG_INODE or st.st_size < _LOG_OFFSET:
            _LOG_PATH, _LOG_INODE, _LOG_OFFSET, _LOG_TASKS = path, st.st_ino, 0, set()
        try:
            with open(path, "rb") as f:
                f.seek(_LOG_OFFSET)
                chunk = f.read()
                _LOG_OFFSET = f.tell()
        except OSError:
            return len(_LOG_TASKS)
        for line in chunk.splitlines():
            match = re.search(rb"\| task (\d+) \| processing task, is_child = 0", line)
            if match:
                _LOG_TASKS.add(int(match.group(1)))
        return len(_LOG_TASKS)


def server_pid():
    try:
        out = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True,
                             text=True, timeout=2).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


def server_log(pid):
    if not pid:
        return None
    try:
        return os.readlink("/proc/%d/fd/1" % pid)
    except OSError:
        return None


def server_uptime(pid):
    if not pid:
        return 0
    try:
        out = subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=2).stdout.strip()
        return int(out) if out else 0
    except Exception:
        return 0


def gpu_rows():
    cmd = ["nvidia-smi",
           "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
           "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        def num(v, cast=float):
            try:
                return cast(v)
            except ValueError:
                return None
        rows.append({
            "index": num(parts[0], int), "name": parts[1],
            "memory_used_mb": num(parts[2]), "memory_total_mb": num(parts[3]),
            "util_pct": num(parts[4]), "temp_c": num(parts[5]), "power_w": num(parts[6]),
        })
    return rows


def host_stats():
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            key, value = line.split(":", 1)
            mem[key] = int(value.split()[0])
    except OSError:
        pass
    return {
        "load_1": round(os.getloadavg()[0], 2),
        "mem_total_gb": round(mem.get("MemTotal", 0) / 1048576, 1),
        "mem_available_gb": round(mem.get("MemAvailable", 0) / 1048576, 1),
        "uptime_s": int(float(open("/proc/uptime").read().split()[0])),
    }


def live_snapshot():
    result = {"ok": False, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "metrics": {}, "slots": [], "model_id": None, "gpus": gpu_rows(),
              "host": host_stats(), "server": {}, "requests": {}, "errors": []}
    try:
        health = llama_get("/health")
        result["ok"] = health.get("status") == "ok"
    except Exception as e:
        result["errors"].append("health: %s" % e)
    try:
        result["model_id"] = llama_get("/v1/models")["data"][0]["id"]
    except Exception as e:
        result["errors"].append("models: %s" % e)
    try:
        metrics = metrics_value(llama_get("/metrics", raw=True))
        result["metrics"] = metrics
    except Exception as e:
        result["errors"].append("metrics: %s" % e)
    try:
        slots = llama_get("/slots")
        keep = ("id", "n_ctx", "is_processing", "n_prompt_tokens",
                "n_prompt_tokens_processed", "n_prompt_tokens_cache", "id_task")
        result["slots"] = [{k: s.get(k) for k in keep if k in s} for s in slots]
    except Exception as e:
        result["errors"].append("slots: %s" % e)

    pid = server_pid()
    log = server_log(pid)
    requests = request_count(log)
    m = result["metrics"]
    result["requests"] = {
        "total_seen": requests,
        "processing": int(m.get("llamacpp:requests_processing", 0)),
        "deferred": int(m.get("llamacpp:requests_deferred", 0)),
    }
    result["tokens"] = {
        "generated": int(m.get("llamacpp:tokens_predicted_total", 0)),
        "prompt": int(m.get("llamacpp:prompt_tokens_total", 0)),
        "cached": int(m.get("llamacpp:prompt_tokens_cached_total", 0)),
    }
    pred_seconds = m.get("llamacpp:tokens_predicted_seconds_total", 0)
    prompt_seconds = m.get("llamacpp:prompt_seconds_total", 0)
    result["speed"] = {
        "prefill_tps": round(m.get("llamacpp:prompt_tokens_seconds", 0), 1),
        "decode_tps": round(m.get("llamacpp:predicted_tokens_seconds", 0), 1),
        "avg_prefill_tps": round(result["tokens"]["prompt"] / prompt_seconds, 1) if prompt_seconds else 0,
        "avg_decode_tps": round(result["tokens"]["generated"] / pred_seconds, 1) if pred_seconds else 0,
    }
    result["spec"] = {
        "draft": int(m.get("llamacpp:spec_decode_num_draft_tokens_total", 0)),
        "accepted": int(m.get("llamacpp:spec_decode_num_accepted_tokens_total", 0)),
        "verification_steps": int(m.get("llamacpp:spec_decode_num_drafts_total", 0)),
    }
    drafted = result["spec"]["draft"]
    accepted = result["spec"]["accepted"]
    result["spec"]["acceptance_pct"] = round(100 * accepted / drafted, 1) if drafted else None
    n_ctx = next((s.get("n_ctx") for s in result["slots"] if s.get("n_ctx")), 0)
    max_tokens = int(m.get("llamacpp:n_tokens_max", 0))
    result["context"] = {
        "n_ctx": n_ctx, "high_water_tokens": max_tokens,
        "fill_pct": round(100 * max_tokens / n_ctx, 1) if n_ctx else None,
        "note": "KV bytes are not exposed; context high-water is the cache-fill proxy",
    }
    cache_hits = result["tokens"]["cached"]
    prompt_total = result["tokens"]["prompt"] + cache_hits
    result["cache_hit_pct"] = round(100 * cache_hits / prompt_total, 1) if prompt_total else None
    result["server"] = {"pid": pid, "log_path": log, "uptime_s": server_uptime(pid)}
    return result


app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET,
                   session_cookie="glicc_dash_session", max_age=7 * 24 * 3600,
                   https_only=False, same_site="lax")
LOGIN_HTML = open(os.path.join(HERE, "login.html")).read()
INDEX_HTML = open(os.path.join(HERE, "index.html")).read()


def authed(request: Request) -> bool:
    return bool(request.session.get("authed"))


def is_guest(request: Request) -> bool:
    return bool(request.session.get("guest"))


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return RedirectResponse("/") if authed(request) else LOGIN_HTML


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
        request.session["guest"] = False
        return {"ok": True}
    record_fail(ip)
    return JSONResponse({"error": "Wrong username or password."}, status_code=401)


@app.post("/api/watch")
def api_watch(request: Request):
    request.session["authed"] = True
    request.session["guest"] = True
    return {"ok": True, "read_only": True}


@app.post("/api/logout")
def api_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return INDEX_HTML if authed(request) else RedirectResponse("/login")


@app.get("/api/live")
def api_live(request: Request):
    if not authed(request):
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    snap = live_snapshot()
    snap["role"] = "guest" if is_guest(request) else "admin"
    return snap


@app.get("/api/status")
def api_status(request: Request):
    if not authed(request):
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    snap = live_snapshot()
    return {
        "healthy": snap["ok"], "model_id": snap["model_id"],
        "models": {k: {"label": v["label"], "sublabel": v["sublabel"],
                       "ready": os.path.exists(os.path.join(ROOT, v["weight_file"]))}
                   for k, v in MODELS.items()},
    }


@app.get("/api/token")
def api_token(request: Request):
    if not authed(request):
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    if is_guest(request):
        return JSONResponse({"error": "read-only session"}, status_code=403)
    return {"token": token()}


@app.post("/api/switch")
async def api_switch(request: Request):
    if not authed(request):
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    if is_guest(request):
        return JSONResponse({"error": "read-only session"}, status_code=403)
    body = await request.json()
    key = body.get("model")
    if key not in MODELS:
        return JSONResponse({"error": "unknown model %r" % key}, status_code=400)
    if not os.path.exists(os.path.join(ROOT, MODELS[key]["weight_file"])):
        return JSONResponse({"error": "%s is not fully downloaded yet" % MODELS[key]["label"]}, status_code=409)
    script = os.path.join(ROOT, MODELS[key]["script"])
    t0 = time.time()
    try:
        out = subprocess.run(["bash", script], cwd=ROOT, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "launcher timed out after 180s"}, status_code=504)
    ok = out.returncode == 0
    return JSONResponse({"ok": ok, "seconds": round(time.time() - t0, 1),
                         "stdout_tail": out.stdout[-800:], "stderr_tail": out.stderr[-800:]},
                        status_code=200 if ok else 500)
