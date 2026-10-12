"""R29: fetch OrcaSAQ-2-27B (EXL3 3.21 bpw Qwen3.8-27B) into models/hf/, then verify sizes."""
import os
import socket

# This box has no working IPv6 route: curl falls back to IPv4, but huggingface_hub's httpx
# client waits on the IPv6 address until it times out. Resolve IPv4 only.
_gai = socket.getaddrinfo


def _ipv4_first(host, *args, **kwargs):
    res = _gai(host, *args, **kwargs)
    return [r for r in res if r[0] == socket.AF_INET] or res


socket.getaddrinfo = _ipv4_first

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

REPO, DEST = "orcarouter/OrcaSAQ-2-27B", "models/hf/OrcaSAQ-2-27B"
path = snapshot_download(REPO, local_dir=DEST, ignore_patterns=["assets/*"])
print("downloaded to", path, flush=True)
bad = 0
for s in HfApi().model_info(REPO, files_metadata=True).siblings:
    if s.rfilename.startswith("assets/"):
        continue
    p = os.path.join(DEST, s.rfilename)
    size = os.path.getsize(p) if os.path.exists(p) else 0
    ok = size == s.size
    bad += not ok
    print(("ok  " if ok else "BAD ") + f"{s.rfilename}  {size:,} / {s.size:,}")
print("R29_DOWNLOAD_DONE" if not bad else f"R29_DOWNLOAD_INCOMPLETE {bad} files", flush=True)
