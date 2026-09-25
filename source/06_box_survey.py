#!/usr/bin/env python3
"""Every hardware fact chapter 06 states, read off the machine.

Chapter 06. The specs there were originally gathered by hand with a dozen
different commands, which made the chapter true but not reproducible. This
prints the same facts in one pass so the chapter can be checked rather than
trusted — and so a change to the machine shows up as a diff.

    python3 source/06_box_survey.py
"""
import os, shutil, subprocess, sys

# uv installs to ~/.local/bin, which a non-login shell does not have on PATH
ENV = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", ""))


def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=20, env=ENV).stdout.strip()
    except Exception as e:
        return f"(failed: {e})"


def section(title):
    print(f"\n=== {title} ===")


print("BOX SURVEY — the facts chapter 06 claims")

section("identity")
print(f"  hostname   {run('hostname')}")
print(f"  os         {run('lsb_release -ds 2>/dev/null || cat /etc/os-release | head -1')}")
print(f"  kernel     {run('uname -r')}")

section("graphics cards")
gpus = run("nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap "
           "--format=csv,noheader")
for line in gpus.splitlines():
    print(f"  {line}")
print(f"  count      {len(gpus.splitlines())}")

section("interconnect — this is the part that matters")
topo = run("nvidia-smi topo -m")
for line in topo.splitlines()[:6]:
    if line.strip():
        print(f"  {line}")
print("  SYS = across the CPU link, between PCIe root complexes. The slow path.")
print("  PHB = same PCIe host bridge. The faster path.")
nvlink = run("nvidia-smi nvlink --status 2>&1 | head -2")
print(f"  NVLink     {'present' if 'Link' in nvlink and 'inactive' not in nvlink.lower() else 'NONE'}")

section("processors and memory")
print(f"  cpu        {run('lscpu | grep \"Model name\" | head -1 | cut -d: -f2 | xargs')}")
print(f"  sockets    {run('lscpu | grep \"^Socket\" | cut -d: -f2 | xargs')}")
print(f"  threads    {run('nproc')}")
mem = run("free -g | awk 'NR==2{print $2\" GB total, \"$7\" GB available\"}'")
print(f"  ram        {mem}")

section("disk")
print(f"  {run('df -h / | tail -1')}")

section("network paths")
print(f"  lan        {run('ip -4 -o addr show scope global | grep -v tailscale | awk \"{print \\$4}\" | head -1')}")
print(f"  tailscale  {run('tailscale ip -4 2>/dev/null | head -1')}")

section("software")
for tool, cmd in [("python3", "python3 --version"), ("uv", "uv --version"),
                  ("nvcc", "nvcc --version | tail -2 | head -1"),
                  ("ollama", "ollama --version 2>/dev/null | head -1"),
                  ("tmux", "tmux -V"), ("git", "git --version")]:
    path = shutil.which(tool, path=ENV["PATH"])
    print(f"  {tool:<10} {run(cmd) if path else '-- not installed'}")
torch = run("/root/Desktop/selfhostllm/.venv/bin/python -c "
            "'import torch;print(f\"{torch.__version__} cuda {torch.version.cuda} "
            "gpus {torch.cuda.device_count()}\")' 2>/dev/null")
print(f"  torch      {torch or '-- project venv not found'}")

print("\nEvery line above appears in chapter 06. If one disagrees, the chapter is stale.")
