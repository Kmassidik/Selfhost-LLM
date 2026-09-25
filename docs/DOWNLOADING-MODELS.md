# Downloading a model from Hugging Face

Everything on this box goes in `models/gguf/`. Nothing else needs to know
where a model came from, and the Arena picks up new files with no restart.

    /root/Desktop/selfhostllm/models/gguf/

---

## 0 · Decide what will actually run, before downloading anything

A 90 GB download that cannot run is a wasted afternoon. Two ceilings on this
box, and they are different ceilings:

| | | Decides |
|---|---|---|
| **24 GB** graphics memory | 3 × 8 GB | how *fast* it runs |
| **125 GB** host memory | DDR3, measured at 23.4 GB/s | whether it runs *at all* |

    under ~20 GB    fits on the cards entirely — full speed
    20–115 GB       cards plus host memory — runs, slower, in proportion
    over ~120 GB    do not download it

Then check the engine actually knows the architecture. A GGUF of a model
llama.cpp has never heard of will download perfectly and refuse to load:

    strings engines/llamacpp/build/bin/libllama.so.0 | grep -i deepseek

If the architecture name is not in there, stop. Build a newer llama.cpp first.

---

## 1 · Find the file, not the repo

Most GGUF repos hold ten or more quantizations of the same model and you want
exactly one. List them with their real sizes rather than guessing from names:

```bash
curl -s "https://huggingface.co/api/models/<owner>/<repo>?blobs=true" | python3 -c '
import sys, json, re, collections
d = json.load(sys.stdin)
g = collections.OrderedDict()
for f in d.get("siblings", []):
    n = f["rfilename"]
    if not n.endswith(".gguf"): continue
    key = re.sub(r"-\d{5}-of-\d{5}", "", n)      # split files share a quant
    g.setdefault(key, [0, 0])
    g[key][0] += f.get("size") or 0
    g[key][1] += 1
for k, (sz, n) in sorted(g.items(), key=lambda kv: kv[1][0]):
    print("  %-52s %7.1f GB  %d file(s)" % (k.split("/")[-1], sz/1e9, n))
'
```

Pick the largest one that still leaves host memory headroom. Quality rises
with quantization size and the gain is small compared to not fitting.

---

## 2 · Download it

### A single file

```bash
planning/fetch-model.sh <owner>/<repo> <filename.gguf>
```

### A split file (`-00001-of-00003`)

All shards, in order, resumable:

```bash
planning/fetch-split.sh <owner>/<repo> \
  UD-IQ2_M/Model-UD-IQ2_M-00001-of-00003.gguf \
  UD-IQ2_M/Model-UD-IQ2_M-00002-of-00003.gguf \
  UD-IQ2_M/Model-UD-IQ2_M-00003-of-00003.gguf
```

Both use `curl -C -`, so a dropped connection costs the last chunk rather than
the whole file. Run the same command again to resume.

Both run detached with `setsid`, because an ssh session that closes must not
take a six-hour download with it.

Watch progress:

```bash
watch -n 5 'ls -la models/gguf/*.gguf | awk "{printf \"%-52s %6.2f GB\n\", \$9, \$5/1e9}"'
tail -f /tmp/fetch.log            # single file
tail -f /tmp/fetch-deepseek.log   # split
```

---

## 3 · Check it actually finished

A truncated GGUF is a valid-looking file of the right name and the wrong
length. Compare against what the server says it should be:

```bash
f=Model-Q4_K_M.gguf
curl -sI -L "https://huggingface.co/<owner>/<repo>/resolve/main/$f" \
  | grep -i x-linked-size
stat -c %s models/gguf/$f
```

Equal, and the first four bytes are `GGUF`, and it is done:

```bash
head -c 4 models/gguf/$f | xxd
```

The Arena checks this itself — it reads the header's own tensor offsets and
refuses to offer a file shorter than its contents require, labelling it
*still downloading* instead. Do not rely on that to notice; it exists so a
half-finished download cannot be loaded by accident.

**Do not delete the small first shard of a split model.** It is a few
megabytes and holds the entire header. It looks like a failed download and is
not one.

---

## 4 · Gated repositories

Public repos need no credentials at all, and most GGUF quantizations are
public. A few originals are gated — Llama family, some Mistral — and return
401 or an HTML page instead of weights.

Accept the licence on the model page first, in a browser, with the account the
token belongs to. Then:

```bash
curl -fL -H "Authorization: Bearer $(cat ~/.cache/huggingface/token)" \
  -C - -o "$f" "https://huggingface.co/<owner>/<repo>/resolve/main/$f"
```

The token lives at `~/.cache/huggingface/token`, mode 600, **outside every git
repository on this box**. Check before committing anything that might have
picked one up:

```bash
git grep -n "hf_" || echo "clean"
```

A token pasted into a terminal is in that shell's history; a token pasted into
a chat is in that chat's log. Rotate at huggingface.co/settings/tokens when
either happens.

---

## 5 · Load it

Nothing to register. The Arena lists whatever is in `models/gguf/`:

    http://10.0.0.20:8090

Pick the model, pick the card count, press Load. The line under the bar says
what that choice will do before it does it — whether it fits, how much is left
for the cache, and that loading stops whatever is running first.

From the command line instead:

```bash
curl -s -X POST http://10.0.0.20:8090/api/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"Model-Q4_K_M.gguf","cards":3}'
```

---

## Worked example: DeepSeek-V4-Flash-0731

What the decision actually looked like.

**Wrong first instinct.** The original repo, `deepseek-ai/DeepSeek-V4-Flash-0731`,
is 166.9 GB of FP8 safetensors across 48 shards. Downloading and converting it
would have cost most of a day. Someone had already quantized it:
`unsloth/DeepSeek-V4-Flash-0731-GGUF`.

**Fifteen quantizations, from 82.5 GB to 161.9 GB.** `UD-Q3_K_M` at 128.1 GB
is past 125 GB of host memory and was never a candidate, however much better
it would have been. `UD-IQ2_M` at 90.9 GB leaves roughly 35 GB spare for the
operating system, the page cache and the engine's own working set.

**Architecture check.** `strings` on `libllama.so.0` reported `DEEPSEEK4` and
`DeepSeek-V4`, so the engine knows it. Without that line the download would
have been pointless.

**Three shards, sequential.** The first is 5.2 MB and holds only the header —
which is why the architecture could be read out of it hours before the weights
finished arriving: 43 layers, 256 experts, 6 active per token plus 1 shared,
4096 wide, a 1,048,576-token context.

```bash
planning/fetch-split.sh unsloth/DeepSeek-V4-Flash-0731-GGUF \
  UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-0000{1,2,3}-of-00003.gguf
```
