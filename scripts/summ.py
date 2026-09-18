import json
files=["L1-llama-3-8b-q4km-1789352930","L1-llama-3-8b-q4km-1789352959","L1-llama-3-8b-q4km-1789353237",
       "L2-SmolLM2-360M-Instruct-1789353636","L2-SmolLM2-360M-Instruct-1789354660","L2-SmolLM2-360M-Instruct-1789354738"]
print("%-32s %9s %8s %8s %8s %9s %22s %7s  %s" % ("label","med t/s","ttft ms","c1","c4","c16","VRAM 0/1/2 MiB","RSS MB","hash(medium)"))
for f in files:
    d=json.load(open("/root/Desktop/selfhostllm/bench/results/%s.json"%f))
    m=d["prompts"]["medium"]; c=d["concurrency"]; v=d["vram_peak_mb"]
    vr="/".join(str(v[str(i)]) for i in range(3))
    print("%-32s %9s %8s %8s %8s %9s %22s %7s  %s" % (d["label"],m["decode_tps"],m["ttft_ms"],
        c["1"]["total_tps"],c["4"]["total_tps"],c["16"]["total_tps"],vr,d["host_rss_peak_mb"],m["output_sha256"]))
