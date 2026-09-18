from huggingface_hub import hf_hub_download
print(hf_hub_download("empero-ai/Qwen3.8-35B-A3B-Distill-GGUF","Qwen3.8-35B-A3B-Q4_K_M.gguf",local_dir="models/gguf/empero"))
