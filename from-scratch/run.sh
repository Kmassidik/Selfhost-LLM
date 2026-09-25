cd "$(dirname "$0")"
V=/root/Desktop/selfhostllm/.venv/bin
echo "===== 1-GPU ====="
STEPS=2500 CUDA_VISIBLE_DEVICES=0 $V/python gpt.py 2>&1 | grep -E "cfg\]|RESULT|sample"
echo "===== 3-GPU DDP ====="
STEPS=2500 CUDA_VISIBLE_DEVICES=0,1,2 $V/torchrun --nproc_per_node=3 --standalone gpt.py 2>&1 | grep -E "cfg\]|RESULT|sample"
echo GPT_DONE
