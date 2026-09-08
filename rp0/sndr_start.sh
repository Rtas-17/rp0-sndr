#!/bin/bash
# RP-0 SHIP CONFIG — SNDR dev748+TQ on 1x3090, 64K, 3 users, RAW logging ON
# (resolved 09-08: TQ_MAX capped to served depth; MTP k=4 under test)
cd /root
source /root/venv-sndr/bin/activate
source /root/sndr-18001-env.sh
exec python -m vllm.entrypoints.openai.api_server \
  --model /root/models/Qwen3.8-27B-Uncensored-W4A16-RTX3090-MTP4 \
  --port 18001 --host 127.0.0.1 \
  --gpu-memory-utilization 0.91 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype turboquant_k8v4 \
  --quantization compressed-tensors \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --enable-chunked-prefill --enable-prefix-caching \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --api-key genesis-local