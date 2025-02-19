python -m vllm.entrypoints.api_server \
    --model "/nvme/big_model/DeepSeek-V3" \
    --served-model-name=deepseek_v3 \
    --gpu-memory-utilization=0.98 \
    --max-model-len=70000 \
    --tensor-parallel-size 8 \
    --host 127.0.0.1 \
    --port 4100 \
    --trust-remote-code \
    --swap-space 4 &