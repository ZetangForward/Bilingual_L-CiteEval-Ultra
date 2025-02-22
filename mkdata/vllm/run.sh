python -m vllm.entrypoints.api_server \
    --model "/data/hf_models/Meta-Llama-3.1-8B-Instruct" \
    --served-model-name=deepseek_v3 \
    --gpu-memory-utilization=0.9 \
    --max-model-len=512 \
    --tensor-parallel-size 8 \
    --host 127.0.0.1 \
    --port 4100 \
    --trust-remote-code \
    --swap-space 4 \