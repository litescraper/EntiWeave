# export HF_ENDPOINT="https://hf-mirror.com"

export HF_HOME="HF_HOME_HERE"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 vllm serve \
    EntiWeave-Q-30B-A3B \
    --max-model-len 128000\
    --port 8224 \
    --data-parallel-size 8 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    --enable-prefix-caching \
    --enable-expert-parallel \