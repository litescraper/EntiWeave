HF_CHECKPOINT="PATH_TO_HF_CHECKPOINT"
SAVE_DIR="/root/.cache/huggingface/megatron/EntiWeave-4B_torch_dist"

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${HF_CHECKPOINT} \
    --save ${SAVE_DIR}