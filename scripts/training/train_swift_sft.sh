# set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="HF_HOME_HERE"

export MODELSCOPE_CACHE="MODELSCOPE_CACHE_HERE  "

# NCCL debugging and workarounds
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0,lo
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,expandable_segments:True

# WANDB
export WANDB_API_KEY="WANDB_API_KEY_HERE"

NPROC_PER_NODE=8 \
swift sft \
    --model ZhipuAI/GLM-4.7-Flash \
    --train_type full \
    --dataset 'DATASET_HERE' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --agent_template glm4_7 \
    --loss_scale hermes \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --eval_steps -1 \
    --save_strategy epoch\
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 128000 \
    --save_only_model true \
    --packing true \
    --output_dir output \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --sequence_parallel_size 8 \
    --use_liger_kernel true \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --report_to wandb \