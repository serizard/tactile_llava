#!/bin/bash
PROMPT_VERSION=llava_v1
DATA_ROOT=./dataset/gpt_instruction
IMAGE_ROOT=./dataset/TacCOCO
model_size=7b
NUM_GPUS=8
# Get the directory containing the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root directory 
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH


deepspeed --num_gpus=$NUM_GPUS llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path mucai/vip-llava-$model_size \
    --version $PROMPT_VERSION \
    --data_path $DATA_ROOT/vip-llava-tactile-task-formatted.json \
    --image_folder $DATA_ROOT \
    --vision_tower clip_4layers_336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/vip-llava-$model_size-task-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
