#!/bin/bash

MODEL_NAME="/nfs/kun2/users/gavin/vlm_checkpoints/model2"
DATA_PATH="/nfs/kun2/users/gavin/vlm_checkpoints/vlm_chat_dataset_deduped.json"
IMAGE_FOLDER="/home/gavinzhang/local_frames"
checkpoint_name="llama-twitch"
OUTPUT_DIR="/nfs/kun2/users/gavin/vlm_checkpoints/model3"

export PYTHONPATH=src:$PYTHONPATH

deepspeed --include localhost:0 src/training/train.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --disable_flash_attn2 True \
    --lora_enable False \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --dataloader_num_workers 4
