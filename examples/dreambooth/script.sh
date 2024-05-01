#!/bin/bash

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="tuan"
export OUTPUT_DIR="dreambooth-tuan-without-prior"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of SKS person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=4000 \
  --push_to_hub \
  --report_to="wandb" \
  --validation_prompt="a photo of SKS person" \
  --validation_steps=100
