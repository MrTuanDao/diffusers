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
  --max_train_steps=400 \
  --push_to_hub \
  --report_to="wandb" \
  --validation_prompt="a photo of SKS person" \
  --validation_steps=100 \
  --use_8bit_adam

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="tuan"
export CLASS_DIR="person"
export OUTPUT_DIR="dreambooth-tuan-with-prior"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of SKS person" \
  --class_prompt="a photo of person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub \
  --report_to="wandb" \
  --validation_prompt="a photo of SKS person" \
  --validation_steps=100 \
  --use_8bit_adam

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="tuan"
export OUTPUT_DIR="dreambooth-LoRA-tuan-without-prior"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of SKS person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub \
  --report_to="wandb" \
  --validation_prompt="a photo of SKS person" \
  --validation_epochs=100 \
  --use_8bit_adam
