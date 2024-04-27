#!/bin/bash

git clone https://huggingface.co/datasets/lewtun/corgi

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./corgi"

accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<tuan-dao-corgi>" \
  --initializer_token="corgi" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --push_to_hub \
  --output_dir="textual_inversion_corgi" \
  --report_to="wandb" \
  --validation_prompt="a <tuan-dao-corgi>" \
  --validation_epochs=100
