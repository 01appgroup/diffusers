CUDA_VISIBLE_DEVICES=1 python ../examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="/nfs/users/zhangsan/models/stable-diffusion-xl-base-1.0" \
  --train_data_dir="/root/zhangsan/sd-pretrain/datasets/fengtangCCG" --caption_column=additional_feature \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=3 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="../output/pradaShirta" \
  --validation_prompt="cute dragon creature"