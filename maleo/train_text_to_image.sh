accelerate launch --config_file default_config.yaml ../examples/text_to_image/train_text_to_image_sdxl.py \
--pretrained_model_name_or_path="/nfs/users/zhangsan/models/stable-diffusion-xl-base-1.0" \
--train_data_dir="/root/zhangsan/sd-pretrain/datasets/fengtangCCG" --caption_column=additional_feature \
--use_ema --resolution=512 --center_crop --random_flip --train_batch_size=1 \
--gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=1500 \
--learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" \
--lr_warmup_steps=0 --output_dir="output/fengtang"
