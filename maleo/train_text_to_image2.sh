accelerate launch --config_file default_config.yaml src/train_text_to_image_sdxl.py \
--pretrained_model_name_or_path="/ML-A100/sshare-app/zhangsan/models/stable-diffusion-xl-base-1.0" \
--train_data_dir="datasets/fengtangCCG" --caption_column=additional_feature \
--use_ema --resolution=512 --center_crop --random_flip --train_batch_size=1 \
--gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=1500 \
--learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" \
--lr_warmup_steps=0 --output_dir="../output/fengtang"

        # launch.json
        # {
        #     "name": "Python: train_sdxl",
        #     "type": "python",
        #     "request": "launch",
        #     "program": "maleo/src/train_text_to_image_sdxl.py",
        #     "console": "integratedTerminal",
        #     // "justMyCode": true,
        #     "args": [
        #         "--pretrained_model_name_or_path=/ML-A100/sshare-app/zhangsan/models/stable-diffusion-xl-base-1.0",
        #         // "--train_data_dir=datasets/fengtangCCG",
        #         "--train_laion_data_dir=/ML-A100/data/multimodal_data/laion2B-en/part-00048-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet.slice.3",
        #         "--caption_column=additional_feature",
        #         "--use_ema",
        #         "--resolution=512",
        #         "--center_crop",
        #         "--random_flip",
        #         "--train_batch_size=1",
        #         "--gradient_accumulation_steps=4",
        #         "--gradient_checkpointing",
        #         "--max_train_steps=1000",
        #         "--learning_rate=1e-05",
        #         "--max_grad_norm=1",
        #         "--lr_scheduler=constant",
        #         "--lr_warmup_steps=0",
        #         "--output_dir=../output/fengtang"
        #     ]
        # }
