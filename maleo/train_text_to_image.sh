# 在物理机上运行分布式训练，需配置 defaunt_config.yaml
# 命令行后面加入 数字，表示第几台机器
#     train_text_to_image.sh 0

if [ $# -lt 1 ]; then
    MACHINE_RANK=0
else
    MACHINE_RANK=$1
fi

echo "MACHINE_RANK is ${MACHINE_RANK}"

##########
# 火山上目录配置
MODEL_DIR="/ML-A100/sshare-app/zhangsan/models/stable-diffusion-xl-base-1.0"
TRAIN_DATA_DIR="datasets/fengtangCCG"
# TRAIN_TARDATA_DIR="/ML-A100/sshare-app/zhangsan/laion-high-resolution-output"     # not implemented

##########
# 非云上目录配置
# MODLE_DIR="/nfs/users/zhangsan/models/stable-diffusion-xl-base-1.0"
# TRAIN_DATA_DIR="datasets/fengtangCCG"
# TRAIN_LOCALDATA_DIR="/pfs/sshare/app/zhangsan/laion-high-resolution-unpack-datafile"


# 使用下面的行替换 --train_data_dir， 支持不同格式的目录数据
# --train_localdata_dir=${TRAIN_LOCALDATA_DIR} \
# --train_tardata_dir=${TRAIN_TARDATA_DIR} \

accelerate launch --config_file default_config.yaml --machine_rank=${MACHINE_RANK} \
../examples/text_to_image/train_text_to_image_sdxl.py \
--pretrained_model_name_or_path=${MODEL_DIR} \
--train_data_dir=${TRAIN_DATA_DIR} \
--caption_column=additional_feature \
--use_ema --resolution=512 --center_crop --random_flip --train_batch_size=1 \
--gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=1500 \
--learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler="constant" \
--report_to=wandb \
--lr_warmup_steps=0 \
--output_dir="../output/fangtang"

# --train_laion_data_dir="/ML-A100//laion2B-en/part-00048-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet.slice.3" \

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
