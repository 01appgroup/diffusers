# in project root dir, run sh maleo/docker-build_ds-pretrain.sh

export SD_PRETRAIN_IAMGE_TAG=ccr-276x7ilk-vpc.cnc.bj.baidubce.com/ai/sd-pretrain:cuda11.7-wandb-df22

docker build -t $SD_PRETRAIN_IAMGE_TAG . -f examples/text_to_image/Dockerfile_train_text_to_image

docker save $SD_PRETRAIN_IAMGE_TAG -o sd-pretrain.tar

~/bcecmd bos cp -y sd-pretrain.tar bos://bj-ai-data/sshare-app/docker-images/sd-pretrain.tar