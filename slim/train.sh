DATASET_DIR=/media/zh/DATA/AgriculturalDisease20181023/tf_data
TRAIN_DIR=/media/zh/DATA/AgriculturalDisease20181023/check_save/resnetv1_101_finetune
CHECKPOINT_PATH=/media/zh/DATA/AgriculturalDisease/check/resnet_v1_101.ckpt

python train_image_classifier1.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_101 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --learning_rate=0.002 \
    #--checkpoint_exclude_scopes=vgg_16/fc8


