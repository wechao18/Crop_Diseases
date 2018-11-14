DATASET_DIR=/media/zh/DATA/AgriculturalDisease/tf_data
TRAIN_DIR=/media/zh/DATA/AgriculturalDisease/check_save/resnet_v1_101_finetune
CHECKPOINT_PATH=/media/zh/DATA/AgriculturalDisease/check/resnet_v1_101.ckpt

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_101 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=resnet_v1_101/logits \
    --trainable_scopes=resnet_v1_101/logits

