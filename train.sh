DATASET_DIR=/media/zh/DATA/AgriculturalDisease20181023/tf_data
TRAIN_DIR=/media/zh/DATA/AgriculturalDisease20181023/check_save/inception_resnet_v2_finetune
CHECKPOINT_PATH=/media/zh/DATA/AgriculturalDisease/check/inception_resnet_v2.ckpt

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --learning_rate=0.002 \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \


