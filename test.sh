DATASET_DIR=/media/zh/DATA/AgriculturalDisease20181023/tf_data
CHECKPOINT_PATH=/media/zh/DATA/AgriculturalDisease20181023/check_save/vgg16_finetune/model.ckpt-40000


python test.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=validation \
    --model_name=vgg_16 \
    --checkpoint_path=${CHECKPOINT_PATH}



 # 10W 84.41
