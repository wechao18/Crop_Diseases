DATASET_DIR=/media/zh/DATA/AgriculturalDisease20181023/tf_data
CHECKPOINT_PATH=/media/zh/DATA/AgriculturalDisease20181023/check_save/inception_resnet_v2_finetune/model.ckpt-40000


python test_json.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=test \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=${CHECKPOINT_PATH}
