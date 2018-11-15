CHECKPOINT_PATH=/media/zh/DATA/AgriculturalDisease20181023/check_save/vgg16_finetune/model.ckpt-40000

python demo.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=validation \
    --model_name=vgg_16_layer \
    --checkpoint_path=${CHECKPOINT_PATH}
