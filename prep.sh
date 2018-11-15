DATASET_DIR=/media/zh/DATA/AgriculturalDisease20181023/tf_data


python preprocessing.py \
    --alsologtostderr \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=train \
    --model_name=vgg_16 \



