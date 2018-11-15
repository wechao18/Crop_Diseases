## Crop_Diseases
Crop Diseases Detection

代码源于Google识别API，根据数据情况做了少许修改。

深度学习框架Tensorflow1.9

[数据集下载](....)

[预训练模型下载](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

### 生成TFrecords

运行 process.py 将数据图像压缩生成TFRecords类型的数据文件，可以提高数据读取效率

```
#修改process.py 主函数路径，改为自己的下载后压缩的路径

python process.py
```
### 训练模型
```

# 配置train.sh参数
#生成的TFrecords路劲(根据自己的实际修改，下同)
DATASET_DIR=/media/zh/DATA/AgriculturalDisease20181023/tf_data
#训练过程产生的模型，迭代保存的数据位置
TRAIN_DIR=/media/zh/DATA/AgriculturalDisease20181023/check_save/resnetv1_101_finetune
#定义预训练模型定义(预训练模型下载地址上面有给出)
CHECKPOINT_PATH=/media/zh/DATA/AgriculturalDisease/check/resnet_v1_101.ckpt 

python train_image_classifier1.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=resnet_v1_101 \              #模型名称
    --checkpoint_path=${CHECKPOINT_PATH} \    #预训练模型位置，如果完全初始化训练则不需要这条
    --learning_rate=0.002 \                   #学习率
    --checkpoint_exclude_scopes=resnet_v1_101/logits \    #使用预训练训练 排除掉最后的分类层（因为和你的数据分类不一样）
    --max_number_of_steps=40000 \             #迭代次数


文件配置好后执行脚本
sh train.sh
```


### 测试模型

```
# 配置test.sh参数
# 测试数据位置
DATASET_DIR=/media/zh/DATA/AgriculturalDisease20181023/tf_data
# 训练后产生的模型
CHECKPOINT_PATH=/media/zh/DATA/AgriculturalDisease20181023/check_save/vgg16_finetune/model.ckpt-40000


python test.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=AgriculturalDisease \
    --dataset_split_name=validation \
    --model_name=resnet_v1_101 \
    --checkpoint_path=${CHECKPOINT_PATH}
```

### 特征图可视化
```
python demo.py
```
