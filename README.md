# Crop_Diseases
Crop Diseases Detection
代码源于Google识别API，做了少许修改

[数据集下载](....)

# 生成TFrecords

运行 process.py 将数据图像压缩生成TFRecords类型的数据文件，可以提高数据读取效率

`
python process.py
`

如果需要测试,执行测试脚本文件

`
cd slim
sh test.sh
`
