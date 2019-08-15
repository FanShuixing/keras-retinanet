### Installation
pip install . --user

### Need to Change
- keras_retinanet/preprocessing/csv_generator.py中的JSON_Classes，self.base_dir
- keras_retinanet/bin/train.py 中的一些参数

### train
keras_retinanet/bin/train.py csv 

### test
- 先将model转换为前向传播的model 
```
keras_retinanet/bin/convert_model.py  ./snapshots/resnet50_csv_50.h5 ./snapshots/inference.h5
```
- 运行examples中的ResNet50Retinanet.ipynb