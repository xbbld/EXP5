# 在 Google Colab 中完成图像分类模型的训练和导出

## 项目概述
本项目旨在通过 Google Colab 完成一个图像分类模型（花卉分类）的训练和导出，过程中涉及自定义虚拟环境以切换到兼容的 Python 3.9 版本，并安装一系列精确版本的依赖，最终克服依赖冲突问题，完成模型的训练、导出及下载。

## 环境准备
### 安装 Python 3.9 及必要工具
```bash
!sudo apt-get update -y
!sudo apt-get install python3.9 python3.9-venv python3.9-distutils curl -y
```

### 创建虚拟环境
```bash
!python3.9 -m venv /content/tflite_env
```

### 下载并安装 pip
```bash
!curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
!/content/tflite_env/bin/python get-pip.py
```

### 验证 pip 是否生效
```bash
!/content/tflite_env/bin/pip --version
```

## 依赖安装
### 安装核心依赖
```bash
! /content/tflite_env/bin/pip install -q \
  tensorflow==2.10.0 \
  keras==2.10.0 \
  numpy==1.23.5 \
  protobuf==3.19.6 \
  tensorflow-hub==0.12.0 \
  tflite-support==0.4.2 \
  tensorflow-datasets==4.8.3 \
  sentencepiece==0.1.99 \
  sounddevice==0.4.5 \
  librosa==0.8.1 \
  flatbuffers==23.5.26 \
  matplotlib==3.5.3 \
  opencv-python==4.8.0.76
```

### 安装 tflite-model-maker 本体
```bash
! /content/tflite_env/bin/pip install tflite-model-maker==0.4.2
```

### 补充缺失依赖
```bash
! /content/tflite_env/bin/pip install matplotlib_inline IPython
```

### 验证是否成功安装
```bash
! /content/tflite_env/bin/python -c "from tflite_model_maker import image_classifier; print('TFLite Model Maker 已成功导入')"
```

## 模型训练
### 编写训练脚本
```python
with open('/content/step_train.py', 'w') as f:
    f.write("""
import tensorflow as tf
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

image_path = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

model = image_classifier.create(train_data)
loss, acc = model.evaluate(test_data)
print(f'✅ 测试准确率: {acc:.4f}')
model.export(export_dir='.')
""")
```

### 执行训练脚本
```bash
! /content/tflite_env/bin/python /content/step_train.py
```

## 模型下载
```python
from google.colab import files
files.download('model.tflite')
```

## 总结
通过本项目，我们掌握了在特定环境下解决依赖冲突的有效方法，深入理解了 Python 虚拟环境的隔离作用以及不同版本库之间相互配合的重要性。同时，也体会到了在 Colab 环境中操作的独特性，对 TFLite Model Maker 的工作机制有了更直观的感受，对机器学习模型从开发到部署的完整链路有了清晰的认知。
