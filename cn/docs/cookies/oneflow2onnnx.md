# OneFlow 与 ONNX 交互

本教程主要介绍 OneFlow 与 ONNX 进行交互的用法，包括 ONNX 简介、如何将 OneFlow 模型导出为 ONNX 模型，以及如何使用 ONNX 模型进行推理。


## ONNX 简介

[ONNX](https://onnx.ai/index.html) 的全称为 Open Neural Network Exchange (开放神经网络交换)，是一种针对机器学习算法所设计的开放式文件格式标准，用于存储训练好的算法模型。许多主流的深度学习框架（如 OneFlow、PyTorch、TensorFlow、MXNet）都支持将模型导出为 ONNX 模型。ONNX 使得不同的深度学习框架可以以一种统一的格式存储模型数据以及进行交互。另外，ONNX 有相应的运行时（Runtime）—— [ONNX Runtime](https://onnxruntime.ai/)，便于在多种平台（Linux、Windows、Mac OS、Android、iOS等）及多种硬件（CPU、GPU等）上进行模型部署和推理。

### ONNX 相关库
ONNX 对应多个相关库，常见的几个库的功能如下所述。本教程中主要涉及 onnxruntime-gpu。

1. [onnx](https://github.com/onnx/onnx): ONNX 模型格式标准

2. [onnxruntime & onnxruntime-gpu](https://github.com/microsoft/onnxruntime): ONNX 运行时，用于加载 ONNX 模型进行推理。onnxruntime 和 onnxruntime-gpu 分别支持 CPU 推理和 GPU推理

3. [onnx-simplifier](https://github.com/daquexian/onnx-simplifier): 用于简化 ONNX 模型的结构，例如消除结果恒为常量的算子
   
4. [onnxoptimizer](https://github.com/onnx/optimizer): 用于通过图变换等方式优化 ONNX 模型


## 将 OneFlow 模型导出为 ONNX 模型
[oneflow-onnx](https://github.com/Oneflow-Inc/oneflow_convert) 是 OneFlow 团队提供的模型转换工具，支持将 OneFlow 静态图模型导出为 ONNX 模型。目前 oneflow-onnx 支持 80 多种 OneFlow OP 导出为 ONNX OP，具体可参见：[OneFlow2ONNX 支持的OP列表](https://github.com/Oneflow-Inc/oneflow_convert/blob/main/docs/oneflow2onnx/op_list.md)。

### 安装 oneflow-onnx
oneflow-onnx 独立于 OneFlow，需要单独安装。安装方式如下所述：

通过 pip 安装：
```python
pip install oneflow-onnx
```

通过源码安装：
```python
git clone https://github.com/Oneflow-Inc/oneflow_convert
cd oneflow_convert
python3 setup.py install
```

### oneflow-onnx 的使用方法
要将 OneFlow 静态图模型导出为 ONNX 模型，只需调用 `export_onnx_model` 函数。

```python
from oneflow_onnx.oneflow2onnx.util import export_onnx_model

export_onnx_model(graph,
                  external_data=False, 
                  opset=None, 
                  flow_weight_dir=None, 
                  onnx_model_path="/tmp", 
                  dynamic_batch_size=False)
```
各参数的含义如下:

1. graph: 需要转换的 graph ( [Graph](../basics/08_nn_graph.md) 对象)

2. external_data: 将权重另存为 ONNX 模型的外部数据，通常是为了避免 protobuf 的 2GB 文件大小限制

3. opset: 指定转换模型的版本 (int，默认为 `oneflow_onnx.constants.PREFERRED_OPSET`)

4. flow_weight_dir: OneFlow 模型权重的保存路径

5. onnx_model_path: 导出的 ONNX 模型保存路径

6. dynamic_batch_size: 导出的 ONNX 模型是否支持动态 batch，默认为False


另外，oneflow-onnx 还提供了一个名为 `convert_to_onnx_and_check` 的函数，用于转换并检查转换出的 ONNX 模型。其中的检查指的是将同样的输入分别送入原本的 OneFlow 模型和转换后的 ONNX 模型，然后比较两个输出中对应的每个数值之差是否在合理的误差范围内。

```python
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check

convert_to_onnx_and_check(...)
```
`convert_to_onnx_and_check` 函数的参数是 `export_onnx_model` 函数的参数的超集，可以额外传入 `print_outlier=True` 来输出检查过程中发现的超出合理误差范围内的异常值。

### 导出模型时的注意点
- 在导出模型之前，需要将模型设置成 eval 模式，因为 dropout、BatchNorm 等操作在训练和推理模型下的行为不同
- 在构建静态图模型时，需要指定一个输入，此输入的值可以是随机的，但要保证它是正确的数据类型和形状
- ONNX 模型接受的输入的形状是固定的，batch 维度的大小可以是变化的，通过将 `dynamic_batch_size` 参数设为 `True` 可以使得导出的 ONNX 模型支持动态 batch 大小
- oneflow-onnx 必须使用静态图模型（Graph 模式）作为导出函数的参数。对于动态图模型（Eager 模式），需要将动态图模型构建为静态图模型，可参见下文的示例。


## 用法示例
在本节中，将以常见的 ResNet-34 模型为例，介绍将 OneFlow 模型导出为 ONNX 模型并进行推理的流程。我们在此直接使用 [FlowVision](https://github.com/Oneflow-Inc/vision) 库提供的 ResNet-34 模型，并使用 FlowVision 提供的在 ImageNet 数据集上训练得到的 ResNet-34 权重。

### 导出为 ONNX 模型
导入相关依赖：
```python
import oneflow as flow
from oneflow import nn
from flowvision.models import resnet34
from oneflow_onnx.oneflow2onnx.util import convert_to_onnx_and_check
```

使用动态图模型构建静态图模型，详情请参见：[静态图模块 nn.Graph](../basics/08_nn_graph.md)
```python
class ResNet34Graph(nn.Graph):
    def __init__(self, eager_model):
        super().__init__()
        self.model = eager_model

    def build(self, x):
        return self.model(x)
```

将 OneFlow 静态图模型导出为 ONNX 模型：
```python
# 模型参数存储目录
MODEL_PARAMS = 'checkpoints/resnet34'

params = flow.load(MODEL_PARAMS)
model = resnet34()
model.load_state_dict(params)

# 将模型设置为 eval 模式
model.eval()

resnet34_graph = ResNet34Graph(model)
# 构建出静态图模型
resnet34_graph._compile(flow.randn(1, 3, 224, 224))

# 导出为 ONNX 模型并进行检查
convert_to_onnx_and_check(resnet34_graph, 
                          flow_weight_dir=MODEL_PARAMS, 
                          onnx_model_path="./", 
                          print_outlier=True,
                          dynamic_batch_size=True)
```
运行完毕后，可以在当前目录中找到名为 `model.onnx` 的文件，即导出的 ONNX 模型。

### 使用 ONNX 模型进行推理
进行推理之前，要保证已经安装了 ONNX Runtime, 即 onnxruntime 或 onnxruntime-gpu。在本教程的实验环境中，安装的是 onnxruntime-gpu 以调用 GPU 进行计算。

我们使用下面这张图像作为模型的输入：
<div align="center">
    <img alt="Demo Image" src="./imgs/cat.jpg" width="300px">
</div>


导入依赖：
```python
import numpy as np
import cv2
from onnxruntime import InferenceSession
```

定义一个函数用于将图像预处理为 ONNX 模型所接受的格式和尺寸：
```python
def preprocess_image(img, input_hw = (224, 224)):
    h, w, _ = img.shape
    
    # 使用图像的较长边确定缩放系数
    is_wider = True if h <= w else False
    scale = input_hw[1] / w if is_wider else input_hw[0] / h

    # 对图像进行等比例缩放
    processed_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # 归一化
    processed_img = np.array(processed_img, dtype=np.float32) / 255

    # 将图像填充到 ONNX 模型预设尺寸
    temp_img = np.zeros((input_hw[0], input_hw[1], 3), dtype=np.float32)
    temp_img[:processed_img.shape[0], :processed_img.shape[1], :] = processed_img
    processed_img = temp_img
    
    # 调整轴的顺序并在最前面添加 batch 轴  
    processed_img = np.expand_dims(processed_img.transpose(2, 0, 1), axis=0)

    return processed_img
```

接下来，使用 ONNX 模型进行推理，主要步骤包括：创建一个 InferenceSession 对象，然后调用其 `run` 方法进行推理。

在 onnxruntime(-gpu) 1.9 及以上版本中，创建 InferenceSession 对象时需要显式指定 `providers` 参数来选择使用的硬件。对于 onnxruntime-gpu，可以指定的值包括 `TensorrtExecutionProvider`、`CUDAExecutionProvider`、`CPUExecutionProvider`。推理时将会按照此先后顺序依次尝试调用对应的硬件进行计算。

ONNX 模型的输入数据的类型是一个 dict，其 keys 为导出 ONNX 模型时的输入名称 "input names"，values 为 NumPy 数组类型的实际输入数据。可以通过 InferenceSession 对象的 `get_inputs` 方法获取"input names"，该方法的返回值是 `onnxruntime.NodeArg` 类型的对象组成的 list，对于 NodeArg 对象，可使用其 `name` 属性获取 str 类型的名称。在本教程中，输入只有图像数据本身，因此可以通过在 InferenceSession 对象上调用 `.get_inputs()[0].name`，获取输入对应的 "input names"，其值为 `_ResNet34Graph_0-input_0/out`，将此值作为 key 构造输入 ONNX 模型的 dict。当然，也可以不预先指定，而在运行时动态获取。

```python
# 从文件中读取 ImageNet 数据集的类别名称
with open('ImageNet-Class-Names.txt') as f:
    CLASS_NAMES = f.readlines()

# 读取图像文件并使用 `preprocess_image` 函数进行预处理
img = cv2.imread('cat.jpg', cv2.IMREAD_COLOR)
img = preprocess_image(img)

# 创建一个 InferenceSession 对象
ort_sess = InferenceSession('model.onnx', providers=['TensorrtExecutionProvider',
                                                     'CUDAExecutionProvider',
                                                     'CPUExecutionProvider'])
# 调用 InferenceSession 对象的 `run` 方法进行推理
results = ort_sess.run(None, {"_ResNet34Graph_0-input_0/out": img})

# 输出推理结果
print(CLASS_NAMES[np.argmax(results[0])])
```

InferenceSession 对象的 `run` 方法的输出是 NumPy 数组构成的 list，每个 NumPy 数组对应一组输出。因为只有一组输入，所以取出索引为 0 的元素作为输出，此元素的形状是 `(1, 1000)`，对应于 1000 个类别的概率 (如果将 n 张图像作为一个 batch 输入，此元素的形状将是 `(n, 1000)`)。通过 `np.argmax` 获取概率最大的类别对应的索引后，将索引映射为类别名称。

运行以上代码，得到：
```text
(base) root@training-notebook-654c6f-654c6f-jupyter-master-0:/workspace# python infer.py 
285: 'Egyptian cat',
```

以上推理是在 Python 环境下进行的，实际使用时可以根据部署环境选择不同的 ONNX Runtime 来使用导出的 ONNX 模型。
