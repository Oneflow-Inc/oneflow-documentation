## oneflow_convert_tools

OneFlow相关的模型转换工具

### oneflow_onnx

#### 简介

oneflow_onnx工具包含两个功能，一个是将OneFlow导出ONNX，另外一个是将各个训练框架导出的ONNX模型转换为OneFlow的模型。本工程已经适配了TensorFlow/Pytorch/PaddlePaddle框架的预训练模型通过导出ONNX转换为OneFlow（我们将这一功能叫作X2OneFlow）。

- OneFlow2ONNX模型支持，支持OneFlow静态图模型转为ONNX，可转换由[flow.checkpoint.save](https://docs.oneflow.org/basics_topics/model_load_save.html)方法保存下来的OneFlow模型，详情可以参考[OneFlow2ONNX模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/oneflow2onnx/oneflow2onnx_model_zoo.md)。
- X2OneFlow模型支持，支持将TensorFlow/Pytorch/PaddlePaddle的模型通过ONNX转换为OneFlow的模型，详情可以参考[X2OneFlow模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/x2oneflow_model_zoo.md)。
- OneFlow2ONNX算子支持，目前稳定支持导出ONNX Opset10，部分OneFlow算子支持更低的ONNX Opset转换，详情可以参考[OneFlow2ONNX算子列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/oneflow2onnx/op_list.md)。
- X2OneFlow算子支持，目前稳定支持TensorFlow/Pytorch/PaddlePaddle中涵盖大部分CV场景的算子，详情可以参考[X2OneFlow算子列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/op_list.md)
- 代码生成支持，支持支持将TensorFlow/Pytorch/PaddlePaddle的模型通过ONNX转换为OneFlow的模型并同时生成OneFlow的代码，详情可以参考[X2OneFlow代码生成模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/code_gen.md)

> 目前OneFlow2ONNX 支持80+的OneFlow OP导出为ONNX OP。X2OneFlow支持80个ONNX OP，50+个TensorFlow OP，80+个Pytorch OP，50+个PaddlePaddle OP，覆盖了大部分CV分类模型常用的操作。注意我们支持的OP和模型均为动态图API下的OP和模型，要求PaddlePaddle的版本>=2.0.0，TensorFlow>=2.0.0，Pytorch无明确版本要求。目前X2OneFlow已经成功转换了50+个TensorFlow/Pytorch/PaddlePaddle官方模型。欢迎体验此项目。

#### 环境依赖

##### 用户环境配置

```sh
python>=3.5
onnx>=1.8.0
onnx-simplifier>=0.3.3
onnxoptimizer>=0.2.5
onnxruntime>=1.6.0
oneflow (https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)
```


如果你想使用X2OneFlow（X代表TensorFlow/Pytorch/PaddlePaddle）需要安装对应的深度学习框架，需要安装对应的深度学习框架，依赖如下：

```sh
pytorch>=1.7.0
paddlepaddle>=2.0.0
paddle2onnx>=0.6
tensorflow>=2.0.0
tf2onnx>=1.8.4
```

#### 安装

##### 安装方式1

```sh
pip install oneflow_onnx
```

#### 安装方式2

```
git clone https://github.com/Oneflow-Inc/oneflow_convert_tools
cd oneflow_onnx
python3 setup.py install
```

#### 使用方法

请参考[使用示例](https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/main/examples/README.md)

#### 相关文档

- [OneFlow2ONNX模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/oneflow2onnx/oneflow2onnx_model_zoo.md)
- [X2OneFlow模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/x2oneflow_model_zoo.md)
- [OneFlow2ONNX算子列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/oneflow2onnx/op_list.md)
- [X2OneFlow算子列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/op_list.md)
- [使用示例](https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/main/examples/README.md)

### nchw2nhwc_tool

#### 简介

本工具的功能是将OneFlow训练的NCHW排布的权重转换为NHWC排布，使用方法[在这里](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/nchw2nhwc_tool/README.md)


### save_serving_tool

#### 简介
本工具的目的是将OneFlow训练的模型转换为Serving端可用的模型，使用方法[在这里](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/save_serving_tool/README.md)

