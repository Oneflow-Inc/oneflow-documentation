## oneflow_convert_tools

OneFlow 相关的模型转换工具

### oneflow_onnx

#### 简介

oneflow_onnx 工具包含两个功能，一个是将 OneFlow 导出 ONNX，另外一个是将各个训练框架导出的 ONNX 模型转换为 OneFlow 的模型。本工程已经适配了 TensorFlow/Pytorch/PaddlePaddle 框架的预训练模型通过导出 ONNX 转换为 OneFlow（我们将这一功能叫作 X2OneFlow）。

- OneFlow2ONNX 模型支持，支持 OneFlow 静态图模型转为 ONNX ，可转换由[flow.checkpoint.save](https://docs.oneflow.org/basics_topics/model_load_save.html)方法保存下来的 OneFlow 模型，详情可以参考[OneFlow2ONNX 模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/oneflow2onnx/oneflow2onnx_model_zoo.md)。
- X2OneFlow 模型支持，支持将 TensorFlow/Pytorch/PaddlePaddle的 模型通过 ONNX 转换为 OneFlow 的模型，详情可以参考[X2OneFlow 模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/x2oneflow_model_zoo.md)。
- OneFlow2ONNX 算子支持，目前稳定支持导出 ONNX Opset10，部分 OneFlow 算子支持更低的 ONNX Opset 转换，详情可以参考[OneFlow2ONNX 算子列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/oneflow2onnx/op_list.md)。
- X2OneFlow 算子支持，目前稳定支持 TensorFlow/Pytorch/PaddlePaddle 中涵盖大部分CV场景的算子，详情可以参考[X2OneFlow 算子列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/op_list.md)
- 代码生成支持，支持支持将 TensorFlow/Pytorch/PaddlePaddle 的模型通过 ONNX 转换为 OneFlow 的模型并同时生成 OneFlow 的代码，详情可以参考[X2OneFlow 代码生成模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/code_gen.md)

> 目前 OneFlow2ONNX 支持80+的 OneFlow OP 导出为 ONNX OP。X2OneFlow 支持80个 ONNX OP，50+个 TensorFlow OP，80+个 Pytorch OP，50+个PaddlePaddle OP，覆盖了大部分 CV 分类模型常用的操作。注意我们支持的 OP 和模型均为动态图 API 下的 OP 和模型，要求 PaddlePaddle 的版本>=2.0.0，TensorFlow>=2.0.0，Pytorch 无明确版本要求。目前 X2OneFlow 已经成功转换了50+个 TensorFlow/Pytorch/PaddlePaddle 官方模型。欢迎体验此项目。

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


如果你想使用 X2OneFlow（X 代表 TensorFlow/Pytorch/PaddlePaddle）需要安装对应的深度学习框架，需要安装对应的深度学习框架，依赖如下：

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

- [OneFlow2ONNX 模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/oneflow2onnx/oneflow2onnx_model_zoo.md)
- [X2OneFlow 模型列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/x2oneflow_model_zoo.md)
- [OneFlow2ONNX 算子列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/oneflow2onnx/op_list.md)
- [X2OneFlow 算子列表](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/docs/x2oneflow/op_list.md)
- [使用示例](https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/main/examples/README.md)

### nchw2nhwc_tool

#### 简介

本工具的功能是将 OneFlow 训练的 NCHW 排布的权重转换为 NHWC 排布，使用方法[在这里](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/nchw2nhwc_tool/README.md)


### save_serving_tool

#### 简介
本工具的目的是将 OneFlow 训练的模型转换为 Serving 端可用的模型，使用方法[在这里](https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/save_serving_tool/README.md)

