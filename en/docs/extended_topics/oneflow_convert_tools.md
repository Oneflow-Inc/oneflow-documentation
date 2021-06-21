## oneflow_convert_tools


### oneflow_onnx

[![PyPI version](https://img.shields.io/pypi/v/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PyPI license](https://img.shields.io/pypi/l/oneflow-onnx.svg)](https://pypi.python.org/pypi/oneflow-onnx/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/oneflow_convert_tools/pulls)

#### Introduction

oneflow_onnx tool package includes two major functions: one is to export OneFlow out of ONNX, while the other is to transform ONNX models, which are obtained from other training frameworks, into Oneflow models. This tool package has already been adapted to TensorFlow/Pytorch/PaddlePaddle pre-trained models. The process of oneflow_onnx extracting ONNX and transforming it into OneFlow's format is called X2OneFlow (X representing TensorFlow/Pytorch/PaddlePaddle).

- OneFlow2ONNX models are supported. Specifically, OneFlow's lazy mode model can be transfomed into ONNX's format. Transformable OneFlow model can be obtained by using the method explained on [flow.checkpoint.save ](https://docs.oneflow.org/basics_topics/model_load_save.html). For more information, please refer to [OneFlow2ONNX Model List](docs/oneflow2onnx/oneflow2onnx_model_zoo.md).
- X2OneFlow models are supported. TensorFlow/Pytorch/PaddlePaddle model can be transformed into OneFlow's format through ONNX.
- OneFlow2ONNX operators are supported. Currently, oneflow_onnx is fully capable of exporting ONNX Opset10, and parts of OneFlow operator can transform ONNX Opsets that are in lower order. Please refer to [OneFlow2ONNX Operator Lists](docs/oneflow2onnx/op_list.md) for more information.
- X2OneFlow operators are supported. Currently, oneflow_onnx is fully capable of supporting most CV operators in TensorFlow/Pytorch/PaddlePaddle. Please refer to [X2OneFlow Operator Lists](docs/x2oneflow/op_list.md) for more information.
- Code generation is also supported. oneflow_onnx is able to generate OneFlow code and transforming models simultaneously . Please refer to [X2OneFlow Code Generation List](docs/x2oneflow/code_gen.md) for more information.

> To sum up,
>
> - OneFlow2ONNX can support over 80 ONNX OP
> - X2OneFlow can support 80 ONNX OP,  50+ TensorFlow OP, 80+ Pytorch OP, and 50+ PaddlePaddle OP
>
> which covers most operations when doing CV model classifications. Since the OPs and models we support are all in eager mode API, users are required to install versions of PaddlePaddle >= 2.0.0, TensorFlow >= 2.0.0, and there is no specific requirements for Pytorch. Until now, X2OneFlow has successfully transformed 50+ official models from TensorFlow/Pytorch/PaddlePaddle, and you're always welcomed to experience our product.

#### Environment Dependencies

##### User's Environment Configuration

```sh
python>=3.5
onnx>=1.8.0
onnx-simplifier>=0.3.3
onnxoptimizer>=0.2.5
onnxruntime>=1.6.0
oneflow (https://github.com/Oneflow-Inc/oneflow#install-with-pip-package)
```

If you'd llike to use X2OneFlow, the following versions of deep learning frameworks are needed:

```sh
pytorch>=1.7.0
paddlepaddle>=2.0.0
paddle2onnx>=0.6
tensorflow>=2.0.0
tf2onnx>=1.8.4
```

#### Installation

##### Method 1

```sh
pip install oneflow_onn
```

**Method 2**

```
git clone https://github.com/Oneflow-Inc/oneflow_convert_tools
cd oneflow_onnx
python3 setup.py install
```

#### Usage

Please refer to [Examples](examples/README.md)

#### Related Documents

- [OneFlow2ONNX Model List](docs/oneflow2onnx/oneflow2onnx_model_zoo.md)
- [X2OneFlow Model List](docs/x2oneflow/x2oneflow_model_zoo.md)
- [OneFlow2ONNX Operator List](docs/oneflow2onnx/op_list.md)
- [X2OneFlow Operator List](docs/x2oneflow/op_list.md)
- [Examples](examples/README.md)

### nchw2nhwc_tool

#### Introduction

This tool is to transform NCHW, which is trained through OneFlow, into NHWC Format. Please click [here](nchw2nhwc_tool/README.md) for more information


### save_serving_tool

#### Introduction

This tool is to transform OneFlow models into models that can be used on the Serving end. Please click [here](save_serving_tool/README.md) for more information


