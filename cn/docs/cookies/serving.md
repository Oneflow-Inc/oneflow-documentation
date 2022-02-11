# 模型部署

模型训练好后，需要经过“模型部署”才能够集成到产品中并上线。因为产品上线时的软硬件环境、模型与业务模块的对接方式都可能变化，所以部署的解决方案也多种多样。比如某些方案会将训练好的模型转为其他格式（如 ONNX）后，再依赖特定的 runtime 部署；某些方案会直接使用 汇编/C/C++ 等能生成 native code 的语言重新实现模型，以追求硬件适配或部署性能。

OneFlow 通过对接了 [Triton Inference Server](https://github.com/Triton-inference-server/server)，做到了训练、部署一体化。

OneFlow 用户训练好模型后，可以直接通过 Triton 部署模型，并借助 Triton 丰富的特性，如 Dynamic batching、Model Pipelines、HTTP/gRPC 接口等，并快速高效地集成到线上产品中。

本文内容组织如下：

- OneFlow 部署快速上手
- OneFlow Serving 架构介绍
- OneFlow 从训练到部署流程解析

## OneFlow 部署快速上手

OneFlow Cloud 上准备了一个 [垃圾分类部署示例：基于 OneFlow-Serving](#xxxx) 项目，参照项目说明用户可以一键部署项目，并且查看项目运行效果。

![](./imgs/oneflow-serving-demo.png)

分析项目代码可以发现，有以下几个关键处：

- [run.sh](#xxx) 中通过 docker 启动了 Triton 服务与 WEB 应用服务：
```bash
/opt/tritonserver/bin/tritonserver --model-store $(pwd)/model_repo > 1.txt && python3 server.py
```

- [server.py](#xxx) 中只是简单和普通的 URL 路由，真正做推理工作是由 [infer.py](#xxx) 中的 `stylize` 完成的。`stylize` 函数内部，通过 HTTP 与 Triton 服务端交互得到推理结果。
```python
def stylize(content_path, output_path, style='udnie'):
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    ...
    inputs.append(httpclient.InferInput('INPUT_0', image.shape, 'FP32'))
    ...
    outputs.append(httpclient.InferRequestedOutput('OUTPUT_0', binary_data=True))
    ...
```

- 预训练模型放置在 [model_repo](#yyy) 下，它按照 Triton 的约定组织格式

这个简单的在线示例展示了 OneFlow 模型如何通过 Triton 部署，同时也展示业务模块如何与 Triton 服务端交互获取推理结果。 

接下来，我们会详细介绍 OneFlow 从训练到部署的详细流程。

## OneFlow 从训练到部署流程解析

我们首先通过下图总体了解 OneFlow 与 Triton 的关系。

![](https://github.com/triton-inference-server/server/raw/main/docs/images/arch.jpg)

从上图可以知晓，Triton 处于联接客户端与 OneFlow 的位置：Triton 提供了 HTTP、gRPC 以及 C 接口，使得用户可以灵活地发起推理请求并得到结果，Triton 内部还提供了任务调度等的内置功能，这使得其它业务很容易与 Triton 集成。

在 Triton 的架构中，OneFlow 与 Model Repository 一起，为 Triton 提供后端推理能力。Triton 对 Model Repository 的组织格式有预设的要求，OneFlow 提供了对应的接口，将训练好的模型导出为 Triton 要求的组织格式。

了解这些基本概念后，让我们详细解析 OneFlow 从模型训练到部署的流程：

- 模型保存
- 模型配置
- 启动服务
- 客户端发送请求

### 模型保存

### 模型配置


### 启动服务


### 客户端发送请求



