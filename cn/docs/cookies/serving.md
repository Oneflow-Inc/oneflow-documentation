# 模型部署

【引言：介绍模型部署的常见作用（去除对训练部分的依赖；适配更多硬件、系统；更方便地把训练结果整合到业务中）】
【OneFlow 使用 Nvidia Triton 作为部署前端，让 OneFlow 用户可以训练、部署一体化，

## OneFlow 部署快速上手

【

准备一个可以马上看到效果的例子，重点体现整体 demo 的运行效果
类似这样展示？

```
wget url:://xxx.zip | unzip xxx.zip && docker run ...
```

然后用户访问 localhost:xxxx 就可以玩某个 demo

】

【
回头稍加解释下以上 xxx.zip 解压后的文件

1. 模型文件：通过 OneFlow 训练后的模型转化而来，具体见下文流程详解
2. xxx_models：它的组织结构，是遵循triton的格式的，具体可以参考 triton url xxx
3. demo 代码：它隶属于业务逻辑，可以看到 xxx 代码，是通过 rpc 方式调用模型的推理能力的，triton 还提供了其它调用方式，具体见 xxxx
】

【以下会详细介绍 OneFlow 从训练到部署的详细流程】

## OneFlow 从训练到部署流程解析

### 模型训练与保存

### 模型导出为部署格式

【利用 Graph 将模型导出 triton 格式的代码，并适当介绍相关原理（如tracing）】

### 启动 serving

【docker 启动的命令解释和更详细的参考资料指引】

## 扩展阅读：模型部署


