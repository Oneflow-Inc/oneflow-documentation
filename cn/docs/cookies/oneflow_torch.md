# OneFlow 与 PyTorch 兼容
OneFlow 的 API 与 PyTorch 对齐，用户可以很方便地将 PyTorch 模型迁移到 OneFlow。本文介绍三种方法，将使用 PyTorch 运行的代码迁移到 Oneflow。
## 1. import oneflow as torch
将原有的`import torch`的代码改为
```py
import oneflow as torch
```
就可以使用OneFlow训练原有模型
## 2. 使用命令行工具
OneFlow 提供了一个命令行工具，在OneFlow的Python Package内模拟了 PyTorch 环境，并将对该模块的引用都转发到实际的 OneFlow 模块中。具体的用法如下

开启模拟 PyTorch
```
eval $(python3 -m oneflow.mock_torch)
```
或
```
eval $(oneflow-mock-torch)
```
其中，参数`mock`的默认选项为`enable`

关闭模拟 PyTorch
```
eval $(python3 -m oneflow.mock_torch disable)
```
或
```
eval $(oneflow-mock-torch disable)
```
## 3. 使用 OneFlow 的内置函数
TODO
