# OneFlow 与 PyTorch 兼容
OneFlow 的 API 与 PyTorch 对齐，用户可以很方便地将 PyTorch 模型迁移到 OneFlow。本文介绍三种方法，将使用 PyTorch 运行的代码迁移到 Oneflow。
## 1. import oneflow as torch
将原有的`import torch`的代码改为
```py
import oneflow as torch
```
就可以使用 OneFlow 训练原有模型；然而，这种方法需要手动修改所有 import torch 的文件，如果第三方库使用了 torch ，还需要更改第三方库的源码。
## 2. 使用命令行工具
OneFlow 提供了一个命令行工具，在 OneFlow 的 Python Package 内模拟了 PyTorch 环境，并将对该模块的引用都转发到实际的 OneFlow 模块中。具体的用法如下

开启模拟 PyTorch
```
eval $(oneflow-mock-torch)
```
或
```
eval $(python3 -m oneflow.mock_torch)
```

运行上述命令后，通过以下示例观察效果
```py
import torch
print(torch.__file__)
import oneflow as flow
x = torch.zeros(2, 3)
print(isinstance(x, flow.Tensor))
```

关闭模拟 PyTorch
```
eval $(oneflow-mock-torch disable)
```
或
```
eval $(python3 -m oneflow.mock_torch disable)
```
## 3. 使用 OneFlow 的内置函数
你也可以在 Python 代码中实现上述效果，只需要在 import torch 前调用 OneFlow 提供的一个函数即可。
```
from oneflow.mock_torch import mock
mock()
```
如果 torch 模块已被引入，该函数不会生效。

## 总结
由于 OneFlow 的 API 与 PyTorch 对齐，用户能够将 PyTorch 代码很方便地迁移到 OneFlow。以上介绍了三种使用 OneFlow 来训练 PyTorch 模型的方法，希望用户能够体验到 OneFlow 极致的性能。
