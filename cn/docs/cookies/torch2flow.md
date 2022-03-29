# 将 PyTorch 预训练模型转为 OneFlow 格式

当需要使用 PyTorch 的预训练模型时，可以利用 OneFlow 与 PyTorch 模型接口对齐的特点，将 PyTorch 预训练模型，转存为 OneFlow 模型。


## 转换示例

我们将定义一个 PyTorch 模型并保存，然后展示如何将其转换成 OneFlow 模型。

```python
import torch
import torch.nn as nn

save_file = 'model.pth'

model_torch = nn.Sequential(
    nn.Linear(128, 2), 
    nn.Softmax()
)

torch.save(model, save_file)
```

运行以上代码，将得到 PyTorch 模型文件 `model.pth` 。将它转为 OneFlow 版本的模型包括两个主要步骤：

- 定义一个具有 **相同结构** 的 OneFlow 模型
- 加载 PyTorch 存储的模型文件 `model.pth`，并将模型参数初始化到 OneFlow 版本的模型中

代码如下所示：

```python
import oneflow as flow
import oneflow.nn as nn

model_flow = nn.Sequential(
    nn.Linear(128, 2), 
    nn.Softmax()
)

parameters = torch.load(save_file).state_dict()

for key, value in parameters.items():
    val = value.detach().cpu().numpy()
    parameters[key] = val

model_flow.load_state_dict(parameters)
```

我们可以发现，通过 `.state_dict()` 获取到以 `key-value` 形式存储的模型参数后，我们通过 `.detach().cpu().numpy()` 即将梯度阻断后的参数值转换成 Numpy 类型，最后通过 `.load_state_dict(parameters)` 将模型参数传递到 OneFlow 模型中。

通过上述简单示例，我们可以发现将 PyTorch 存储的数据（无论是模型还是变量等等）转换成 OneFlow 的思路是 **使用 Numpy 作为二者的媒介**，只要确保 PyTorch 和 OneFlow 定义的模型是一致的，那么无论多么复杂的模型都可以通过上述方式转换。


## 拓展

[flowvision](https://github.com/Oneflow-Inc/vision) 与 torchvision 相同，提供了许多预训练好的模型，同时 flowvision 各个模型能够做到与 torchvision 对齐。在这一部分，我们使用 flowvision，以经典的 AlexNet 为例，看看如何将 PyTorch 中 **复杂的预训练模型** 转换成 OneFlow 版本。转换代码如下所示：

```python
import torchvision.models as models_torch
import flowvision.models as models_flow

alexnet_torch = models_torch.alexnet(pretrained=True)
alexnet_flow = models_flow.alexnet()

parameters = alexnet_torch.state_dict()
for key, value in parameters.items():
    val = value.detach().cpu().numpy()
    parameters[key] = val

alexnet_flow.load_state_dict(parameters)
```

flowvision 也配备了预训练模型，设置 pretrained=True 即可：

```python
alexnet_flow = models_flow.alexnet(pretrained=True)
```

关于 flowvision 的详细使用，欢迎访问 [flowvision documentation](https://flowvision.readthedocs.io/en/latest/index.html) 。



