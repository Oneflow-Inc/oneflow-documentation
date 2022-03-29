
自然语言处理、图像等深度学习领域常使用预训练模型+微调机制来快速解决具体任务，优化器参数等实验信息往往也有存储需求。因此，在使用 OneFlow 进行模型训练时，不可避免地会遇到如何将 Pytorch 存储的预训练模型转为 OneFlow 模型的问题。



## 转换示例


将 Pytorch 存储的数据（无论是模型还是变量等等）转换成 OneFlow 的思路是 **使用 Numpy 作为二者的媒介**，具体如下示例所示。

首先我们使用 Pytorch 定义一个简单模型，并将其保存为 `pth` 文件。

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

接着，我们需要定义一个具有 **相同结构** 的 OneFlow 版本的模型。

```python
import oneflow as flow
import oneflow.nn as nn

model_flow = nn.Sequential(
    nn.Linear(128, 2), 
    nn.Softmax()
)
```

最后，我们加载 Pytorch 存储的模型文件 `model.pth`，并将其初始化到 OneFlow 版本的模型中。

```python
parameters = torch.load(save_file).state_dict()

for key, value in parameters.items():
    val = value.detach().cpu().numpy()
    parameters[key] = val

model_flow.load_state_dict(parameters)
```

我们可以通过如下代码检验模型是否加载成功。

```python
import torch
import torch.nn as nn

save_file = 'model.pth'

model_torch = nn.Sequential(
    nn.Linear(128, 2), 
    nn.Softmax()
)

torch.save(model_torch, save_file)

x = torch.randn(4, 128)
y = model_torch(x)
print("torch init: \n", y)

import oneflow as flow
import oneflow.nn as nn

model_flow = nn.Sequential(
    nn.Linear(128, 2), 
    nn.Softmax()
)

x = flow.tensor(x.numpy())
y = model_flow(x)
print("oneflow init: \n", y)

parameters = torch.load(save_file).state_dict()
for key, value in parameters.items():
    val = value.detach().cpu().numpy()
    parameters[key] = val

model_flow.load_state_dict(parameters)
y = model_flow(x)
print("torch to oneflow: \n", y)
```

输出如下：

```shell
torch init: 
 tensor([[0.4520, 0.5480],
        [0.6272, 0.3728],
        [0.6275, 0.3725],
        [0.5735, 0.4265]], grad_fn=<SoftmaxBackward0>)
oneflow init: 
 tensor([[0.9034, 0.0966],
        [0.8312, 0.1688],
        [0.3003, 0.6997],
        [0.7384, 0.2616]], dtype=oneflow.float32, grad_fn=<softmax_backward>)
torch to oneflow: 
 tensor([[0.4520, 0.5480],
        [0.6272, 0.3728],
        [0.6275, 0.3725],
        [0.5735, 0.4265]], dtype=oneflow.float32, grad_fn=<softmax_backward>)
```



## 拓展

torchvision 中有许多预训练的图像模型，那么如何将 torchvision 中的预训练模型转为 OneFlow 模型呢？以经典的  AlexNet 为例，转换代码如下所示：

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

使用预训练模型进行预测：

```python
import oneflow as flow

x = flow.randn(1, 3, 63, 63)
y = alexnet_flow(x)
```

关于 flowvision 的详细使用，欢迎访问 [flowvision documentation](https://flowvision.readthedocs.io/en/latest/index.html) 。



