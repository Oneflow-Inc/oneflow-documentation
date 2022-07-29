# 自动混合精度训练

## AMP 简介

当我们在训练深度学习模型时，通常情况下使用的是 32 位单精度浮点数 (FP32)，而 **自动混合精度 (Automatic Mixed Precision, AMP)** 是一种允许在训练模型时同时使用 FP32 和 FP16 的技术。这样可以使得训练模型时的内存占用更少、计算更快，但由于 FP16 的数值范围比 FP32 小，因此更容易出现数值溢出的问题，同时可能存在一定误差。但大量实践证明，很多深度学习模型可以用这种技术来训练，并且没有精度损失。

## AMP 使用示例

首先，我们定义一个简单的模型、损失函数及优化器，和以往的用法完全相同。

```python
import oneflow as flow
import oneflow.nn as nn

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

model = nn.Sequential(
    nn.Linear(256, 128), 
    nn.ReLU(),
    nn.Linear(128, 10)
)
model = model.to(DEVICE)
model.train()

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)
```

如果要开启 AMP 模式，只需在 [nn.Graph](../basics/08_nn_graph.md) 模型中添加 `self.config.enable_amp(True)`，此 API 详见： [enable_amp](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.graph.graph_config.GraphConfig.enable_amp.html)。

```python
class CustomGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)
        self.config.enable_amp(True)    # 开启 AMP 模式

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return y_pred
```

然后，像以往那样开始训练等操作即可。

```python
graph_model = CustomGraph()

for _ in range(100):
    x = flow.randn(128, 256).to(DEVICE)
    y = flow.ones(128, 1, dtype=flow.int64).to(DEVICE)

    graph_model(x, y)
```

## Gradient Scaling

**Gradient Scaling (梯度缩放)** 是一种用于解决 FP16 易导致数值溢出问题的方法，其基本原理是在反向传播的过程中使用一个 scale factor 对损失和梯度进行缩放，以改变其数值的量级，从而尽可能缓解数值溢出问题。

OneFlow 提供了 `GradScaler` 来在 AMP 模式下使用 Gradient Scaling，只需要在 nn.Graph 模型的 `__init__` 方法中实例化一个`GradScaler` 对象，然后通过 [set_grad_scaler](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.Graph.set_grad_scaler.html) 接口进行指定即可，nn.Graph 将会自动管理 Gradient Scaling 的整个过程。以上文中的 `CustomGraph` 为例，我们需要在其 `__init__` 方法中添加：

```python
grad_scaler = flow.amp.GradScaler(
    init_scale=3000,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=1000,
)
self.set_grad_scaler(grad_scaler)
```

scale factor 的计算过程以及 GradScaler 的参数的含义如下：

scale factor 的大小在迭代更新中动态估计（初始值由 `init_scale` 指定），为了尽可能减少数值下溢 (underflow)，scale factor 应该更大；但如果太大，FP16 又容易发生数值上溢 (overflow)，导致出现 inf 或 NaN。动态估计的过程就是在不出现 inf 或 NaN 的情况下，尽可能增大 scale factor。在每次迭代中，都会检查是否有 inf 或 NaN 的梯度出现：

1. 如果有：此次权重更新将被忽略，并且 scale factor 将会减小（乘上 `backoff_factor`）

2. 如果没有：权重正常更新，当连续多次迭代中（由 `growth_interval` 指定）没有出现 inf 或 NaN，则 scale factor 将会增大（乘上 `growth_factor`）
