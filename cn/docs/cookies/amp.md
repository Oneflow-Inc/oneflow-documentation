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

如果要开启 AMP 模式，只需在 [Graph](../basics/08_nn_graph.md) 模型中添加 `self.config.enable_amp(True)`，此 API 详见： [enable_amp](https://oneflow.readthedocs.io/en/master/graph.html#oneflow.nn.graph.graph_config.GraphConfig.enable_amp)。

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
