# Activation Checkpointing

## Activation Checkpointing 简介

Activation Checkpointing 是陈天奇团队于 2016 年在论文 [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) 中提出的一种亚线性内存优化技术，旨在减少深度学习模型训练过程中的中间激活 (activation) 带来的显存占用。Activation Checkpointing 的基本原理是**以时间换空间**，经过计算图分析后，前向过程中一些暂时用不到的中间激活特征将被删除以减少显存占用，后向过程中需要时再借助额外的前向计算恢复它们。

## Activation Checkpointing 使用示例

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

如果要开启 activation checkpointing，只需在 [Graph](../basics/08_nn_graph.md) 模型中的 Eager 模型成员上指定 `.config.activation_checkpointing = True`（如果 Graph 模型包含多个 Eager 模型成员，则需要在每个 Eager 模型成员上以这种方式指定），此 API 详见：[activation_checkpointing](https://oneflow.readthedocs.io/en/master/graph.html#oneflow.nn.graph.block_config.BlockConfig.activation_checkpointing)。
```python
class CustomGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.model.config.activation_checkpointing = True   # 开启 activation checkpointing
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)

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
