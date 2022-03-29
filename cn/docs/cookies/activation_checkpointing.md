# Activation Checkpointing

## Activation Checkpointing 简介

Activation Checkpointing 是陈天奇团队于 2016 年在论文 [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) 中提出的一种亚线性内存优化技术，旨在减少训练过程中的显存占用。Activation Checkpointing 的基本原理是 **以时间换空间** ：经过计算图分析后，前向过程中一些暂时用不到的中间激活特征将被删除以减少显存占用，后向过程中需要时再借助额外的前向计算恢复它们。

OneFlow 的静态图模块 `nn.Graph` 已经支持 Activation Checkpointing，本文将介绍如何在训练中开启它。

## Activation Checkpointing 使用示例

首先，我们定义一个简单的模型（由两部分组成）、损失函数及优化器，和以往的用法完全相同。

```python
import oneflow as flow
import oneflow.nn as nn

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

model_part1 = nn.Sequential(
    nn.Linear(256, 128), 
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU()
)
model_part1 = model_part1.to(DEVICE)
model_part1.train()

model_part2 = nn.Sequential(
    nn.Linear(64, 32), 
    nn.ReLU(),
    nn.Linear(32, 10)
)
model_part2 = model_part2.to(DEVICE)
model_part2.train()

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD([{'params': model_part1.parameters()},
                            {'params': model_part2.parameters()}],
                           lr=1e-3)
```

如果要开启 activation checkpointing，只需在 [nn.Graph](../basics/08_nn_graph.md) 模型中的 Eager 模型成员 (即 nn.Module 对象) 上指定 `.config.activation_checkpointing = True`。此 API 详见：[activation_checkpointing](https://oneflow.readthedocs.io/en/master/graph.html#oneflow.nn.graph.block_config.BlockConfig.activation_checkpointing)。对于每个打开 "activation checkpointing" 的 nn.Module，其输入 activation 将会被保留，而其它中间 activation 在反向传播过程中被使用时会被重新计算。

```python
class CustomGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model_part1 = model_part1
        self.model_part2 = model_part2
        # 在连续的两个 nn.Module 上开启 activation checkpointing
        self.model_part1.config.activation_checkpointing = True
        self.model_part2.config.activation_checkpointing = True
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)

    def build(self, x, y):
        y_pred = self.model_part2(self.model_part1(x))
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return y_pred, loss
```

然后，像以往那样开始训练等操作即可。

```python
graph_model = CustomGraph()

for _ in range(100):
    x = flow.randn(128, 256).to(DEVICE)
    y = flow.ones(128, 1, dtype=flow.int64).to(DEVICE)
    graph_model(x, y)
    # 其他代码...
```
