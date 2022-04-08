# Activation Checkpointing

## Activation Checkpointing 简介

Activation Checkpointing 是陈天奇团队于 2016 年在论文 [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) 中提出的一种亚线性内存优化技术，旨在减少训练过程中的中间激活(activation)带来的显存占用。Activation Checkpointing 的基本原理是 **以时间换空间** ：经过计算图分析后，前向过程中一些暂时用不到的中间激活特征将被删除以减少显存占用，后向过程中需要时再借助额外的前向计算恢复它们。

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

## 在 BERT 模型上的对比实验

为了验证 Activation Checkpointing 的实际效果，我们可以在 [BERT](https://arxiv.org/abs/1810.04805) 模型上进行对比实验。可以直接使用 [libai](https://github.com/Oneflow-Inc/libai) 库提供的 BERT 模型，只需通过在配置文件中将 `train.activation_checkpoint.enabled` 设置为 `True` 就可以开启 Activation Checkpointing。

首先，按照 [Prepare the Data and the Vocab](https://libai.readthedocs.io/en/latest/tutorials/get_started/quick_run.html#prepare-the-data-and-the-vocab) 准备好数据。为简单起见，我们使用单卡训练（实验环境使用的 GPU 为 NVIDIA GeForce RTX 3090，显存大小为 24268 MB）：

```bash
time python tools/train_net.py --config-file configs/bert_large_pretrain.py
```

在命令最开头加上 `time` 命令来计量训练过程所耗费的时间。

实验结果如下：

| 是否开启 Activation Checkpointing | 平均显存占用  | 训练完成所用时间  |
|:-----------------------------:|:-------:|:---------:|
| 否                             | 9141 MB | 25 分 16 秒 |
| 是                             | 5978 MB | 33 分 36 秒 |

从上表可以看出，Activation Checkpointing 显著减少了训练时的显存占用。同时，训练所用时间由于需要额外的前向计算而有所增加。总体来说，当缺乏显存时，Activation Checkpointing 不失为一种很有效的解决办法。
