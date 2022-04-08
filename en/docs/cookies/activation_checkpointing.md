# Activation Checkpointing

## Introduction to Activation Checkpointing 

Activation Checkpointing is a sub-linear memory optimization technique proposed in 2016, by Chen Tianqi's team in their paper [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), aiming to reduce the memory usage during training. The basic principle of Activation Checkpointing is **exchange time for space**: After the analysis of the computational graph, some intermediate activation features that are not used temporarily in the forward process will be deleted to reduce the memory usage, and they will be restored with additional forward computation when needed in the backward process.

Taking a Transformer network as an example, the changes brought by Activation Checkpointing to the computational graph are shown in the following figure:

![Activation Checkpointing](https://oneflow-static.oss-cn-beijing.aliyuncs.com/Activation%20Checkpointing.jpg)

1. The upper part is the logical subgraph under normal conditions. T1 and T2 are the forward calculation part of Transformer Layer. The intermediate activation features obtained after each op calculation in the subgraph will continue to occupy memory. When the calculation is reversed to (T1_grad, T2_grad), these intermediate activations will be directly used for reverse calculation.

2. The lower part is the logical subgraph after Activation Checkpointing is turned on. It can be seen that the part enclosed by the dotted line is added in the middle, that is, the fake subgraph used for recalculation. Due to the existence of the fake subgraph, the normal forward subgraph does not need to save the intermediate activation when forwarding. When the reverse calculation needs to be used, the forward recalculation is temporarily performed according to the fake subgraph.​

OneFlow's static graph module `nn.Graph` already supports Activation Checkpointing. This article will introduce how to turn on it during training.

## Example of us Activation Checkpointing

First, we define a simple model consist of loss function and optimizer in exactly the same way as before.

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

To turn on activation checkpointing, you only need to specify `.config.activation_checkpointing = True` on the Eager model member (i.e. the nn.Module object) in the [nn.Graph](../basics/08_nn_graph.md) model. For more details of this API, please refer to: [activation_checkpointing](https://oneflow.readthedocs.io/en/master/graph.html#oneflow.nn.graph.block_config.BlockConfig.activation_checkpointing). For each nn.Module with "activation checkpointing" turned on, its input activations will be preserved, while other intermediate activations will be recomputed when used during backpropagation.

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

Then, you can start training and other operations as usual.

```python
graph_model = CustomGraph()

for _ in range(100):
    x = flow.randn(128, 256).to(DEVICE)
    y = flow.ones(128, 1, dtype=flow.int64).to(DEVICE)
    graph_model(x, y)
    # 其他代码...
```

##  Comparative experiment on Bert model

In order to verify the actual effect of Activation Checkpointing, we can conduct comparative experiments on the model [Bert](https://arxiv.org/abs/1810.04805). We can directly use the Bert model provided by [libai](https://github.com/Oneflow-Inc/libai). To turn on Activation Checkpointing, we just need to set `train.activation_checkpoint.enabled` to `True` in the configuration file.

First, get data ready according to [Prepare the Data and the Vocab](https://libai.readthedocs.io/en/latest/tutorials/get_started/quick_run.html#prepare-the-data-and-the-vocab). For simplicity, we use a single card for training (the GPU used in the experimental environment is NVIDIA GeForce RTX 3090, and the memory size is 24268 MB):

```bash
time python tools/train_net.py --config-file configs/bert_large_pretrain.py
```

Add the `time` command at the beginning of the whole command to measure the time spent in the training process.

The experimental results are as follows:

| Whether to Turn on Activation Checkpointing | Average Memory Usage| Time Spent |
|:-----------------------------:|:-------:|:---------:|
| No | 9141 MB | 25 minutes 16 seconds |
| Yes | 5978 MB | 33 minutes 36 seconds |

We can see from the above table that Activation Checkpointin significantly reduces the memory usage during training. At the same time, the time spent increases due to the additional forward computation required. Overall, Activation Checkpointing is a very effective solution when there is a lack of video memory.