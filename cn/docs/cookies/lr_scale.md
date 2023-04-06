# 如何分层设置学习率

在训练神经网络模型时，有时候需要为不同的网络层指定不同的学习率。例如，当我们在使用预训练的模型时，常常在预训练的主干网络模型上加入一些分支网络，这个时候我们希望在进行训练过程中，主干网络只进行微调，不需要过多改变参数，因此需要设置较小的学习率。而分支网络则需要快速地收敛，所以需要设置较大的学习率。这时设置统一的学习率很难满足要求，故需要对不同的网络层设置不同的学习率提升训练表现。

这篇文章以 MobileNet_v2 为例，展示如何在 Eager 和 Graph 模式下在不同层设置不同的学习率。

## Eager模式

### 基础实现

首先导入必要的库

```python
import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms
```

接下来设置训练参数以及运行设备

```python
BATCH_SIZE = 64
EPOCH_NUM = 1
DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))
```

使用 FlowVision 加载数据集，这里我们从国内站点加载 CIFAR-10 数据集

```python
training_data = flowvision.datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz",
)

train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True
)
```

搭建网络，使用 FlowVision 中的 MobileNet_v2 模型，并将分类器修改为输出层为 10 个神经元。并设置损失函数为交叉熵损失。

```python
model = flowvision.models.mobilenet_v2().to(DEVICE)
model.classifer = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 10))
model.train()
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
```

然后，为了使网络的不同层使用不同的学习率，需要准备一个字典，网络参数对应 `params`，学习率对应 `lr` 。

```python
param_groups = [
    {'params':model.features.parameters(), 'lr':1e-3},
    {'params':model.adaptive_avg_pool2d.parameters(), 'lr':1e-4},
    {'params':model.classifier.parameters(), 'lr':1e-5},
]
optimizer = flow.optim.SGD(param_groups)
```

参数列表 `param_groups` 将不同的参数分组保存在不同的字典中，字典属性 `params` 指定了参数，`lr` 属性指定了学习率大小。优化器接收 `params_groups` 后在更新参数时，会对不同的参数使用指定的学习率进行更新。

`param_groups` 是一个 `list`，每一项是一个字典，将不同的参数分组保存在不同的字典中，字典属性 `params` 指定了参数，`lr` 属性指定了学习率大小。优化器接收 `params_groups` 这个 `list` 后，会遍历这个 `list` 中的每一项。对其中的 `params` 使用指定的学习率 `lr` 进行更新。

接下来对模型进行训练

```python
for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

### 自定义分层的学习率衰减策略

在 Eager 模式下，不同层设置不同的学习率实现很简单，我们只需要直接指定不同参数 `lr` 。然而，我们经常需要配合学习率衰减策略一起使用，这时，上面的方法不能满足要求。

不过，我们依然可以通过动态调整 `param_groups` 中各个字典的 `lr` 属性达到目的。

在之前代码的基础上，我们为每个字典新增一个属性 `lr_decay_scale` 作为衰减因子。

```python
param_groups = [
    {'params':model.features.parameters(), 'lr':1e-3, 'lr_scale':0.9},
    {'params':model.adaptive_avg_pool2d.parameters(), 'lr':1e-4, 'lr_scale':0.8},
    {'params':model.classifier.parameters(), 'lr':1e-5, 'lr_scale':0.7},
]
optimizer = flow.optim.SGD(param_groups)
```

然后自定义一个学习率调整函数。它读取字典中 `lr_decay_scale` 属性，更新 `lr` 属性。

```python
def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= param_group["lr_scale"]
```

最后，只需要每段时间调整一次学习率即可。

```python
for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # Adjust the learning rate per 10 batches
        if batch % 10 == 0:
        	adjust_learning_rate(optimizer)
```

## Graph模式

在 Graph 模式下，同样的，我们导入必要的库，设置参数和设备，准备数据集。

```python
import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms

BATCH_SIZE = 64
EPOCH_NUM = 1
DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

training_data = flowvision.datasets.CIFAR10(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz",
)

train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True, drop_last=True
)
```

搭建模型并设置损失函数。

```python
model = flowvision.models.mobilenet_v2().to(DEVICE)
model.classifer = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 10))
model.train()
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
```

在设置优化器时，在 Eager 模式下，我们可以直接指定 `params_groups` 中的 `lr` 属性来设置学习率，而在 Graph 模型下，我们需要对不同的参数设置 `lr_scale` 属性来达到修改 `lr` 的目的。其中的 `lr_scale` 是 Graph 模式下内置的标准参数。

```python
param_groups = [
    {'params':model.features.parameters(), 'lr_scale':0.9},
    {'params':model.adaptive_avg_pool2d.parameters(), 'lr_scale':0.8},
    {'params':model.classifier.parameters(), 'lr_scale':0.7},
]
optimizer = flow.optim.SGD(param_groups, lr=1e-3)
```

一旦配置了 `lr_scale` 属性，OneFlow 会在静态图编译阶段检测到，并且在运行时使用`lr=lr*lr_scale` 来更新学习率。

接下来的使用同 [使用 Graph 做训练](basics/08_nn_graph.md#graph_2) 中一样，即：

```python
class GraphMobileNetV2(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return loss
```

训练静态图模型。

```python
graph_mobile_net_v2 = GraphMobileNetV2()

for t in range(EPOCH_NUM):
    print(f"Epoch {t+1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        loss = graph_mobile_net_v2(x, y)
        current = batch * BATCH_SIZE
        if batch % 5 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

至此，我们了解了在 Eager 模式和 Graph 模式下如何设置分层学习率。
