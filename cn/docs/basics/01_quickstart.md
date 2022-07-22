# 快速上手

本文将以训练 FashionMNIST 数据集为例，简单地介绍如何使用 OneFlow 完成深度学习中的常见任务。通过文章中的链接可以跳转到各个子任务的专题介绍。

详细的介绍请阅读本文。让我们先从导入必要的库开始：

```python
import oneflow as flow
import oneflow.nn as nn
from flowvision import transforms
from flowvision import datasets
```
[FlowVision](https://github.com/Oneflow-Inc/vision) 是与 OneFlow 配套的、专用于计算机视觉任务的工具库，包含诸多模型、数据增强方法、数据变换操作、数据集等。我们在这里导入并使用 FlowVision 提供的数据变换模块 `transforms` 和数据集模块 `datasets`。

设置 batch size 以及运行设备：

```python
BATCH_SIZE=64

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))
```

## 加载数据

OneFlow 可以使用 [Dataset 与 DataLoader](./03_dataset_dataloader.md) 加载数据。

[flowvision.datasets](https://flowvision.readthedocs.io/en/stable/flowvision.datasets.html) 模块中包含了不少真实的数据集(如 MNIST、CIFAR10、FashionMNIST)。

我们通过 `flowvision.datasets.FashionMNIST` 获取 FashionMNIST 的训练集和测试集数据。

```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/",

)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/",
)
```

输出：

```text
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
26422272/? [00:15<00:00, 2940814.54it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw
...
```

数据集下载并解压到 `./data` 目录下。

利用 [oneflow.utils.data.DataLoader](https://oneflow.readthedocs.io/en/v0.8.1/utils.data.html?highlight=oneflow.utils.data.DataLoader#oneflow.utils.data.DataLoader) 可以将 `dataset` 封装为迭代器，方便后续训练。

```python
train_dataloader = flow.utils.data.DataLoader(
    training_data, BATCH_SIZE, shuffle=True
)
test_dataloader = flow.utils.data.DataLoader(
    test_data, BATCH_SIZE, shuffle=False
)

for x, y in train_dataloader:
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    break
```

输出：

```text
x.shape: flow.Size([64, 1, 28, 28])
y.shape: flow.Size([64])
```

> [:link: Dataset 与 Dataloader](./03_dataset_dataloader.md){ .md-button .md-button--primary}

## 搭建网络

想要搭建网络，只需要实现一个继承自 `nn.Module` 的类就可以了。在它的 `__init__` 方法中定义神经网络的结构，在它的 `forward` 方法中指定前向传播的计算逻辑。

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(DEVICE)
print(model)
```

输出：

```text
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

> [:link: 搭建神经网络](./04_build_network.md){ .md-button .md-button--primary}

## 训练模型

为了训练模型，我们需要损失函数 `loss_fn` 和优化器 `optimizer`，损失函数用于评价神经网络预测的结果与 label 的差距；`optimizer` 调整网络的参数，使得网络预测的结果越来越接近 label（标准答案），这里选用 [oneflow.optim.SGD](https://oneflow.readthedocs.io/en/master/optim.html?highlight=optim.SGD#oneflow.optim.SGD)。这一过程被称为反向传播。

```python
loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)
```

定义一个 `train` 函数进行训练，完成前向传播、计算 loss、反向传播更新模型参数等工作。

```python
def train(iter, model, loss_fn, optimizer):
    size = len(iter.dataset)
    for batch, (x, y) in enumerate(iter):
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
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

同时，定义一个 `test` 函数，用于检验模型的精度：

```python
def test(iter, model, loss_fn):
    size = len(iter.dataset)
    num_batches = len(iter)
    model.eval()
    test_loss, correct = 0, 0
    with flow.no_grad():
        for x, y in iter:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)
            test_loss += loss_fn(pred, y)
            bool_value = (pred.argmax(1).to(dtype=flow.int64)==y)
            correct += float(bool_value.sum().numpy())
    test_loss /= num_batches
    print("test_loss", test_loss, "num_batches ", num_batches)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f}")
```

然后可以开始训练，定义5轮 epoch，每训练完一个 epoch 都使用 `test` 来评估一下网络的精度：

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

输出：

```text
Epoch 1
-------------------------------
loss: 2.152148  [    0/60000]
loss: 2.140148  [ 6400/60000]
loss: 2.147773  [12800/60000]
loss: 2.088032  [19200/60000]
loss: 2.074728  [25600/60000]
loss: 2.034325  [32000/60000]
loss: 1.994112  [38400/60000]
loss: 1.984397  [44800/60000]
loss: 1.918280  [51200/60000]
loss: 1.884574  [57600/60000]
test_loss tensor(1.9015, device='cuda:0', dtype=oneflow.float32) num_batches  157
Test Error:
 Accuracy: 56.3, Avg loss: 1.901461
Epoch 2
-------------------------------
loss: 1.914766  [    0/60000]
loss: 1.817333  [ 6400/60000]
loss: 1.835239  [12800/60000]
...
```

> [:link: 自动求梯度](./05_autograd.md){ .md-button .md-button--primary}
> [:link: 反向传播与 optimizer](./06_optimization.md){ .md-button .md-button--primary}

## 保存与加载模型

调用 [oneflow.save](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.save#oneflow.save) 可以保存模型。保存的模型可以通过 [oneflow.load](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load) 加载，用于预测等工作。

```python
flow.save(model.state_dict(), "./model")
```

> [:link: 模型的加载与保存](./07_model_load_save.md){ .md-button .md-button--primary}

## 交流 QQ 群

安装或使用过程遇到问题，欢迎入群与众多 OneFlow 爱好者共同讨论交流：

加 QQ 群 **331883** 或扫描二维码：

![OneFlow 技术交流](./imgs/qq_group.png)
