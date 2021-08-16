# 快速上手

本文将以训练 MNIST 数据集为例，简单的介绍一套从建立到使用深度学习模型的流程， 以及OneFlow 完成深度学习中所使用的常见 API。通过文章中的链接可以找到关于某类 API 更深入的介绍。

可以通过以下命令直接体验 OneFlow 训练

```shell
wget https://docs.oneflow.org/master/code/basics/quickstart.py
python ./quickstart.py
```

详细的介绍请阅读本文。让我们先从导入必要的库开始：

```python
import oneflow as flow
import oneflow.nn as nn
import oneflow.utils.vision.transforms as transforms

BATCH_SIZE=128
```


## 加载数据

OneFlow 可以使用 [Dataset 与 Dataloader](./03_dataset_dataloader.md) 加载数据。

[oneflow.utils.vision.datasets](todo_dataset_rst.md) 模块中包含了不少真实的数据集(如 MNIST、CIFAR10、FashionMNIST)。

我们通过 `oneflow.utils.vision.datasets.MNIST` 获取 MNIST 的训练集和测试集数据。

```python
mnist_train = flow.utils.vision.datasets.MNIST(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",
)
mnist_test = flow.utils.vision.datasets.MNIST(
    root="data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",
)
```
输出：

```text
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/train-images-idx3-ubyte.gz
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/train-images-idx3-ubyte.gz to data/MNIST/raw/train-images-idx3-ubyte.gz
9913344it [00:00, 36066177.85it/s]                                                          
Extracting data/MNIST/raw/train-images-idx3-ubyte.gz to data/MNIST/raw
...
```

数据集下载并解压到 `./data` 目录下。

利用 [oneflow.utils.data.DataLoader](todo_rst_dataloader.md) 可以将 `dataset` 封装为迭代器，方便后续训练。

```pytohn
train_iter = flow.utils.data.DataLoader(
    mnist_train, BATCH_SIZE, shuffle=True
)
test_iter = flow.utils.data.DataLoader(
    mnist_test, BATCH_SIZE, shuffle=False
)
```

```python
for x, y in train_iter:
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    break
```

输出：

```text
x.shape: flow.Size([128, 1, 28, 28])
y.shape: flow.Size([128])
```

> 点击 [Dataset 与 Dataloader](./03_dataset_dataloader.md) 获取更详细信息。

## 搭建网络

想要搭建网络，只需要实现一个继承自 `nn.Module` 的类就可以了。在它的 `__init__` 方法中定义神经网络的结构，在它的 `forward` 方法中指定数据计算的顺序（正向传播）。

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
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
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
    (5): ReLU()
  )
)
```

> 点击 [搭建神经网络](./04_build_network.md) 获取更详细信息。

## 训练模型

为了训练模型，我们需要 `loss` 函数和 `optimizer`，`loss` 函数用于评价神经网络预测的结果与 label 的差距；`optimizer` 调整网络的参数，使得网络预测的结果越来越接近 label（标准答案），其格式为（input tensor，学习率）。这一过程被称为反向传播。

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)
```

定义一个 `train` 函数进行训练，完成正向传播、计算 loss、反向传播更新模型参数等工作。

```python
def train(iter, model, loss_fn, optimizer):
    size = len(iter.dataset)
    for batch, (x, y) in enumerate(iter):
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
            pred = model(x)
            test_loss += loss_fn(pred, y)
            bool_value = (pred.argmax(1).to(dtype=flow.int64)==y)
            correct += float(bool_value.sum().numpy())
    test_loss /= num_batches
    print("test_loss", test_loss, "num_batches ", num_batches)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}
```

然后可以开始训练，定义5轮 epoch，先训练后校验精度：

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_iter, model, loss_fn, optimizer)
    test(test_iter, model, loss_fn)
print("Done!")
```

输出：

```text
loss: 2.299633
loss: 2.303208
loss: 2.298017
loss: 2.297773
loss: 2.294673
loss: 2.295637
Test Error: 
 Accuracy: 22.1%, Avg loss: 2.292105 

Epoch 2
-------------------------------
loss: 2.288640
loss: 2.286367
...
```

> 点击 [自动求梯度](./05_autogra.md) 与 [反向传播与 optimizer](./06_optimization.md) 获取更详细信息。

## 保存与加载模型

调用 [oneflow.save](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.save#oneflow.save) 可以保存模型。保存的模型可以通过 [oneflow.load](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load) 加载，用于预测等工作。

```python
flow.save(model.state_dict(), "./model")
```

> 点击 [模型的加载与保存](./07_model_load_save.md) 获取更详细信息。

