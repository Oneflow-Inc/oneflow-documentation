#  模型的加载与保存



## 1. 保存预训练模型的意义

​	如今，自然语言处理应用已经变得无处不在。自然语言处理应用能够快速增长，很大程度上要归功于通过预训练模型实现迁移学习的概念。

​	我们知道在目前的深度学习神经网络中，训练过程是基于梯度下降法来进行参数优化的。通过迭代进而求出最小的损失函数与最优的模型权重。在进行梯度下降时，我们需要给每一个参数赋予一个初始值。那么为什么在大多数情况下要使用预训练模型来进行初始化？

​	这是因为通过预训练模型我们可以在自己的自然语言处理数据集上使用预训练模型，而不是从头构建模型来解决类似的自然语言处理问题。尽管仍然需要进行一些微调，但它已经为我们节省了大量的时间和计算资源。这对于进行模型训练的我们以及那些没有时间或者资源的使用者们，或者初学者来说是一个福报。并且通过加载已保存的预训练模型作为初始化还有其他更多的优势。比如加快梯度下降的收敛速度，更有希望获得一个低泛化误差的模型。以及降低初始化不当导致的梯度爆炸等问题。这也诠释了保存预训练的意义。



## 2. 模型的保存与加载

### 模型保存

在`oneflow`中，我们使用`save`进行模型的保存。

```
oneflow.save(obj, save_dir)
```

主要参数：

- obj：保存的对象，可以是模型。也可以是 dict。因为一般在保存模型时，不仅要保存模型，还需要保存优化器、此时对应的 epoch 等参数，这时就可以用 dict 包装起来。 

- save_dir：模型的输出路径 

```python
import os
import oneflow as flow
import oneflow.nn as nn
import oneflow.utils.vision.transforms as transforms


def load_data_mnist(
    batch_size, resize=None, root="./data/mnist", download=True, source_url=None
):
    """Download the MNIST dataset and then load into memory."""
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [transforms.Resize(resize)]
    transformer += [transforms.ToTensor()]
    transformer = transforms.Compose(transformer)

    mnist_train = flow.utils.vision.datasets.MNIST(
        root=root,
        train=True,
        transform=transformer,
        download=download,
        source_url=source_url,
    )
    mnist_test = flow.utils.vision.datasets.MNIST(
        root=root,
        train=False,
        transform=transformer,
        download=download,
        source_url=source_url,
    )
    train_iter = flow.utils.data.DataLoader(
        mnist_train, batch_size, shuffle=True
    )
    test_iter = flow.utils.data.DataLoader(
        mnist_test, batch_size, shuffle=False
    )
    return train_iter, test_iter

# 下载并设置数据
batch_size=128
train_iter, test_iter = load_data_mnist(
    batch_size, download=True, source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/"
)

def evaluate_accuracy(data_iter, net, device=None):
    n_correct, n_samples = 0.0, 0
    net.to(device)
    net.eval()
    with flow.no_grad():
        for images, labels in data_iter:
            images = images.reshape((-1, 28*28))
            images = images.to(device=device)
            labels = labels.to(device=device)
            n_correct += (net(images).argmax(dim=1).numpy() == labels.numpy()).sum()
            n_samples += images.shape[0]
    net.train()
    return n_correct / n_samples

# 设置模型需要的参数
input_size = 784
hidden_size1 = 128
hidden_size2 = 64
num_classes = 10

# 构建模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, num_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out

device = flow.device("cuda")
model = Net(input_size, hidden_size1, hidden_size2, num_classes)
print(model)
model.to(device)

loss = nn.CrossEntropyLoss().to(device)
optimizer = flow.optim.SGD(model.parameters(), lr=0.003)

# 训练循环
num_epochs = 10
final_accuracy = 0
for epoch in range(num_epochs):
    train_loss, n_correct, n_samples = 0.0, 0.0, 0
    for images, labels in train_iter:
        images = images.reshape((-1, 28*28))
        images = images.to(device=device)
        labels = labels.to(device=device)
        features = model(images)
        l = loss(features, labels).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.numpy()
        n_correct += (features.argmax(dim=1).numpy() == labels.numpy()).sum()
        n_samples += images.shape[0]
    
    # 验证精度
    test_acc = evaluate_accuracy(test_iter, model, device)
    train_acc = n_correct / n_samples
    print("epoch %d, train loss %.4f, train acc %.3f, test acc %.3f" % 
        ( epoch + 1, train_loss / n_samples, train_acc, test_acc))

# 只储存模型的参数
flow.save(model.state_dict(), "./mnist_model")
print("Saved OneFlow Model")
```

### 模型加载

由于我们保存的是模型的参数，可以使用 `load_state_dict() `进行模型的参数加载：

```python
test_net = Net(input_size, hidden_size1, hidden_size2, num_classes)
test_net.load_state_dict(flow.load("./mnist_model"))
```



## 3. OneFlow 的模型保存格式

`OneFlow` 模型是一组已经被训练好的网络的 **参数值** 。模型所保存的路径下，有多个子目录，每个子目录对应了模型的name。

同`2.`我们定义的`Net`网络：

```python
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, num_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out
```

假设在训练过程中，我们调用以下代码保存模型：

```
flow.save(model.state_dict(), "./mnist_model")
```

 那么 `mnist_model` 及其子目录结构为： 

```
mnist_model/
├── l1.bias
│   ├── meta
│   └── out
├── l1.weight
│   ├── meta
│   └── out
├── l2.bias
│   ├── meta
│   └── out
├── l2.weight
│   ├── meta
│   └── out
├── l3.bias
│   ├── meta
│   └── out
├── l3.weight
│   ├── meta
│   └── out
└── snapshot_done
```

可以看到：

- `Net` 中的网络模型，每个变量对应一个子目录 
- 以上每个子目录中，都有 `out` 和 `meta` 文件，`out` 以二进制的形式存储了网络参数的值，`meta` 以文本的形式存储了网络的结构信息 
- `snapshot_done` 是一个空文件，如果它存在，表示网络已经训练完成 