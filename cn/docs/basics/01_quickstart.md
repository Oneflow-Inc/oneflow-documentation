# 快速上手

本文将以训练 MNIST 数据集为例，简单的介绍一套从建立到使用深度学习模型的流程， 以及OneFlow 完成深度学习中所使用的常见 API。通过文章中的链接可以找到关于某类 API 更深入的介绍。

本文分为六大板块：

- 使用已有模型识别图片中的数字
- 加载数据
- 构建网络
- 训练模型
- 保存模型
- 加载模型

其中，第一个板块会简单的展示模型预测效果，而后之后的板块会依次介绍实现模型的五个步骤。

让我们先从导入必要的库开始：

```python
import oneflow as flow
import oneflow.nn as nn
import oneflow.utils.vision.transforms as transforms
```



## 使用模型识别图片中的数字

我们先通过OneFlow已有的模型，体验一下其识别效果。
OneFlow 提供了预训练模型，可以直接用于识别图片中的数字：

识别前我们需要先[下载图片](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/mnist_raw_images.zip)。若不想将整个 dataset 全部下载，则可用以下为一系列从 0 到 9 的图片：

[1](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_22.jpg) [2](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_1.jpg) [3](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_5.jpg) [4](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_15.jpg) [5](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_11.jpg) [6](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_35.jpg) [7](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_12.jpg) [8](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_45.jpg) [9](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_21.jpg) [0](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_7.jpg)

下载图片后，运行以下命令，即可预测制定图片。

```shell
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git
cd cn/docs/code/basics
python3 test_mnist.py <图片文件绝对路径>
```

## 加载数据

OneFlow 主要有两类将数据用作训练的方式：使用 `numpy` 数据或者使用 [Dataloader 与 Dataset](https://url)。

我们在此使用后者，以下为具体实现脚本：

```python
# Dataloader 设置
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
train_iter = flow.utils.data.DataLoader(
    mnist_train, batch_size, shuffle=True
)
test_iter = flow.utils.data.DataLoader(
    mnist_test, batch_size, shuffle=False
)
```


## 构建网络

想要构建网络，只需要实现一个继承自 `nn.Module` 的类就可以了。在它的 `__init__` 方法中定义神经网络的结构，在它的 `forward` 方法中指定数据计算的顺序（正向传播）。

```python
# 设置模型需要的参数
input_size = 784
hidden_size1 = 512
num_classes = 10

# 具体模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, num_classes):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size1)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size1, num_classes)
    def forward(self, x):
        x = self.flatten(x)
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out

device = flow.device("cuda")
model = Net(input_size, hidden_size1, num_classes)
print(model)
model.to(device)
```

打印网络结构：

```shell
Net(
  (l1): Linear(in_features=784, out_features=512, bias=True)
  (relu1): ReLU()
  (l2): Linear(in_features=512, out_features=512, bias=True)
  (relu2): ReLU()
  (l3): Linear(in_features=512, out_features=10, bias=True)
)
```

## 训练模型

为了训练模型，我们需要 `loss` 函数和 `optimizer`，`loss` 函数用于评价神经网络预测的结果与 label 的差距；`optimizer` 调整网络的参数，使得网络预测的结果越来越接近 label（标准答案），其格式为（input tensor，学习率）。这一过程被称为反向传播。

```python
loss = nn.CrossEntropyLoss().to(device)
optimizer = flow.optim.SGD(model.parameters(), lr=0.003)
```

以上展示了一次迭代所需要的正向传播、计算梯度、参数更新。接下来我们将展示如何用 for loop 进行完整的训练。我们一共准备 10 个 epoch，每个 epoch 中迭代 60000 次。

```python
# 训练模型
def train(dataloader, model, loss_fn, optimizer):
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

# 精度测试
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
    return n_correct/n_samples
  
# 具体训练循环
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, n_correct, n_samples = 0.0, 0.0, 0
    train(train_iter, model, loss, optimizer)
    evaluate_accuracy(test_iter, model, device)
     # 验证精度
    test_acc = evaluate_accuracy(test_iter, model, device)

    print("epoch %d, test acc %.3f" % 
        ( epoch + 1, test_acc))
print("Done!")
```

其中，images 代表输入模型的图片，而 labels 代表每张图片的标准答案。例如，假设有一张写有数字 5 的图片，images 就是可以这张图片的 784 个像素，而 labels 就是数字 "5"。

如果你还不熟悉深度学习，可能会好奇，为什么图片可以转变为数字。这是因为计算机本身就是使用数字来表示图片中的像素的，一个 28x28 的灰度图片，表示图片的 784 个像素，其实就是 784 个数字。

输出效果展示：

```shell
epoch 1, test acc 0.098
epoch 2, test acc 0.309
epoch 3, test acc 0.546
epoch 4, test acc 0.653
epoch 5, test acc 0.746
epoch 6, test acc 0.783
epoch 7, test acc 0.809
epoch 8, test acc 0.828
epoch 9, test acc 0.847
epoch 10, test acc 0.861
```

<!--注：打印过程前也许会出现`Parameters in optimizer do not have gradient` 的情况，可忽略。-->

验证精度会在 **加载模型** -> **模型准确率 **中展示。

## 保存模型

```python
flow.save(model.state_dict(), "文件夹路径")
print("Saved OneFlow Model")
```
保存后，OneFlow 会将训练好的模型权重保存到对应的文件夹内。因为模型较大导致文件较多，所以我们推荐为模型单独建立一个文件夹。

## 加载模型

#### 预测效果

```python
test_net = Net(input_size, hidden_size1, hidden_size2, num_classes)
test_net.load_state_dict(flow.load("./mnist_model"))
# 预测
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
test_net.eval()
for images, labels in test_iter:
    pred = test_net(images.reshape((-1, 28*28)))
    x = pred[0].argmax().numpy()
    predicted, actual = classes[x.item(0)], labels[0].numpy()
    print(f'Predicted:"{predicted}", Actual: "{actual}"')
    break
```

这里我们可以通过 model.eval() 来使用模型。

#### 代码完整输出效果展示

```shell
epoch 1, test acc 0.098
epoch 2, test acc 0.309
epoch 3, test acc 0.546
epoch 4, test acc 0.653
epoch 5, test acc 0.746
epoch 6, test acc 0.783
epoch 7, test acc 0.809
epoch 8, test acc 0.828
epoch 9, test acc 0.847
epoch 10, test acc 0.861
Saved OneFlow Model
Predicted:"7", Actual: "7"
```

