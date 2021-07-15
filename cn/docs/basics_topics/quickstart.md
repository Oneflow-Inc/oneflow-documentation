# 快速上手

本文将以 LeNet-5 网络训练 MNIST 数据集为例。介绍 OneFlow 完成深度学习中所使用的常见 API，通过文章中的链接可以找到关于某类 API 更深入的介绍。

## 使用 LeNet-5 识别图片中的数字

OneFlow 提供了 LeNet-5 的预训练模型，可以直接用于识别图片中的数字：

```python
>>> import oneflow as flow
>>> model = flow.LeNet()
>>> num = model.run(flow.load_image("xxx.jpg"))
>>> num
5
```

你可以将以上 `xxx.jpg` 替换为其它图片的路径【支持网页图片效果更好】，看看识别效果。

【链接一个视频，展示预测效果】

## 加载数据

OneFlow 主要有两类将数据用作训练的方式：使用 `numpy` 数据或者使用 [Dataloader 与 Dataset](https://url)。

我们在此使用前者，直接加载图并且将它转为 numpy 数据：

```python
加载图片
显示 numpy 数据
```

如果你还不熟悉深度学习，可能会好奇，为什么图片可以转变为数字，这是因为计算机本身就是使用数字来表示图片中的像素的，一个 28x28 的灰度图片，表示图片的 784 个像素，其实就是 784 个数字。

【gif 动图，图片变数字】


## 构建网络

想要构建网络，只需要实现一个继承自 `nn.Module` 的类就可以了，在它的 `__init__` 方法中定义神经网络的结构，在它的 `forward` 方法中指定数据计算的顺序。

```python
【伪代码，待对齐LeNet5】
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
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

model = LeNet5()
print(model)
```

## 训练模型

为了训练模型，我们需要 `loss` 函数和 `optimizer`，`loss` 函数用于评价神经网络预测的结果与 label 的差距；`optimizer` 调整网络的参数，使得网络预测的结果越来越接近 label（标准答案）。

```python
>>> loss_fn = nn.CrossEntropyLoss()
>>> optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

```python
前向、反向代码 + 注释
```

以上展示了一次迭代所需要的正向传播、计算梯度、参数更新。完整的训练，我们一共准备5个 epoch，每个 epoch 中迭代60000次。

```python
完整训练代码
```

【输出效果】

## 保存模型

## 加载模型
