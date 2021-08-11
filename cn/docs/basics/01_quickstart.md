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

## 使用模型识别图片中的数字

我们先通过OneFlow已有的模型，体验一下其识别效果。
OneFlow 提供了预训练模型，可以直接用于识别图片中的数字：

识别前我们需要先[下载图片](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/mnist_raw_images.zip)。若不想将整个 dataset 全部下载，则可用以下为一系列从 0 到 9 的图片：

[1](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_22.jpg) [2](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_1.jpg) [3](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_5.jpg) [4](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_15.jpg) [5](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_11.jpg) [6](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_35.jpg) [7](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_12.jpg) [8](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_45.jpg) [9](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_21.jpg) [0](https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-documentation/mnist_test/img_7.jpg)

```python
>>> import oneflow as flow
>>> model = flow.Net()
>>> num = model.run(flow.load_image("图片路径 + 文件名"))
>>> num
```

## 加载数据

OneFlow 主要有两类将数据用作训练的方式：使用 `numpy` 数据或者使用 [Dataloader 与 Dataset](https://url)。

我们在此使用前者，直接加载图并且将它转为 numpy 数据：

```python
# 下载并设置数据
BATCH_SIZE = 100
(train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE, BATCH_SIZE) #设置训练模型以及预测所需的参数(np.float)

# 将参数转换为flow.tensor的格式
tr_images = flow.tensor(train_images)
tr_labels = flow.tensor(train_labels)
te_images = flow.tensor(test_images)
te_labels = flow.tensor(test_labels)
```

其中，images 代表输入模型的图片，而 labels 代表每张图片的标准答案。例如，假设有一张写有数字 5 的图片，images 就是可以这张图片的 784 个像素，而 labels 就是数字 “5”。

而 train(tr) 和 test(te) 的区别就在于前者负责训练模型，后者则是负责测试模型预测准确率。

如果你还不熟悉深度学习，可能会好奇，为什么图片可以转变为数字。这是因为计算机本身就是使用数字来表示图片中的像素的，一个 28x28 的灰度图片，表示图片的 784 个像素，其实就是 784 个数字。

【gif 动图，图片变数字】


## 构建网络

想要构建网络，只需要实现一个继承自 `nn.Module` 的类就可以了，在它的 `__init__` 方法中定义神经网络的结构，在它的 `forward` 方法中指定数据计算的顺序（正向传播）。

```python
# 设置模型需要的参数
input_size = 784
hidden_size1 = 128
hidden_size2 = 64
num_classes = 10
batch_size = 100
n_total_steps = 600

# 具体模型
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

model = Net()
print(model)
```

打印网络结构：

```
Net(
  (l1): Linear(in_features=784, out_features=128, bias=True)
  (relu1): ReLU()
  (l2): Linear(in_features=128, out_features=64, bias=True)
  (relu2): ReLU()
  (l3): Linear(in_features=64, out_features=10, bias=True)
)
```

## 训练模型

为了训练模型，我们需要 `loss` 函数和 `optimizer`，`loss` 函数用于评价神经网络预测的结果与 label 的差距；`optimizer` 调整网络的参数，使得网络预测的结果越来越接近 label（标准答案），其格式为（input tensor，学习率）。这一过程被称为反向传播。

```python
>>> loss_fn = nn.CrossEntropyLoss()
>>> optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

以上展示了一次迭代所需要的正向传播、计算梯度、参数更新。接下来我们将展示如何用 for loop 进行完整的训练。我们一共准备 15 个 epoch，每个 epoch 中迭代 60000 次。

```python
num_epochs = 15
n_total_steps = 60000
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):

        #调整参数格式
        T_images = flow.tensor(images, dtype=flow.float32)
        T_labels = flow.tensor(labels, dtype=flow.long)
        
        #矩阵格式对齐
        T_images = flow.reshape(T_images, shape=[-1, 28*28])
        T_labels = T_labels

        #正向传播和loss
        outputs = model(T_images)
        loss = loss_fn(outputs, T_labels)

        #反向传播和更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 600 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.mean()}')
```

输出效果展示：

```
poch [1/15], Step [1/600], Loss: tensor([2.3062], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [2/15], Step [1/600], Loss: tensor([2.2592], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [3/15], Step [1/600], Loss: tensor([2.1636], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [4/15], Step [1/600], Loss: tensor([1.9094], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [5/15], Step [1/600], Loss: tensor([1.3981], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [6/15], Step [1/600], Loss: tensor([0.9669], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [7/15], Step [1/600], Loss: tensor([0.7557], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [8/15], Step [1/600], Loss: tensor([0.6434], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [9/15], Step [1/600], Loss: tensor([0.5711], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [10/15], Step [1/600], Loss: tensor([0.519], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [11/15], Step [1/600], Loss: tensor([0.4794], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [12/15], Step [1/600], Loss: tensor([0.4482], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [13/15], Step [1/600], Loss: tensor([0.4229], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [14/15], Step [1/600], Loss: tensor([0.402], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [15/15], Step [1/600], Loss: tensor([0.3842], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
```

<!--注：打印过程前也许会出现`Parameters in optimizer do not have gradient` 的情况，可忽略。-->

## 保存模型

```python
flow.save(model.state_dict(), "文件夹路径")
print("Saved OneFlow Model")
```
保存后，OneFlow 会将训练好的模型权重保存到对应的文件夹内。因为模型较大导致文件较多，所以我们推荐为模型单独建立一个文件夹。

## 加载模型

#### 模型准确率

```python
with flow.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in zip(test_images, test_labels):
        T_images = flow.tensor(images, dtype=flow.float32)
        T_labels = flow.tensor(labels, dtype=flow.float32)
        T_images = flow.reshape(T_images, shape=[-1, 28*28])
        
        T_labels = T_labels
        outputs = model(T_images)
        
        predictions = flow.argmax(outputs, dim=1)
        n_samples +=labels.shape[0]
        T_correct = flow.eq(predictions, T_labels)
        x = flow.sum(T_correct).numpy()
        n_correct += x[0]
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}%')
```

在进行非训练操作时，要用 `with flow.no_grad()` 将操作包裹住，以达到不改变模型权重的效果。

#### 预测效果

```python
# 预测
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
model.eval()
te_images = flow.reshape(te_images, shape=[-1, 28*28])
x, y = te_images, test_labels[0]
with flow.no_grad():
    pred = model(x)
    x = pred[0].argmax().numpy()
    predicted, actual = classes[x.item(0)], y[0]
    print(f'Predicted:"{predicted}", Actual: "{actual}"')
```

这里我们可以通过 model.eval() 来使用模型。

#### 代码完整输出效果展示

```
poch [1/15], Step [1/600], Loss: tensor([2.3062], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [2/15], Step [1/600], Loss: tensor([2.2592], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [3/15], Step [1/600], Loss: tensor([2.1636], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [4/15], Step [1/600], Loss: tensor([1.9094], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [5/15], Step [1/600], Loss: tensor([1.3981], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [6/15], Step [1/600], Loss: tensor([0.9669], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [7/15], Step [1/600], Loss: tensor([0.7557], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [8/15], Step [1/600], Loss: tensor([0.6434], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [9/15], Step [1/600], Loss: tensor([0.5711], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [10/15], Step [1/600], Loss: tensor([0.519], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [11/15], Step [1/600], Loss: tensor([0.4794], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [12/15], Step [1/600], Loss: tensor([0.4482], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [13/15], Step [1/600], Loss: tensor([0.4229], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [14/15], Step [1/600], Loss: tensor([0.402], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Epoch [15/15], Step [1/600], Loss: tensor([0.3842], dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
Finished Training
accuracy = 89.38%
Predicted:"7", Actual: "7"
```

