在这篇文章中，我们将学习：

- 使用 OneFlow 接口配置软硬件环境
- 使用 OneFlow 的接口定义模型
- 使用 `train` 类型作业函数做模型训练
- 模型的保存和加载
- 使用 `predict` 类型作业函数做模型校验
- 使用 `predict` 类型作业函数做图像识别

本文通过使用 LeNet 模型，训练 MNIST 数据集向大家介绍使用 OneFlow 的各个核心环节，文末附有完整示例代码的链接。

在学习之前，也可以通过以下命令查看各脚本功能（**脚本运行依赖 默认选择机器上的0号GPU，如果你安装的是CPU版本OneFlow，则脚本会自动调用CPU来做训练。**）。

首先，同步本文档仓库并切换到对应路径：
```shell
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git
cd oneflow-documentation/cn/docs/code/quick_start/
```

**模型训练**
```shell
python lenet_train.py
```
以上命令将对 MNIST 数据集进行训练，并保存模型。

输出：

```she
File mnist.npz already exist, path: ./mnist.npz
5.9947124
1.0865117
0.5317516
0.20937675
0.26428983
0.21764673
0.23443426
...
```

> 以下的两个脚本 `lenet_eval.py` 与 `lenet_test.py` 都依赖以上训练的结果，因此需要先运行以上脚本。或者你可以直接下载我们已经训练好的模型，则可以略过以上步骤，下载方法如下：

```shell
#在仓库docs/code/quick_start/目录下
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip
unzip lenet_models_1.zip
```

**模型校验**
```shell
python lenet_eval.py
```
以上命令，使用 MNIST 测试集对刚刚生成的模型进行校验，并给出准确率。

输出：

```text
File mnist.npz already exist, path: ./mnist.npz
accuracy: 99.4%
```

**图像识别**

```shell
python lenet_test.py ./9.png
# 输出：prediction: 9
```
以上命令将使用之前训练的模型对我们准备好的 `9.png` 图片文件中的内容进行预测。
你也可以下载我们[提取好的 mnist 图片](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)，自行对更多图片文件的预测效果进行验证。

## MNIST 数据集介绍

MNIST 是一个手写数字的数据库。包括了训练集与测试集；训练集包含了60000张图片以及图片对应的标签，测试集包含了10000张图片以及图片测试的标签。Yann LeCun 等已经将图片进行了大小归一化及居中处理，并且打包为二进制文件供下载([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))。本文涉及的脚本会自动下载 MNIST 数据集。

## 定义训练模型

在 [oneflow.nn](https://oneflow.readthedocs.io/en/master/nn.html) 及 [oneflow.layers](https://oneflow.readthedocs.io/en/master/layers.html) 模块提供了常见的用于构建模型的算子。

```python
def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv1",
        kernel_initializer=initializer,
    )
    pool1 = flow.nn.max_pool2d(
        conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
    )
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(
        conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
    )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")
```

以上代码中，我们搭建了一个 LeNet 网络模型。

## 实现训练作业函数

OneFlow 中提供了 [oneflow.global_function](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.global_function) 装饰器，通过它，可以将一个 Python 函数转变为作业函数（job function）。

### global_function 装饰器

`oneflow.global_function` 装饰器需要两个参数：`type` 与 `function_config`。`type`用于指定作业函数的类型，`type="train"` 意味着作业函数用于训练，`type="predict"` 意味着作业函数用于预测。`function_config` 参数为一个 [oneflow.function_config](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.python.framework.function_util.FunctionConfig#oneflow.FunctionConfig) 对象，可用它配置作业函数的细节。

以下代码片段展示，我们定义了一个 `train` 类型的作业函数，因为没有设置 `function_config`，所以作业函数的其它配置为默认配置。

```python
@flow.global_function(type="train")
def train_job(images:tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Numpy:
    #作业函数实现 ...
```

其中的 `tp.Numpy.Placeholder` 是数据占位符， `-> tp.Numpy` 指定这个作业函数在调用时，将返回一个 `numpy` 对象。

### 指定优化目标
我们可以通过 [oneflow.optimizer](https://oneflow.readthedocs.io/en/master/optimizer.html) 下的接口指定优化器及其优化目标。这样，OneFlow 在每次迭代训练作业的过程中，将以指定的方式优化目标。

```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```

以上，我们通过 `flow.nn.sparse_softmax_cross_entropy_with_logits` 求得 loss ，并且将 loss 作为优化目标。

 - **lr_scheduler** 设定了学习率计划，[0.1]表明初始学习率为0.1；
 - **flow.optimizer.SGD** 则指定了优化器为 SGD；loss 作为参数传递给 minimize 表明优化器将以最小化 loss 为目标。

 更多 `optimizer` 及其使用方法可以参见 [oneflow.optimizer](https://oneflow.readthedocs.io/en/master/optimizer.html)

## 调用作业函数并交互

调用作业函数就可以开始训练。

调用作业函数的返回结果，由定义作业函数时指定的返回值类型决定，可以返回一个，也可以返回多个结果。

### 返回一个结果的例子
在 [lenet_train.py](../code/quick_start/lenet_train.py) 中定义的作业函数：
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```
该作业函数的返回值类型为 `tp.Numpy`，则当调用时，会返回一个 `numpy` 对象：
```python
for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
```

我们调用了 `train_job` 并每循环20次打印1次 `loss`。

### 返回多个结果的例子
在校验模型的代码 [lenet_eval.py](../code/quick_start/lenet_eval.py) 中定义的作业函数：
```python
@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> Tuple[tp.Numpy, tp.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)
```

该作业函数的返回值类型为 `Tuple[tp.Numpy, tp.Numpy]`，则当调用时，会返回一个 `tuple` 容器，里面有2个元素，每个元素都是一个 `numpy` 对象：
```python
for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            labels, logits = eval_job(images, labels)
            acc(labels, logits)
```
我们调用作业函数返回了 `labels` 与 `logits`，并用它们评估模型准确率。

### 同步与异步调用
本文所有代码都是同步方式调用作业函数，实际上 OneFlow 还支持异步方式调用作业函数，具体可参考[获取作业函数的结果](../basics_topics/async_get.md)一文。

## 模型的初始化、保存与加载

### 模型的初始化与保存

通过 `flow.checkpoint.save` 方法保存模型。如下例：

```python
if __name__ == '__main__':
  #加载数据及训练 ...
  flow.checkpoint.save("./lenet_models_1")
```

保存成功后，我们将得到名为 `lenet_models_1` 的 **目录** ，该目录中包含了与模型参数对应的子目录及文件。

### 模型的加载

在预测过程中，我们可以通过 `flow.checkpoint.get` 从文件中加载参数值到内存，再通过 `flow.load_variables` 将参数值更新到模型上。如下例：

```python
if __name__ == '__main__':
  flow.load_variables(flow.checkpoint.get("./lenet_models_1"))
  #校验过程 ...
```

## 模型的校验
用于校验的 `predict` 类型的作业函数与 `train` 类型的作业函数 **几乎没有区别** ，不同之处在于校验过程中的模型参数来自于已经保存好的模型，因此不需要初始化，不需要更新模型参数（所以也不用指定 `optimizer`）。

### 用于校验的作业函数的编写

```python
@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> Tuple[tp.Numpy, tp.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)
```

以上是用于校验的作业函数的实现，声明了返回值类型是 `Tuple[tp.Numpy, tp.Numpy]`， 因此返回一个 `tuple`， `tuple` 中有2个元素，每个元素都是1个 `numpy` 对象。我们将调用训练作业函数，并根据返回结果计算准确率。

### 迭代校验
以下 `acc` 函数中统计样本的总数目，以及校验正确的总数目，我们将调用作业函数，得到 `labels` 与 `logits`：
```python
g_total = 0
g_correct = 0


def acc(labels, logits):
    global g_total
    global g_correct

    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count

```

调用校验作业函数：

```python
if __name__ == "__main__":
    flow.load_variables(flow.checkpoint.get("./lenet_models_1"))
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )

    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            labels, logits = eval_job(images, labels)
            acc(labels, logits)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
```

以上，循环调用校验函数，并且最终输出在 MNIST 测试集上的判断准确率。

## 预测图片

将以上校验代码修改，使得校验数据来自于原始的图片而不是现成的数据集，我们就可以使用模型进行图片内容预测。

```python
def load_image(file):
    im = Image.open(file).convert("L")
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im - 128.0) / 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im


def main():
    if len(sys.argv) != 2:
        usage()
        return
    flow.load_variables(flow.checkpoint.get("./lenet_models_1"))

    image = load_image(sys.argv[1])
    logits = eval_job(image, np.zeros((1,)).astype(np.int32))

    prediction = np.argmax(logits, 1)
    print("prediction: {}".format(prediction[0]))


if __name__ == "__main__":
    main()
```

## 完整代码

### 训练模型

代码：[lenet_train.py](../code/quick_start/lenet_train.py)

### 校验模型

代码：[lenet_eval.py](../code/quick_start/lenet_eval.py)

预训练模型：[lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)


### 数字预测

代码：[lenet_test.py](../code/quick_start/lenet_test.py)

预训练模型：[lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

MNIST 数据集图片：[mnist_raw_images.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)
