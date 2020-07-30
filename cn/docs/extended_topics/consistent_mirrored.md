# Consistent 与 Mirrored 视角

在进行分布式训练时，OneFlow 框架提供了两种角度看待数据与模型的关系，被称作 `consistent` 视角与 `mirrored` 视角。

本文将介绍：

* 数据并行与模型并行的区别及适用场景

* 在分布式任务中采用 `mirrored` 视角及其特点

* 在分布式任务中采用 `consistent` 视角及其特点

## 数据并行与模型并行
为了更好地理解OneFlow中的 `consistent` 和 `mirrored` 视角，我们需要了解分布式任务中的 **数据并行** 、**模型并行** 两种并行方式的区别。

为了更直观地展示两者的差别，我们先看一个简单的 op (在 OneFlow 中，逻辑上的运算都被抽象为了 operator ，称作 op)：矩阵乘法。

我们假定在模型训练中，存在一个输入矩阵 I ，通过矩阵 I 与矩阵 W 做矩阵乘法，得到输出矩阵 O 。

![I×W](imgs/i_mul_w.png)

如以上所示，I的大小为(N, C1)，W的大小为(C1, C2)，O的大小为(N, C2)。

结合机器学习的业务逻辑，可以赋予以上几个矩阵直观意义：

* I 矩阵作为输入矩阵，每一行都是一个样本，一行中的各列代表了样本的特征

* W 矩阵代表了模型参数

* O 是预测结果或者 label ，如果是预测作业，那么就是由 I、W 求解 O，得到分类结果的过程；如果是训练作业，那么就是由 I 与 O 求解 W 的过程

当以上 I 矩阵的行 N 很大，说明样本很多；如果 W 矩阵的列 C2 很大，说明模型复杂；当样本数目、模型复杂程度复杂到一定程度时，单机单卡的硬件条件已经无法承载训练作业，就需要考虑分布式的方式训练。而在分布式系统中，我们可以选择 **数据并行** 和 **模型并行**。

<a id="mat_mul_op"></a>
为了便于理解数据并行与模型并行，我们先用下图作为矩阵相乘 op 的示例：

![mat_mul_op](imgs/mul_op_illustrated.png)

等式左边第1个灰色的矩阵代表输入样本，每一行是一个样本；等式左边第2个蓝色的矩阵代表模型。

在后文中，我们将看到以上的 op，在数据并行与模型并行下，不同的“切分”方式。


### 数据并行图示

在 **数据并行** 中，将样本数据进行切分，**切分后的数据** 被送至各个训练节点，与 **完整的模型** 进行运算，最后将多个节点的信息进行合并，如下图所示：

![mat_mul_op](imgs/mul_op_data_parr.png)

### 模型并行图示

在 **模型并行** 中，将模型进行切分，**完整的数据** 被送至各个训练节点，与 **切分后的模型** 进行运算，最后将多个节点的运算结果合并，如下图所示：

![mat_mul_op](imgs/mul_op_model_parr.png)

总之：

* 数据并行下，各个训练节点的模型是完全一样的，数据被切分；

* 模型并行下，各个训练节点都接收一样的完整数据， 模型被切分。

接下来我们将介绍 OneFlow 看待分布式系统的两种视角（`mirrored` 视角与 `consistent` 视角），学习在不同的视角下如何选择并行方式。

### 两类占位符
在[使用OneFlow搭建神经网络](../basics_topics/build_nn_with_op_and_layer.md)及[定义与调用作业函数](./job_function_define_call.md)中已经介绍了 **数据占位符** 与 **Blob** 的概念。

实际上，针对并行，OneFlow的数据占位符还可以细分为两类：分别通过接口 `oneflow.typing.Numpy.Placeholder` 和 `oneflow.typing.ListNumpy.Placeholder` 构造的占位符，分别对应 `Consistent`  与 `Mirrored`情况。

我们将在下文中看到它们的具体应用。


## 在 OneFlow 中使用 mirrored 视角

其它的框架，如 TensorFlow、Pytorch 均支持 `mirroed strategy`；OneFlow 的 mirrored 视角与它们类似。

在 mirrored 视角下，模型被镜像复制到每张卡上，每个节点的模型构图是完全相同的，只能采用 **数据并行** 。

在 OneFlow 中，默认不是 mirrored 策略，需要通过 `flow.function_config()` 的 `default_logical_view` 接口来显式指定：

```python
    func_config = flow.function_config()
    func_config.default_logical_view(flow.scope.mirrored_view())
```

在 `mirrored_view` 下，只能采用 **数据并行** 的并行模式，在调用作业函数时，我们需要将数据按照训练节点的数目（显卡总数）进行平均切分，并将切分后的数据放入 `list` 中进行传递，`list` 中的每个元素，就是后分配给 **各个显卡** 的实际数据。

训练函数的返回值类型，也变作了 `oneflow.typing.ListNumpy`，是一个 `list`， `list` 中的每个元素，对应了每张卡上训练结果。

以上提及的 `list` 中的所有元素 **拼接在一起** ，才是一个完整的 BATCH。

### 代码示例
在以下的代码中，我们使用采用默认的 `mirrored_view` 视角，使用2个 GPU 进行训练。

完整代码：[mirrored_strategy.py](../code/extended_topics/mirrored_strategy.py)

重点部分的说明请见后文“代码解析”部分。

```python
import numpy as np
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100
GPU_NUM = 2
BATCH_SIZE_PER_GPU = int(BATCH_SIZE / GPU_NUM)


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.default_logical_view(flow.scope.mirrored_view())
    return config


@flow.global_function(type="train", function_config=get_train_config())
def train_job(
    images: tp.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU, 1, 28, 28), dtype=flow.float),
    labels: tp.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU,), dtype=flow.int32),
) -> tp.ListNumpy:
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
    )
    logits = flow.layers.dense(
        hidden, 10, kernel_initializer=initializer, name="dense2"
    )
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == "__main__":
    flow.config.gpu_device_num(2)  # 设置GPU数目
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )

    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        images1 = images[:BATCH_SIZE_PER_GPU]
        images2 = images[BATCH_SIZE_PER_GPU:]
        labels1 = labels[:BATCH_SIZE_PER_GPU]
        labels2 = labels[BATCH_SIZE_PER_GPU:]

        imgs_list = [images1, images2]
        labels_list = [labels1, labels2]

        loss = train_job(imgs_list, labels_list)
        total_loss = np.array([*loss[0], *loss[1]])
        if i % 20 == 0:
            print(total_loss.mean())
```

### 代码解析
以上代码中：

* 使用 `flow.config.gpu_device_num` 设置 GPU 数目为2 
```python
flow.config.gpu_device_num(2)
```

* `oneflow.typing.ListNumpy.Placeholder` 定义的样本数目，是被切分后的数目，即代码中的 `BATCH_SIZE_PER_GPU` 与总样本数 `BATCH_SIZE` 的关系为：`BATCH_SIZE=BATCH_SIZE_PER_GPU×GPU_NUM`
```python
def train_job(
    images: tp.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU, 1, 28, 28), dtype=flow.float),
    labels: tp.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU,), dtype=flow.int32),
) -> tp.ListNumpy:
```

* 切分后的数据，需要保存至 `list` 中传入训练函数；`list` 中元素的个数与 **参与训练的GPU数目** 一致；OneFlow 将按照 `list` 中元素顺序，向各卡传递数据( `list` 中第 i 个元素对应第 i 张卡)：
```python
  images1 = images[:BATCH_SIZE_PER_GPU]
  images2 = images[BATCH_SIZE_PER_GPU:]
  labels1 = labels[:BATCH_SIZE_PER_GPU]
  labels2 = labels[BATCH_SIZE_PER_GPU:]

  imgs_list = [images1, images2]
  labels_list = [labels1, labels2]

  loss = train_job(imgs_list, labels_list)
```

* 返回的得到的结果 `loss`，是一个 `list`，该 `list` 中元素个数与 **参与训练的GPU数目** 一致；`list` 中的第i个元素对应了第 i 张 GPU 卡上的运算结果。我们做了拼接后，计算并打印了 `total_loss`
```python
  total_loss = np.array([*loss[0], *loss[1]])
  if i % 20 == 0:
      print(total_loss.mean())
```

## 在 OneFlow 中使用 consistent 视角
我们已经了解了 mirrored 视角，知道在 `mirrored_view` 视角下，样本会被平均分配到多个完全一样的模型上进行分布式训练，各个训练节点上的结果，需要组装才能得到真正完整的 BATCH，对应了逻辑上的 op 与 Blob。

除了 mirroed 视角外，OneFlow 还提供了 consistent 视角。 consistent 视角是 OneFlow 的一大特色，与 mirrored 视角相比有很大的优势。 

默认情况下 OneFlow 采取的是 consistent 视角，如果想显式声明，也可以通过代码设置：
```python
  config = flow.function_config()
  config.default_logical_view(flow.scope.consistent_view())
```

之所以说 consistent 视角是 OneFlow 的一大特色，是因为在 OneFlow 的设计中，若采用 `consistent_view`，那么从用户的视角看，所使用的op、blob将获得 **逻辑上的统一**，同样以本文开头的矩阵乘法为例，我们只需要关注[矩阵乘法](#mat_mul_op)本身数学计算上的意义；而在工程上到底如何配置、采用模型并行还是数据并行等细节问题，可以使用 OneFlow 的接口轻松完成。OneFlow 内部会高效可靠地解决 **数据并行中的数据切分** 、**模型并行中的模型切分** 、**串行逻辑** 等问题。

 在 OneFlow 的 consistent 视角下，可以自由选择模型并行、数据并行、流水并行或者混合并行。 

### 代码示例
以下代码，我们采用 consistent 视角，使用2个 GPU 进行训练，consistent 策略下默认的并行方式仍然是 **数据并行**。关于如何在consistent 策略下设置 **模型并行** 及 **混合并行** 不在本文讨论范围，我们在[OneFlow的并行特色](model_mixed_parallel.md)中有专门的介绍与示例。

完整代码：[consistent_strategy.py](../code/extended_topics/consistent_strategy.py)

```python
import numpy as np
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="conv1",
    )
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding="SAME", name="pool1")
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="conv2",
    )
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding="SAME", name="pool2")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="hidden",
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(
        hidden, 10, kernel_initializer=initializer, name="outlayer"
    )


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == "__main__":
    flow.config.gpu_device_num(2)
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )

    for epoch in range(50):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
```

### 代码解析
以上代码中：

* 使用 `flow.config.gpu_device_num` 设置GPU数目：
```python
flow.config.gpu_device_num(2)
```

* 使用 `tp.Numpy.Placeholder` 定义 consistent 视角下的占位符，因为`Numpy.Placeholder`产出的 Blob 代表逻辑上的 op 及数据占位符，因此此处的 BATCH_SIZE 就是整个分布式训练的样本总和，不需要人为切分或者组合
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
```

* 调用作业函数，直接得到训练结果，训练结果已经由 OneFlow 完成分布式过程中切分与合并的工作。在 consistent 视角下，多卡的分布式训练与单卡的训练，代码差别极少，上手体验几乎一样
```python
for i, (images, labels) in enumerate(zip(train_images, train_labels)):
  loss = train_job(images, labels)
  if i % 20 == 0:
      print(loss.mean())
```

## 更多扩展
随着机器学习理论与实践发展，现在已经出现了很多单机无法训练的网络；也出现了越来越多仅采用数据并行无法很好完成训练的模型。

采用OneFlow的 `consistent` 视角，通过自由选择及组合并行方式，可以很好地解决以上问题，我们在[OneFlow的并行特色](model_mixed_parallel.md)进行了专门的介绍。


