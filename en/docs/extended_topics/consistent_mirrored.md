# Consistent 与 Mirrored 视角

在进行分布式训练时，OneFlow 框架提供了两种角度看待数据与模型的关系，被称作 `consistent` 视角与 `mirrored` 视角。

In this article, we will introduce:

* The difference and applicable scenario of the data parallel and model parallel.

* 在分布式任务中采用 `mirrored` 视角及其特点

* 在分布式任务中采用 `consistent` 视角及其特点

## Data parallel and model parallel.
为了更好地理解OneFlow中的 `consistent` 和 `mirrored` 视角，我们需要了解分布式任务中的 **数据并行** 、**模型并行** 两种并行方式的区别。

为了更直观地展示两者的差别，我们先看一个简单的 op (在 OneFlow 中，逻辑上的运算都被抽象为了 operator ，称作 op)：矩阵乘法。

We assume that in training model have a matrix I as input. Multiply matrix I and W then get result O.

![I×W](imgs/i_mul_w.png)

As the description above, size of I is (N, C1), size of W is (C1, C2) and size of O is (N, C2).

Combined  machine learning logic. It can give definitions to the matrixes above:

* Matrix I as the input object, each line is a sample and each column is represents the characteristics of sample.

* Matrix W represents the parameters of model.

* O 是预测结果或者 label ，如果是预测作业，那么就是由 I、W 求解 O，得到分类结果的过程；如果是训练作业，那么就是由 I 与 O 求解 W 的过程

当以上 I 矩阵的行 N 很大，说明样本很多；如果 W 矩阵的列 C2 很大，说明模型复杂；当样本数目、模型复杂程度复杂到一定程度时，单机单卡的硬件条件已经无法承载训练作业，就需要考虑分布式的方式训练。In distributed training system, we can choose ** data parallel **and **model parallel**.

<a id="mat_mul_op"></a>
In order to better understand data parallel and model parallel, we use the following figure as the demo of matrix multiplication operator:

![mat_mul_op](imgs/mul_op_illustrated.png)

The first matrixe in grey on left of equation is the input sample. Each line is a sample. The second matrixe in blue on left of equation is the model.

In this article, we will see the operators above switching to different way under data parallel and model parallel.


### Data parallel diagram

In **data parallel**, divide the sample data in small parts. **Data after dividing **will send to each training nodes and calculate with the **completely models**. Finally combined the information in each nodes. Like figure show below:

![mat_mul_op](imgs/mul_op_data_parr.png)

### Model parallel diagram

In **model parallel**, model will be divided. **Completely data** will send to each nodes and calculate with **model after dividing**. Finally combined the model in each nodes. Like figure show below:

![mat_mul_op](imgs/mul_op_model_parr.png)

Basically:

* In data parallel, each node use the same model to train, data will be cut.

* In model parallel, each node received same data, model will be cut.

接下来我们将介绍 OneFlow 看待分布式系统的两种视角（`mirrored` 视角与 `consistent` 视角），学习在不同的视角下如何选择并行方式。

### Two type of place holder
在[使用OneFlow搭建神经网络](../basics_topics/build_nn_with_op_and_layer.md)及[定义与调用作业函数](./job_function_define_call.md)中已经介绍了 **数据占位符** 与 **Blob** 的概念。

实际上，针对并行，OneFlow的数据占位符还可以细分为两类：分别通过接口 `oneflow.typing.Numpy.Placeholder` 和 `oneflow.typing.ListNumpy.Placeholder` 构造的占位符，分别对应 `Consistent`  与 `Mirrored`情况。

We will see the detailed examples below.


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

### Example
在以下的代码中，我们使用采用默认的 `mirrored_view` 视角，使用2个 GPU 进行训练。

Name: [mirrored_strategy.py](../code/extended_topics/mirrored_strategy.py)

The key part of the description in "script explanation" section.

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
def train_job(images:tp.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU, 1, 28, 28), dtype=flow.float),
              labels:tp.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU,), dtype=flow.int32)) -> tp.ListNumpy:
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="dense1")
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == '__main__':
    flow.config.gpu_device_num(2)  # 设置GPU数目
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)

    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        images1 = images[:BATCH_SIZE_PER_GPU]
        images2 = images[BATCH_SIZE_PER_GPU:]
        labels1 = labels[:BATCH_SIZE_PER_GPU]
        labels2 = labels[BATCH_SIZE_PER_GPU:]

        imgs_list = [images1, images2]
        labels_list = [labels1, labels2]

        loss = train_job(imgs_list, labels_list)
        total_loss = np.array([*loss[0], *loss[1]])
        if i % 20 == 0: print(total_loss.mean())
```

### Script explanation
In the above script:

* Use  `flow.config.gpu_device_num` to set GPU amount as two.
```python
flow.config.gpu_device_num(2)
```

* `oneflow.typing.ListNumpy.Placeholder` defined the sample amount which is the amount after dividing. And the relationship between `BATCH_SIZE_PER_GPU`  and `BATCH_SIZE` is `BATCH_SIZE=BATCH_SIZE_PER_GPU×GPU_NUM`.
```python
def train_job(images:tp.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU, 1, 28, 28), dtype=flow.float),
              labels:tp.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU,), dtype=flow.int32)) -> tp.ListNumpy:
```

* The data after dividing need to store in the  `list` and pass to training functions. The number of elements in `list` need be same as the **GPU number in training**. OneFlow will pass the data according to the order of the elements in `list ` to each GPU(the number i element in `list` is corresponding to number i GPU):
```python
  images1 = images[:BATCH_SIZE_PER_GPU]
  images2 = images[BATCH_SIZE_PER_GPU:]
  labels1 = labels[:BATCH_SIZE_PER_GPU]
  labels2 = labels[BATCH_SIZE_PER_GPU:]

  imgs_list = [images1, images2]
  labels_list = [labels1, labels2]

  loss = train_job(imgs_list, labels_list)
```

* 返回的得到的结果 `loss`，是一个 `list`，该 `list` 中元素个数与 **参与训练的GPU数目** 一致；`list` 中的第i个元素对应了第 i 张 GPU 卡上的运算结果。Then we do the combination then print the  `total_loss`
```python
  total_loss = np.array([*loss[0], *loss[1]])
  if i % 20 == 0: print(total_loss.mean())
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

### Example
以下代码，我们采用 consistent 视角，使用2个 GPU 进行训练，consistent 策略下默认的并行方式仍然是 **数据并行**。The issue of how to set **model parallel** and **mix parallel** in consistent strategy will not be discussed in this article. We have special introduction of that in [parallels characters of OneFlow](model_mixed_parallel.md).

Name: [consistent_strategy.py](../code/extended_topics/consistent_strategy.py)

```python
import numpy as np
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer, name="conv1")
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name="pool1")
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer, name="conv2")
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name="pool2")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="outlayer")


@flow.global_function(type="train")
def train_job(images:tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Numpy:
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == '__main__':
    flow.config.gpu_device_num(2)
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)

    for epoch in range(50):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0: print(loss.mean())
```

### Script explanation
In above script:

* Use  `flow.config.gpu_device_num` to define the GPU number:
```python
flow.config.gpu_device_num(2)
```

* 使用 `tp.Numpy.Placeholder` 定义 consistent 视角下的占位符，因为`Numpy.Placeholder`产出的 Blob 代表逻辑上的 op 及数据占位符，因此此处的 BATCH_SIZE 就是整个分布式训练的样本总和，不需要人为切分或者组合
```python
@flow.global_function(type="train")
def train_job(images:tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Numpy:
```

* 调用作业函数，直接得到训练结果，训练结果已经由 OneFlow 完成分布式过程中切分与合并的工作。在 consistent 视角下，多卡的分布式训练与单卡的训练，代码差别极少，上手体验几乎一样
```python
  for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels)
      if i % 20 == 0: print(loss.mean())
```

## More extending
随着机器学习理论与实践发展，现在已经出现了很多单机无法训练的网络；也出现了越来越多仅采用数据并行无法很好完成训练的模型。

采用OneFlow的 `consistent` 视角，通过自由选择及组合并行方式，可以很好地解决以上问题，我们在[OneFlow的并行特色](model_mixed_parallel.md)进行了专门的介绍。


