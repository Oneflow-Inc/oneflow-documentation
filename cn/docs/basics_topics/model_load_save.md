# 模型的加载与保存

对于模型的加载与保存，常用的场景有：

* 将已经训练一段时间的模型保存，方便下次继续训练

* 将训练好的模型保存，方便后续直接部署使用

严格来说，尚未训练好的模型的保存，称为保存检查点 `checkpoint` 或者快照 `snapshot` 。与将已经训练好的模型保存 `model saving` ，在概念上，略有不同。

不过，无论模型是否训练完毕，我们都可以使用 **统一的接口** 将其保存，因此，我们在其它框架中看到的`model`、`checkpoint`、`snapshot`，在 OneFlow 的操作中不做区分。它们在 OneFlow 中，都通过`flow.train.CheckPoint`类作为接口操作。

本文将介绍：

* 如何创建模型参数

* 如何保存/加载模型

* OneFlow 模型的存储结构

* 如何微调与扩展模型

## 使用 get_variable 创建/获取模型参数对象

我们可以使用 `oneflow.get_variable` 方法创造或者获取一个对象，该对象可以用于在全局作业函数中交互信息；当调用 `oneflow.CheckPoint` 的对应接口时，该对象也会被自动地保存或从存储设备中恢复。

因为这个特点，`get_variable` 创建的对象，常用于存储模型参数。实际上，OneFlow 中很多较高层接口（如 `oneflow.layers.conv2d`），内部使用 `get_variable` 创建模型参数。

### get_variable 创建/获取对象的流程

`get_variable` 需要一个指定一个 `name` 参数，该参数作为创建对象的标识。

如果 `name` 指定的值在当前上下文环境中已经存在，那么 get_variable 会取出已有对象，并返回。

如果 `name` 指定的值不存在，则 `get_varialbe` 内部会创建一个 blob 对象，并返回。

### 使用 get_variable 创建对象

`oneflow.get_variable` 的原型如下：

```python
def get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
)
```

以下是 `oneflow.layers.conv2d` 中，使用 get_variable 创造参数变量，并进一步构建网络的例子：

```python
    #...
    weight = flow.get_variable(
        weight_name if weight_name else name_prefix + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=kernel_initializer
        if kernel_initializer is not None
        else flow.constant_initializer(0),
        regularizer=kernel_regularizer,
        trainable=trainable,
        model_name="weight",
    )

    output = flow.nn.conv2d(
        inputs, weight, strides, padding, data_format, dilation_rate, groups=groups, name=name
    )
    #...
```

### initializer 设置初始化方式

我们在上文中已经看到，在调用 `get_variable` 时，通过设置初始化器 `initializer` 来指定参数的初始化方式，OneFlow 中提供了多种初始化器，它们在 `oneflow/python/ops/initializer_util.py` 中。

设置 `initializer` 后，初始化工作由 OneFlow 框架完成，具体时机为：当用户调用下文中的 `CheckPoint.init` 时，OneFlow 会根据 `initializer` 对所有 get_variable 创建的对象进行 **数据初始化**。

以下列举部分常用的 `initializer` ：

* constant_initializer

* zeros_initializer

* ones_initializer

* random_uniform_initializer

* random_normal_initializer

* truncated_normal_initializer

* glorot_uniform_initializer

* variance_scaling_initializer

* kaiming_initializer




## OneFlow 模型的 python 接口

我们通过 `oneflow.train.CheckPoint()` 实例化得到 CheckPoint 对象。
在 `CheckPoint` 类有三个关键方法：

* `init` : 根据缺省的初始化方式，初始化参数变量；

* `save` : 负责保存当前的模型到指定路径；

* `load` : 从指定`path`中导入模型值，并用这些值初始化相应的参数变量。

`init` 的原型如下，在训练开始前，我们需要调用 `init` 初始化网络中的参数变量。

```python
def init(self)
```

`save` 的原型如下，可以将模型保存至 `path` 所指定的路径。
```python
def save(self, path)
```

`load` 的原型如下，可以加载之前已经保存的，由 `path` 路径所指定的模型。

```python
def load(self, path)
```

### 调用 init 初始化模型
在训练开始前，我们需要先获取 `CheckPoint` 对象，再调用其中的 `init` 方法初始其中的网络参数。
如以下示例:

```python
check_point = flow.train.CheckPoint() #构造CheckPoint对象
check_point.init() #初始化网络参数

#... 调用作业函数等操作
```

### 调用 save 保存模型

训练过程的任意阶段，都可以通过调用 `CheckPoint` 对象的 `save` 方法来保存模型。
```python
check_point.save('./path_to_save')
```
注意：

* `save` 参数所指定路径对应的目录要么不存在，要么应该为空目录，否则 `save` 会报错(防止覆盖掉原有保存的模型)
* OneFlow 模型以一定的组织形式保存在指定的路径中，具体结构参见下文中的OneFlow模型的存储结构
* 虽然OneFlow对 `save` 的频率没有限制，但是过高的保存频率，会加重磁盘及带宽等资源的负担。

### 调用 load 加载模型
通过调用 `CheckPoint` 对象的 `load` 方法，可以从指定的路径中加载模型。

以下代码，构造 `CheckPoint` 对象并从指定路径加载模型：
```python
check_point = flow.train.CheckPoint() #构造对象
check_point.load("./path_to_model") #加载先前保存的模型
```


## OneFlow 模型的存储结构
OneFlow 模型是一组已经被训练好的网络的 **参数值** ，目前OneFlow的模型中没有包括网络的元图信息（Meta Graph）。
模型所保存的路径下，有多个子目录，每个子目录对应了 `作业函数` 中模型的 `name` 。
比如，我们先通过代码定义以下的模型：

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
假设在训练过程中，我们调用以下代码保存模型：
```python
check_point = flow.train.CheckPoint()
check_point.save('./lenet_models_name') 
```
那么 `lenet_models_name` 及其子目录结构为：
```
lenet_models_name
├── conv1-bias
│   └── out
├── conv1-weight
│   └── out
├── conv2-bias
│   └── out
├── conv2-weight
│   └── out
├── hidden-bias
│   └── out
├── hidden-weight
│   └── out
├── outlayer-bias
│   └── out
├── outlayer-weight
│   └── out
├── snapshot_done
└── System-Train-TrainStep-train_job
    └── out
```

可以看到：

* 作业函数中的网络模型，每个变量对应一个子目录

* 以上每个子目录中，都有一个 `out` 文件，它是以二进制的方式存储的网络参数信息。`out` 是默认文件名，可以通过设置网络中的 `variable op` 修改。

* `snapshot_done` 是一个空文件，如果它存在，表示网络已经训练完成

* `System-Train-TrainStep-train_job` 中保存有快照的训练步数


## 模型的微调与扩展

在模型的微调和迁移学习中，我们经常需要：

* 模型中的一部分参数加载自原有模型

* 模型中的另一部分（新增的）参数需要初始化

对此，OneFlow 的 `flow.train.CheckPoint.load` 内部，预设了以下流程：

* 按照作业函数中所描述的网络模型，遍历模型保存的路径，尝试加载各个参数

* 如果找到了对应的参数，则加载该参数

* 如果没有找到，则自动初始化，同时打印警告提醒已经自动初始化部分参数

在 OneFlow Benchmark 的 [BERT](../adv_examples/bert.md) 中，可以看到微调的实际应用。

以下举一个用于阐述概念的简单例子。

首先，我们先定义一个模型，训练后保存至 `./mlp_models_1`：
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="dense1",
        )
        dense2 = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="dense2"
        )

        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, dense2)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

    return loss
```

然后，我们拓展网络结构，为以上模型多增加一层 `dense3`：
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        #... 原有网络结构

        dense3 = flow.layers.dense(
            dense2, 10, kernel_initializer=initializer, name="dense3"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, dense3)

    #...
```

最后，从原来保存的模型加载参数，并开始训练：
```python
if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.load("./mlp_models_1")

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
    check_point.save("./mlp_ext_models_1")
```

会得到如下输出：
```text
WARNING! CANNOT find variable path in : ./mlp_models_1/dense3-bias/out. It will be initialized. 
WARNING! CANNOT find variable path in : ./mlp_models_1/dense3-weight/out. It will be initialized. 
2.8365176
0.38763675
0.24882479
0.17603233
...
```
表示新增的 `dense3` 层所需的参数在原保存的模型中没有找到，并且已经自动初始化。

### 完整代码

以下代码来自 [mlp_mnist_origin.py](../code/basics_topics/mlp_mnist_origin.py)，作为“骨干网络”，将训练好的模型保存至 `./mlp_models_1`。
```python
# mlp_mnist_origin.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="dense1",
        )
        dense2 = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="dense2"
        )

        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, dense2)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

    return loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
    check_point.save("./mlp_models_1")
```

以下代码来自 [mlp_mnist_finetune.py](../code/basics_topics/mlp_mnist_finetune.py)，“微调”（为骨干网络增加一层`dense3`）后，加载 `./mlp_models_1`，并继续训练。
```python
# mlp_mnist_finetune.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="dense1",
        )
        dense2 = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="dense2"
        )

        dense3 = flow.layers.dense(
            dense2, 10, kernel_initializer=initializer, name="dense3"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, dense3)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

    return loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.load("./mlp_models_1")

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
    check_point.save("./mlp_ext_models_1")
```
