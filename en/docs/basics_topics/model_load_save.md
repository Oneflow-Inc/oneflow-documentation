# Loading and saving of model

For loading and saving for model, the common scenario have:

* Save the model which have been training for a while and make it ready for next train.

* Save the model have completed training and make it ready for deployment.

Strictly speaking, to save a incompleted model is called save `checkpoint` or `snapshot`.It is different with `model saving` of a completed model.

But, no matter is the model have completed the training process, we can use **unified port**. Thus, like the `model`、`checkpoint`、`snapshot` we saw in other framework is no difference in OneFlow framework.In OneFlow, we all use `flow.train.CheckPoint` as port controls.

In this article, we will introduce:

* How to create model parameters

* 如何保存/加载模型

* Storage structure of OneFlow model

* Part of the initialization technique of the model

## Use get_variable to create/access object of model parameters

我们可以使用 `oneflow.get_variable` 方法创造或者获取一个对象，该对象可以用于在全局作业函数中交互信息；当调用 `oneflow.CheckPoint` 的对应接口时，该对象也会被自动地保存或从存储设备中恢复。

Because of this character, the object create by `get_variable` always used in store model parameters.In fact, there are many high levels ports in OneFlow like `oneflow.layers.conv2d`. We use `get_variable` to create model parameters.

### Process of get_variable get/create object

`Get_variable`  need a specified `name`. This parameter will be the name when create a object.

If the` name` value is existing in the program, then get_variable will get the existing object and return.

如果 `name` 指定的值不存在，则 `get_varialbe` 内部会创建一个 blob 对象，并返回。

### Use get_variable create object

The prototype of `oneflow.get_variable`:

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

Next code is an example of in `oneflow.layers.conv2d`, use get_variable to create parameters and keep forward build network:

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

    output = flow.nn.conv2d(
        inputs, weight, strides, padding, data_format, dilation_rate, groups=groups, name=name
    )
    #...
```

### Initializer setting

In the previous chapters, when we calling `get_variable`, we specified the method of initiating the parameters by `initializer`. In OneFlow, we provide many initializer which can be find in `oneflow/python/ops/initializer_util.py`

After config `initializer`, the initialize work is done by OneFlow framework. Exactly time was: when user called the `CheckPoint.init` later on, OneFlow will initialize all data created by get_variable according to `initializer`.

Some common `initializer`:

* constant_initializer

* zeros_initializer

* ones_initializer

* random_uniform_initializer

* random_normal_initializer

* truncated_normal_initializer

* glorot_uniform_initializer

* variance_scaling_initializer

* kaiming_initializer




## The python port of OneFlow

We use `oneflow.train.CheckPoint()` to achieve object of CheckPoint. There are three critical methods in `CheckPoint`:

* `init` : According to method of lacking to initializa parameters.

* `save` : Responsible for save the current model to the specified path.

* `load` : Import the model from `path` and use the model to initialize parameters.

The `init`  work like this. Before you training, we need use  `init` to initialize the parameters in net work.

```python
def init(self)
```

The `save` work like this. It could save the model under a specified  `path`.
```python
def save(self, path)
```

The `load` work like this. Can load the model we train perviously from the specified  `path`.
```python
def load(self, path)
```

### Initialize model
Before training, we need get the object of  `CheckPoint` then called the  `init` to initialize the parameters in network. For example:

```python
check_point = flow.train.CheckPoint() #构造CheckPoint对象
check_point.init() #初始化网络参数

#... 调用作业函数等操作
```

### Save model

At any step of training process, we can called the `save`  which is the obejct of `CheckPoint`  to save model.
```python
check_point.save('./path_to_save')
```
Attention:

* `save` 参数所指定路径对应的目录要么不存在，要么应该为空目录，否则 `save` 会报错(防止覆盖掉原有保存的模型)
* OneFlow 模型以一定的组织形式保存在指定的路径中，具体结构参见下文中的OneFlow模型的存储结构
* 虽然OneFlow对 `save` 的频率没有限制，但是过高的保存频率，会加重磁盘及带宽等资源的负担。

### Load model
We can called the `load` which is the obejct of `CheckPoint` to load model from specificed path. 注意，从磁盘中加载的模型需要与当前作业函数中使用使用的网络模型匹配，否则会出错。

There is a example of load model from a specific path and construct  `CheckPoint object` :
```python
check_point = flow.train.CheckPoint() #constructing object 
check_point.load("./path_to_model") #load model
```


## The structure of OneFlow saved model
Model of OneFlow are the **parameters** of network. For now there are no Meta Graph information in OneFlow model. 模型所保存的路径下，有多个子目录，每个子目录对应了 `作业函数` 中模型的 `name` 。 For example, we define the model in the first place:

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
Assume that in the process of training, we called the following code to save model:
```python
check_point = flow.train.CheckPoint()
check_point.save('./lenet_models_name') 
```
Then  `lenet_models_name` and the subdirectories will look like:
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

We can see:

* 作业函数中的网络模型，每个变量对应一个子目录

* All subdirectories above have a  `out` document. It store the parameters of network in binary.`out` is the default file name. We can change that by  `variable op` in the network.

* `snapshot_done` is an empty folder. If it exists, means the network is completed training.

* Snapshots of the training steps is store in `System-Train-TrainStep-train_job`.


## 模型的微调与扩展

我们在系统微调(finetune)或者迁移学习的时候常碰到以下场景：

* 微调(finetune)：以一个已经训练好的骨干网络(backbone)为基础，拓展一些新的网络结构后，继续训练。原有骨干网络那部分模型需要加载原来保存的参数，而新拓展网络部分的模型需要初始化；

* 迁移学习：在已经训练好的原网络基础上，按照新的优化方式重新训练，新的优化方式带来了一些额外的参数变量，比如 `momentum` 或者 `adam` ；原来的参数变量加载自原来保存的参数，而额外的参数变量需要被初始化；

总之，以上情况都属于：

* 模型中的一部分参数加载自原有模型

* 模型中的另一部分（新增的）参数需要初始化

对此，OneFlow 的 `flow.train.CheckPoint.load` 内部，预设了以下流程：

* 按照作业函数中所描述的网络模型，遍历模型保存的路径，尝试加载各个参数

* 如果找到了对应的参数，则加载该参数

* 如果没有找到，则自动初始化，同时打印警告提醒已经自动初始化部分参数

在 OneFlow Benchmark 的 [BERT](../adv_examples/bert.md) 中，可以看到微调的实际应用。

以下举一个用于阐述概念的简单例子。

首先，我们先定义一个模型，形状如下，训练后保存至 `./mlp_models_1`：
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