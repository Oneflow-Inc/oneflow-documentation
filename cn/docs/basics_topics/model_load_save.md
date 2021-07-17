# 模型的加载与保存

对于模型的加载与保存，常用的场景有：

* 将已经训练一段时间的模型保存，方便下次继续训练

* 将训练好的模型保存，方便后续直接部署使用

严格来说，尚未训练好的模型的保存，称为 `checkpoint` 或者 `snapshot` 。与保存已训练好的模型（`model saving`） ，在概念上，略有不同。

不过，在 OneFlow 中，无论模型是否训练完毕，我们都使用 **统一的接口** 将其保存，因此，在其它框架中看到的`model`、`checkpoint`、`snapshot` 等表述，在 OneFlow 中不做区分。

在 OneFlow 中，`flow.checkpoint` 名称空间下有模型保存、加载的接口。

本文将介绍：

* 如何创建模型参数

* 如何保存/加载模型

* OneFlow 模型的存储结构

* 如何微调与扩展模型

## get_variable 创建或获取参数

我们可以使用 [oneflow.get_variable](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.get_variable) 方法创造或者获取一个对象，该对象可以用于在全局作业函数中交互信息；当调用 `oneflow.get_all_variables` 和 `oneflow.load_variables` 接口时，可以获取或更新 `get_variable` 创建的对象的值。

因为这个特点，`get_variable` 创建的对象，常用于存储模型参数。实际上，OneFlow 中很多较高层接口（如 `oneflow.layers.conv2d`），内部使用 `get_variable` 创建模型参数。

### 流程

`get_variable` 需要指定一个 `name` 参数，该参数作为创建对象的标识。

如果 `name` 指定的值在当前上下文环境中已经存在，那么 `get_variable` 会取出已有对象，并返回。

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

我们在上文中已经看到，在调用 `get_variable` 时，通过设置初始化器 `initializer` 来指定参数的初始化方式，OneFlow 中提供了多种初始化器，可以在 [oneflow](https://oneflow.readthedocs.io/en/master/oneflow.html) 模块下查看。

在静态图机制下，设置 `initializer` 后，参数初始化工作由 OneFlow 框架自动完成。

OneFlow 目前支持的 `initializer` 列举如下，点击链接可以查看相关算法：

* [constant_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.constant_initializer)

* [zeros_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.zeros_initializer)

* [ones_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.ones_initializer)

* [random_uniform_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.random_uniform_initializer)

* [random_normal_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.random_normal_initializer)

* [truncated_normal_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.truncated_normal_initializer)

* [glorot_uniform_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.glorot_uniform_initializer)

* [glorot_normal_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.glorot_normal_initializer)

* [variance_scaling_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.variance_scaling_initializer)

* [kaiming_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.kaiming_initializer)

* [xavier_normal_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.xavier_normal_initializer)

* [xavier_uniform_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.xavier_uniform_initializer)



## OneFlow 模型的 Python 接口
**注意**：由于多版本兼容的原因，使用本节介绍的接口，在脚本中都需先配置：

```python
flow.config.enable_legacy_model_io(False)
```

### 获取/更新 variable 对象的值
我们可以使用以下两个接口，获取或更新作业函数中由 `oneflow.get_variable` 所创建的 `variable` 对象的值

- `oneflow.get_all_variables` : 获取所有作业函数中的 `variable` 对象
- `oneflow.load_variables` : 更新作业函数中的 `variable` 对象

`oneflow.get_all_variables` 会返回一个字典，字典的 key 就是创建 `variable` 时指定的 `name`，key 对应的 value 就是一个张量对象，该张量对象通过 `numpy()` 方法转为 numpy 数组。

比如，在作业函数中创建了名为 `myblob` 的对象：
```python
@flow.global_function()
def job() -> tp.Numpy:
    ...
    myblob = flow.get_variable("myblob",
        shape=(3,3),
        initializer=flow.random_normal_initializer()
        )
    ...
```
如果想打印 `myblob` 的值，可以调用：

```python
...
for epoch in range(20):
    ...
    job()
    all_variables = flow.get_all_variables()
    print(all_variables["myblob"].numpy())
    ...
```

其中的 `flow.get_all_variables` 获取到了字典，`all_variables["myblob"].numpy()` 获取了 `myblob` 对象并将其转为 numpy 数组。

与 `get_all_variables` 相反，我们可以使用 `oneflow.load_variables` 更新 variable 对象的值。
`oneflow.load_variables` 的原型如下：

```python
def load_variables(value_dict, ignore_mismatch = True)
```

使用 `load_variables` 前，我们要准备一个字典，该字典的 key 为创建 `variable` 时指定的 `name`，value 是 numpy 数组；将字典传递给 `load_variables` 后，`load_variables` 会根据 key 找到作业函数中的 variable 对象，并更新值。

如以下代码：

```python
@flow.global_function(type="predict")
def job() -> tp.Numpy:
    myblob = flow.get_variable("myblob",
        shape=(3,3),
        initializer=flow.random_normal_initializer()
        )
    return myblob

myvardict = {"myblob": np.ones((3,3)).astype(np.float32)}
flow.load_variables(myvardict)
print(flow.get_all_variables()["myblob"].numpy())
```
虽然我们选择了 `random_normal_initializer` 的初始化方式，但是因为 `flow.load_variables(myvardict)` 更新了 `myblob` 的值，所以最终输出结果是：

```text
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
```


### 模型的保存与加载
我们通过以下两个方法，可以保存/加载模型：

- `oneflow.checkpoint.save` : 负责保存当前的模型到指定路径
- `oneflow.checkpoint.get` :  从指定路径中导入模型

`save` 的原型如下，可以将模型保存至 `path` 所指定的路径。
```python
def save(path, var_dict=None)
```
可选参数 `var_dict` 如果不为 `None`，则将 `var_dict` 中指定的对象保存到指定路径。

`get` 的原型如下，可以加载之前已经保存的，由 `path` 路径所指定的模型。

```python
def get(path)
```

它将返回一个字典，该字典可以用上文介绍的 `load_variables` 方法更新到模型中：

```python
flow.load_variables(flow.checkpoint.get(save_dir))
```

**注意**：

- `save` 参数所指定路径对应的目录要么不存在，要么应该为空目录，否则 `save` 会报错(防止覆盖掉原有保存的模型)
- OneFlow 模型以一定的组织形式保存在指定的路径中，具体结构参见下文中的 OneFlow 模型的存储结构
- 虽然 OneFlow 对 `save` 的频率没有限制，但是过高的保存频率，会加重磁盘及带宽等资源的负担。

## OneFlow 模型的存储结构
OneFlow 模型是一组已经被训练好的网络的 **参数值** 。模型所保存的路径下，有多个子目录，每个子目录对应了 `作业函数` 中模型的 `name`。
比如，我们先通过代码定义如下的模型：

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
flow.checkpoint.save('./lenet_models_name')
```
那么 `lenet_models_name` 及其子目录结构为：
```
lenet_models_name/
├── conv1-bias
│   ├── meta
│   └── out
├── conv1-weight
│   ├── meta
│   └── out
├── conv2-bias
│   ├── meta
│   └── out
├── conv2-weight
│   ├── meta
│   └── out
├── dense1-bias
│   ├── meta
│   └── out
├── dense1-weight
│   ├── meta
│   └── out
├── dense2-bias
│   ├── meta
│   └── out
├── dense2-weight
│   ├── meta
│   └── out
├── snapshot_done
└── System-Train-TrainStep-train_job
    ├── meta
    └── out
```

可以看到：

* 作业函数中的网络模型，每个变量对应一个子目录

* 以上每个子目录中，都有 `out` 和 `meta` 文件，`out` 以二进制的形式存储了网络参数的值，`meta` 以文本的形式存储了网络的结构信息

* `snapshot_done` 是一个空文件，如果它存在，则表示网络已经训练完成

* `System-Train-TrainStep-train_job` 中保存有快照的训练步数


## 模型的微调与扩展

在模型的微调和迁移学习中，我们经常需要：

* 模型中的一部分参数加载自原有模型

* 模型中的另一部分（新增的）参数需要初始化

我们可以使用 `oneflow.load_variables` 完成以上操作。以下举一个用于阐述概念的简单例子。

首先，我们定义一个模型，训练后保存至 `./mlp_models_1`：
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
    flow.load_variables(flow.checkpoint.get("./mlp_models_1"))

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
    flow.checkpoint.save("./mlp_ext_models_1")
```

新增的 `dense3` 层参数，在原模型中不存在，OneFlow 会自动初始化它们的值。

### 代码

脚本 [mlp_mnist_origin.py](../code/basics_topics/mlp_mnist_origin.py) 中构建了“骨干网络”，并将训练好的模型保存至 `./mlp_models_1`。

运行：
```
wget https://docs.oneflow.org/code/basics_topics/mlp_mnist_origin.py
python3 mlp_mnist_origin.py
```

训练完成后，将会在当前工作路径下得到 `mlp_models_1` 目录。

脚本 [mlp_mnist_finetune.py](../code/basics_topics/mlp_mnist_finetune.py) 中的网络在原有基础上进行“微调”（为骨干网络增加一层`dense3`）后，加载 `./mlp_models_1`，并继续训练。

运行：
```
wget https://docs.oneflow.org/code/basics_topics/mlp_mnist_finetune.py
python3 mlp_mnist_finetune.py
```

微调后的模型，保存在 `./mlp_ext_models_1` 中。
