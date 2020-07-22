# Loading and saving of model

For loading and saving for model, the common scenario have:

* Save the model which have been training for a while and make it ready for next train.

* Save the model have completed training and make it ready for deployment.

Strictly speaking, to save a incompleted model is called save `checkpoint` or `snapshot`.It is different with `model saving` of a completed model.

But, no matter is the model have completed the training process, we can use **unified port**. Thus, like the `model`、`checkpoint`、`snapshot` we saw in other framework is no difference in OneFlow framework.In OneFlow, we all use `flow.train.CheckPoint` as port controls.

In this article, we will introduce:

* How to create model parameters

* How to save and load model

* Storage structure of OneFlow model

* Part of the initialization technique of the model

## Use get_variable to create/access object of model parameters

We can use `oneflow.get_variable` to create or obtain an object. This object could used for submitting information with global job function. When calling the port of `OneFlow.CheckPoint`. This object also will be store automatically or recover from storage devices.

Because of this character, the object create by `get_variable` always used in store model parameters.In fact, there are many high levels ports in OneFlow like `oneflow.layers.conv2d`. We use `get_variable` to create model parameters.

### Process of get_variable get/create object

`Get_variable`  need a specified `name`. This parameter will be the name when create a object.

If the` name` value is existing in the program, then get_variable will get the existing object and return.

If the` name` value is not existing in the program, then get_variable will create a blob object and return.

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

We use `oneflow.train.CheckPoint()` to achieve object of CheckPoint. 在 `CheckPoint` 类有三个关键方法：

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
在训练开始前，我们需要先获取 `CheckPoint` 对象，再调用其中的 `init` 方法初始其中的网络参数。 如以下示例:

```python
check_point = flow.train.CheckPoint() #构造CheckPoint对象
check_point.init() #初始化网络参数

#... 调用任务函数等操作
```

### 调用 save 保存模型

训练过程的任意阶段，都可以通过调用 `CheckPoint` 对象的 `save` 方法来保存模型。
```python
check_point.save('./path_to_save')
```
注意：

* 保存的路径必须为空，否则 `save` 会报错

* 虽然OneFlow对 `save` 的频率没有限制，但是过高的保存频率，会加重磁盘及贷款等资源的负担。

* OneFlow 模型以一定的组织形式保存在指定的路径中，具体结构参见下文中的OneFlow模型的存储结构

### 调用 load 加载模型
通过调用 `CheckPoint` 对象的 `load` 方法，可以从指定的路径中加载模型。 注意，从磁盘中加载的模型需要与当前任务函数中使用使用的网络模型匹配，否则会出错。

以下代码，构造 `CheckPoint对象` 并从指定路径加载模型：
```python
check_point = flow.train.CheckPoint() #构造对象
check_point.load("./path_to_model") #加载先前保存的模型
```


## OneFlow模型的存储结构
OneFlow 模型是一组已经被训练好的网络的 **参数值** ，目前OneFlow的模型中没有包括网络的元图信息（Meta Graph）。 模型所保存的路径下，有多个子目录，每个子目录对应了 `任务函数` 中模型的 `name` 。 比如，我们先通过代码定义以下的模型：

```python
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

* 任务函数中的网络模型，每个变量对应一个子目录

* 以上每个子目录中，都有一个 `out` 文件，它是以二进制的方式存储的网络参数信息。`out` 是默认文件名，可以通过设置网络中的 `variable op` 修改。

* `snapshot_done` 是一个空文件，如果它存在，表示网络已经训练完成

* `System-Train-TrainStep-train_job` 中保存有快照的训练步数


## 常见问题

目前 OneFlow 框架支持了模型处理方面最基础的功能，在实际的操作中可能会碰到一些问题，这里罗列一些。

### 模型参数的初始化
在进行网络的训练或者推理前，需要初始化模型，也就是初始化网络中的参数变量(variable op)，否则这些参数的初始值就很可能不符合期待。

为网络中的参数填充值的方式有两种：

* 调用前面介绍的 `init` 函数，这样每个参数变量(variable op)都会根据自己的初始化方式进行初始化；

* 调用 `load` 函数，从指定目录中读取用于初始化的值。

### 模型部分初始化和部分导入
实际使用中经常碰到这么一些场景，特别是在系统精调或者迁移学习的时候碰到：

* 新的网络以一个经典的网络为骨干网，拓展一些新的网络结构，骨干网部分的模型已经被训练好了，训练新的网络时需要被导入(`load`)；而新拓展网络部分的模型需要被按照指定方式初始化(`init`)；

* 原来网络已经被训练，需要按照新的优化方式重新训练，新的优化方式带来了一些额外的参数变量，比如 `momentum` 或者 `adam` ；原来的参数变量需要被导入(load)，而额外的参数变量需要被初始化(init)；

总之，以上情况都属于：

* 模型中的一部分参数由 `load` 导入

* 模型中的另一部分参数由 `init` 初始化

对此，目前我们的建议是：

* 先保存扩展前的模型；

* 对于扩展后的模型，首先使用 `init` 初始化所有参数，并保存；

* 合并模型目录：将扩展前模型的子目录，覆盖掉扩展后模型的对应目录；

* 最后，用 `load` 方法加载合并后的模型，并运行训练脚本。
