# Loading and saving of model

For loading and saving for model, the common scenario have:

* Save the model which have been training for a while and make it ready for next train.

* Save the model have completed training and make it ready for deployment.

Strictly speaking, to save a incompleted model is called save `checkpoint` or `snapshot`.It is different with `model saving` of a completed model.

不过，无论模型是否训练完毕，我们都可以使用 **统一的接口** 将其保存，因此，我们在其它框架中看到的`model`、`checkpoint`、`snapshot`，在OneFlow的操作中不做区分。它们在OneFlow中，都通过`flow.train.CheckPoint`类作为接口操作。它们在OneFlow中，都通过`flow.train.CheckPoint`类作为接口操作。

本文将介绍：

* 如何创建模型参数

* 如果保存/加载模型

* OneFlow模型的存储结构

* 模型的部分初始化技巧

## 使用get_variable创建/获取模型参数对象

我们可以使用`oneflow.get_variable`方法创造或者获取一个对象，该对象可以用于在全局任务函数中交互信息；当调用`OneFlow.CheckPoint`的对应接口时，该对象也会被自动地保存或从存储设备中恢复。

因为这个特点，`get_variable`创建的对象，常用于存储模型参数。因为这个特点，`get_variable`创建的对象，常用于存储模型参数。实际上，OneFlow中很多较高层接口（如`oneflow.layers.conv2d`），内部使用`get_variable`创建模型参数。

### get_variable获取/创建对象的流程

`get_variable`需要一个指定一个`name`参数，该参数作为创建对象的标识。

如果`name`指定的值在当前上下文环境中已经存在，那么get_variable会取出已有对象，并返回。

如果`name`指定的值不存在，则get_varialbe内部会创建一个blob对象，并返回。

### 使用get_variable创建对象

`oneflow.get_variable`的原型如下：

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

以下是`oneflow.layers.conv2d`中，使用get_variable创造参数变量，并进一步构建网络的例子：

```python
    #... weight = flow.get_variable(
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

### initializer设置初始化方式

我们在上文中已经看到，在调用`get_variable`时，通过设置初始化器`initializer`来指定参数的初始化方式，OneFlow中提供了多种初始化器，它们在`oneflow/python/ops/initializer_util.py`中。

设置`initializer`后，初始化工作由OneFlow框架完成，具体时机为：当用户调用下文中的`CheckPoint.init`时，OneFlow会根据`initializer`对所有get_variable创建的对象进行 **数据初始化**。

以下列举部分常用的`initializer`：

* constant_initializer

* zeros_initializer

* ones_initializer

* random_uniform_initializer

* random_normal_initializer

* truncated_normal_initializer

* glorot_uniform_initializer

* variance_scaling_initializer

* kaiming_initializer




## OneFlow模型的python接口

我们通过`oneflow.train.CheckPoint()`实例化得到CheckPoint对象。 在`CheckPoint`类有三个关键方法： 在`CheckPoint`类有三个关键方法：

* `init` : 根据缺省的初始化方式，初始化参数变量；

* `save` : 负责保存当前的模型到指定路径；

* `load` : 从指定`path`中导入模型值，并用这些值初始化相应的参数变量。

`init`的原型如下，在训练开始前，我们需要调用`init`初始化网络中的参数变量。

```python
def init(self)
```

`save`的原型如下，可以将模型保存至`path`所指定的路径。
```python
def save(self, path)
```

`load`的原型如下，可以加载之前已经保存的，由`path`路径所指定的模型。
```python
def load(self, path)
```

### 调用init初始化模型
在训练开始前，我们需要先获取`CheckPoint`对象，再调用其中的`init`方法初始其中的网络参数。 如以下示例: 如以下示例:

```python
check_point = flow.train.CheckPoint() #构造CheckPoint对象
check_point.init() #初始化网络参数

#... 调用任务函数等操作
```

### 调用save保存模型

训练过程的任意阶段，都可以通过调用`CheckPoint`对象的`save`方法来保存模型。
```python
check_point.save('./path_to_save')
```
注意：

* 保存的路径必须为空，否则`save`会报错

* 虽然OneFlow对`save`的频率没有限制，但是过高的保存频率，会加重磁盘及贷款等资源的负担。

* OneFlow模型以一定的组织形式保存在指定的路径中，具体结构参见下文中的`OneFlow模型的存储结构`

### 调用load加载模型
通过调用`CheckPoint`对象的`load`方法，可以从指定的路径中加载模型。 注意，从磁盘中加载的模型需要与当前任务函数中使用使用的网络模型匹配，否则会出错。 注意，从磁盘中加载的模型需要与当前任务函数中使用使用的网络模型匹配，否则会出错。

以下代码，构造`CheckPoint对象`并从指定路径加载模型：
```python
check_point = flow.train.CheckPoint() #构造对象
check_point.load("./path_to_model") #加载先前保存的模型
```


## OneFlow模型的存储结构
OneFlow模型是一组已经被训练好的网络的`参数值`，目前OneFlow的模型中没有包括网络的元图信息（Meta Graph）。 模型所保存的路径下，有多个子目录，每个子目录对应了`任务函数`中模型的`name`。 比如，我们先通过代码定义以下的模型： 模型所保存的路径下，有多个子目录，每个子目录对应了`任务函数`中模型的`name`。 比如，我们先通过代码定义以下的模型：

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
那么`lenet_models_name`机器子目录结构为：
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

* 以上每个子目录中，都有一个`out`文件，它是以二进制的方式存储的网络参数信息。`out`是默认文件名，可以通过设置网络中的`variable op`修改。`out`是默认文件名，可以通过设置网络中的`variable op`修改。

* `snapshot_done`是一个空文件，如果它存在，表示网络已经训练完成

* `System-Train-TrainStep-train_job`中保存有快照的训练步数


## 常见问题

目前OneFlow框架支持了模型处理方面最基础的功能，在实际的操作中可能会碰到一些问题，这里罗列一些。

### 模型参数的初始化
在进行网络的训练或者推理前，需要初始化模型，也就是初始化网络中的参数变量（variable op），否则这些参数的初始值就很可能不符合期待。

为网络中的参数填充值的方式有两种：

* 调用前面介绍的`init`函数，这样每个参数变量（variable op）都会根据自己的初始化方式进行初始化；

* 调用`load`函数，从指定目录中读取用于初始化的值。

### 模型部分初始化和部分导入
实际使用中经常碰到这么一些场景，特别是在系统精调或者迁移学习的时候碰到：

* 新的网络以一个经典的网络为骨干网，拓展一些新的网络结构，骨干网部分的模型已经被训练好了，训练新的网络时需要被导入（`load`）；而新拓展网络部分的模型需要被按照指定方式初始化（`init`）；

* 原来网络已经被训练，需要按照新的优化方式重新训练，新的优化方式带来了一些额外的参数变量，比如`momentum`或者`adam`；原来的参数变量需要被导入（load），而额外的参数变量需要被初始化（init）；

总之，以上情况都属于：

* 模型中的一部分参数由`load`导入

* 模型中的另一部分参数由`init`初始化

对此，目前我们的建议是：

* 先保存扩展前的模型；

* 对于扩展后的模型，首先使用`init`初始化所有参数，并保存；

* 合并模型目录：将扩展前模型的子目录，覆盖掉扩展后模型的对应目录；

* 最后，用`load`方法加载合并后的模型，并运行训练脚本。
