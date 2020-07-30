# 配置优化算法和超参

After neural network was set up, normally need be training before use for prediction.而训练的过程就是网络模型参数(Variable)被优化的过程，通常采用反向传播算法和指定的优化器/优化策略(Optimizer)更新网络模型参数，本文重点介绍在 OneFlow 中如何设置优化策略(Optimizer)和超参(Hyperparameters)。

可以在不了解 OneFlow 设计和概念的情况下，直接采用下面的预测配置或训练配置；如果有更进一步的需要，可以参考如何配置 `model_update_conf`，自定义优化方法；本文的最后会通过逐层推进的方式，介绍 OneFlow 在设计的一些概念，详细解释如何设置训练时候的优化算法和超参数。

## 预测/推理配置
在 OneFlow 中，无论是训练还是验证、预测/推理，都需要通过装饰器 `@flow.global_function` 来指定，而优化器和超参数的设置通过函数自定义，并作为参数传递到 `@flow.global_function` 中。通过这种方式，做到了 **参数配置和任务的分离** 。

例如：下面我们定义了一个用于验证的作业函数(job function)：`eval_job`

我们通过 get_eval_config() 定义了 eval_job() 的配置，并将 get_eval_config() 作为 `@flow.global_function` 的参数，应用到eval_job()函数。

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config

@flow.global_function(get_eval_config())
def eval_job() -> tp.Numpy:
  # build up NN here
```
当然，上面的 `get_eval_config` 中只配置了网络的基本参数。在下面的例子中，我们将介绍一个训练作业，并配置学习率 lr 和 sgd 优化器等参数。

## Configuration of training
同样，只要按照下面的方式给 `train_job` 函数配上一个装饰器 `@flow.global_function())` 就能够实现一个用于训练的网络。

```python
@flow.global_function(type="train")
def train_job(images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Numpy:
    logits = mlp(images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```
其中 `type="train"` 表明作业函数是一个训练任务，学习率和优化器的设置如下：

1. 利用 `flow.optimizer.PiecewiseConstantScheduler` 设置了学习率 learning rate的策略—初始学习率为0.1的分段缩放策略。当然，你也可以使用其他的学习率策略如：`flow.optimizer.CosineSchedule`r(余弦策略)

2. 在 `flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)` 设置了 optimizer 优化器/优化算法。(naive_conf 即使用默认的 sgd 优化算法)

## 配置 `model_update_conf`
OneFlow 中很多数据结构都是用 [protobuf](https://developers.google.cn/protocol-buffers/) 描述的，python的字典对象能够方便的转换成 protobuf 对象，`model_update_conf` 的输入要求是一个 python 字典对象，我们需要参考下面的 protobuf 定义，构建好一个 python 字典对象，传给 `model_update_conf` 作为输入。
```protobuf
message NormalModelUpdateOpUserConf {
  optional LearningRateDecayConf learning_rate_decay = 1;
  optional WarmupConf warmup_conf = 2;
  optional ClipConf clip_conf = 3;
  optional WeightDecayConf weight_decay_conf = 4;
  oneof normal_mdupdt {
    NaiveModelUpdateConf naive_conf = 1000;
    MomentumModelUpdateConf momentum_conf = 1001;
    RMSPropModelUpdateConf rmsprop_conf = 1002;
    LARSModelUpdateConf lars_conf = 1003;
    AdamModelUpdateConf adam_conf = 1004;
    LazyAdamModelUpdateConf lazy_adam_conf = 1005;
  }
}
```

### Choosing optimization algorithm
从上面的定义中可以看到，目前 OneFlow 支持6种优化算法，分别是：

- `naive_conf` 代表 SGD

- `momentum_conf`

- `rmsprop_conf`

- `lars_conf`

- `adam_conf`

- `lazy_adam_conf`

We must choose one of these algorithms. For example the previous code is using `naive_conf` and the syntax is:

```
config.train.model_update_conf({"naive_conf": {}})
```

`naive_conf` 不需要额外配置参数，所以传入的字典是 `{"naive_conf": {}}`，其 key 是 `"naive_conf"` ，value 是一个空的字典`{}`。

如果选择其他的优化器，就需要配置相应的参数，如下面配置了一个惯量为0.875的 SGD 优化器，key 是 `momentum_conf` , value 是一个非空的字典`{'beta': 0.875}`。

```
config.train.model_update_conf({"momentum_conf": {'beta': 0.875}})
```

这里不对每个优化器做详细说明，详细请参考[optimizer api](https://oneflow-api.readthedocs.io/en/latest/optimizer.html)

### Other optimizations
The difinition previously have 4 more optional optimisations.

- `learning_rate_decay` - 学习率的衰减方式

- `warmup_conf` - 学习率预热方式

- `clip_conf` - 梯度截取

- `weight_decay_conf` - 权重衰减

这4个选项可以不选或多选，配置方式就是在 python 字典中加入新的key-value 项，详细请参考[optimizer api](https://oneflow-api.readthedocs.io/en/latest/optimizer.html)，下面仅举出两种形式的例子供参考。

```python
# example 1
model_update_conf = {
    'momentum_conf': {
        'beta': 0.875
    }, 
    'warmup_conf': {
        'linear_conf': {
            'warmup_batches': 12515, 
            'start_multiplier': 0
        }
    }, 
    'learning_rate_decay': {
        'cosine_conf': {
            'decay_batches': 112635
        }
    }
}
```

```python
# example 2
model_update_conf = dict(
    adam_conf=dict(
        epsilon=1e-6
    ),
    learning_rate_decay=dict(
        polynomial_conf=dict(
            decay_batches=100000, 
            end_learning_rate=0.0,
        )
    ),
    warmup_conf=dict(
        linear_conf=dict(
            warmup_batches=1000, 
            start_multiplier=0,
        )
    ),
    clip_conf=dict(
        clip_by_global_norm=dict(
            clip_norm=1.0,
        )
    ),
)
```
## The global function and configuration of OneFlow
这个章节递进的介绍 OneFlow 全局函数的概念，函数配置的概念以及如何在函数配置中区分训练或预测配置。

### OneFlow 全局函数(OneFlow Global Function)
在介绍优化策略和超参的设置之前，需要先提到`OneFlow Global Function`这个概念，被 `oneflow.global_function` 修饰的函数就是`OneFlow Global Function`，通常也可以被称作`job function`作业函数，下面就是一个简单的例子：

```python
import oneflow as flow
@flow.global_function(flow.function_config(),type="test")
def test_job():
  # build up NN here
```
`test_job`就是一个`OneFlow Global Function`，它能够被 OneFlow 框架识别，根据配置把函数里面定义的网络编译成适合计算图，放到设备上进行计算。

作业函数包括两部分信息：使用算子搭建的网络(NN)，以及使用这个网络需要的配置信息(config)。Building networks reference to [Use OneFlow build the neural network ](build_nn_with_op_and_layer.md).This article focus on introduce how to config information.

### 函数配置(function_config)
前面的例子中，你也可能注意到 `@flow.global_function` 装饰器接受`flow.function_config()` 的返回对象作为参数。这个参数就是设置作业函数配置信息的入口。There is a more complex example:

```python
config = flow.function_config()
config.default_data_type(flow.float)

@flow.global_function(type="predict", function_config=config)
def test_job():
  # build up NN here
```
上面的例子中，通过 `function_config` 设置了网络的缺省数据类型为 float；将被用于训练；学习率是0.1；采用了 `naive_conv` 优化算法，也就是 `SGD`。

function_config 中还包含哪些配置请参考[function_config API](https://oneflow-api.readthedocs.io/en/latest/oneflow.html?highlight=functionconfig#oneflow.FunctionConfig)。

### 训练还是预测配置
默认情况下，作业函数只能做预测作业，如果想要做训练作业，需要设置 `type=train` 属性。

```python
@flow.global_function(type="train")
def train_job():
    #网络模型...
```

并且，通过 `flow.optimizer` 下的方法设置学习速率及优化方法，如以下代码设置了学习率和模型更新的策略(针对 `loss` 变量，使用 SGD)：

```python
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
```

反之，如果省略掉以上配置，那么得到的就是作业函数就可用于预测。

## Summary

一个 OneFlow 的全局函数由 `@oneflow.global_function` 修饰，解耦了网络的搭建过程和任务相关配置(function_config)，`function_config` **采取集中配置的方式，既方便任务切换，又方便集群调度配置。**
