# 使用flow.function_config()配置优化算法和超参
当一个神经网络被搭建好之后，通常是需要经过训练之后才能够被拿来做预测。当一个神经网络被搭建好之后，通常是需要经过训练之后才能够被拿来做预测。而训练的过程就是网络模型参数（Variable）被优化的过程，通常采用反向传播算法和指定的优化器/优化策略（Optimizer）更新网络模型参数，本文重点介绍在OneFlow中如何设置优化策略(Optimizer)和超参（Hyperparameters）。

可以在不了解OneFlow设计和概念的情况下，直接采用下面的`预测配置`或`训练配置`；如果有更进一步的需要，可以参考`如何配置model_update_conf`，自定义优化方法；本文的最后会通过逐层推进的方式，介绍OneFlow在设计的一些概念，详细解释如何设置训练时候的优化算法和超参数。

## 预测配置
在OneFlow中，无论是训练还是验证、预测，都需要通过装饰器@flow.global_function来指定，而优化器和超参数的设置通过函数自定义，并作为参数传递到@flow.global_function中。通过这种方式，做到了**参数/配置和任务的分离。**通过这种方式，做到了**参数/配置和任务的分离。**

例如：下面我们定义了一个用于验证的任务(Job)：`eval_job`()

我们通过get_eval_config()定义了eval_job()的配置，并将get_eval_config()作为@flow.global_function的参数传递到eval_job()函数。

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config

@flow.global_function(get_eval_config())
def eval_job():
  # build up NN here
```
当然，上面的`get_eval_config`中只配置了网络的基本参数。在下面的例子中，我们将介绍一个训练任务，并配置学习率lr和sgd优化器等参数。在下面的例子中，我们将介绍一个训练任务，并配置学习率lr和sgd优化器等参数。

## 训练配置
同样，只要按照下面的方式给`train_job`函数配上一个装饰器`@flow.global_function(get_train_config())`就能够实现一个用于训练任务的网络。
```python
def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config

@flow.global_function(get_train_config())
def train_job():
  # build up NN here
```
其中`get_train_config`是定义了训练任务(train_job)中的配置，主要的参数设置如下：
1. 利用`config.train.primary_lr`设置了学习率learning rate；
2. 在`config.train.model_update_conf`设置了optimizer优化器/优化算法。(naive_conf即使用默认的sgd优化算法)(naive_conf即使用默认的sgd优化算法)

## 配置`model_update_conf`
因为OneFlow中很多数据结构都是用protobuf定义的，python的字典对象能够方便的转换成protobuf对象，`model_update_conf`的输入要求是一个python字典对象，我们需要参考下面的protobuf定义，构建好一个python字典对象，传给`model_update_conf`作为输入。
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

### 选择优化算法
从上面的定义中可以看到，目前OneFlow支持6种优化算法，分别是：
- `naive_conf`代表SGD
- `momentum_conf`
- `rmsprop_conf`
- `lars_conf`
- `adam_conf`
- `lazy_adam_conf`

这六种算法必须选择其一，比如前面的例子中选择的是`naive_conf`，语法是：
```
config.train.model_update_conf({"naive_conf": {}})
```
`naive_conf`不需要额外配置参数，所以传入的字典是`{"naive_conf": {}}`，其key是`"naive_conf"`，value是一个空的字典`{}`。

如果选择其他的优化器，就需要配置相应的参数，如下面配置了一个惯量为0.875的SGD优化器，key是`momentum_conf`,value是一个非空的字典`{'beta': 0.875}`。
```
config.train.model_update_conf({"momentum_conf": {'beta': 0.875}})
```
这里不对每个优化器做详细说明，详细请参考[optimizer api](http://183.81.182.202:8000/html/train.html#)
### 其他优化选项
前面的定义中还有4个可选的优化选项：
- `learning_rate_decay` - 学习率的衰减方式
- `warmup_conf` - 学习率预热方式
- `clip_conf` - 梯度截取
- `weight_decay_conf` - 权重衰减

这4个选项可以不选或多选，配置方式就是在python字典中加入新的key-value项，详细请参考[optimizer api](http://183.81.182.202:8000/html/train.html#)，下面仅举出两种形式的例子供参考。
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
## OneFlow的全局函数和配置
这个章节递进的介绍OneFlow全局函数的概念，函数配置的概念以及如何在函数配置中区分训练或预测配置。
### OneFlow 全局函数（OneFlow Global Function）
在介绍优化策略和超参的设置之前，需要先提到`OneFlow Global Function`这个概念，被oneflow.global_function修饰的函数就是`OneFlow Global Function`，通常也可以被称作`job function`任务函数，下面就是一个简单的例子：
```python
import oneflow as flow
@flow.global_function(flow.function_config())
def test_job():
  # build up NN here
```
`test_job`就是一个`OneFlow Function`，它能够被OneFlow框架识别，根据配置把函数里面定义的网络编译成适合计算图，放到设备上进行计算。

任务函数包括两部分信息：使用算子搭建的网络（NN），以及使用这个网络需要的配置信息（config）。网络的搭建请参考[如何使用OneFlow搭建网络](build_nn_with_op_and_layer.md)。本文专注介绍如何设置配置信息。网络的搭建请参考[如何使用OneFlow搭建网络](build_nn_with_op_and_layer.md)。本文专注介绍如何设置配置信息。

### 函数配置（function_config）
前面的例子中，你也可能注意到`@flow.global_function`装饰器接受一个参数`flow.function_config()`。这个参数`function_config`就是设置`OneFlow Function`配置信息的入口。比如下面这个略微复杂一点的例子：这个参数`function_config`就是设置`OneFlow Function`配置信息的入口。比如下面这个略微复杂一点的例子：

```python
config = flow.function_config()
config.default_data_type(flow.float)
config.train.primary_lr(0.1)
config.train.model_update_conf({"naive_conf": {}})

@flow.global_function(config)
def test_job():
  # build up NN here
```
上面的例子中，通过`function_config`设置了网络的缺省数据类型为float；将被用于训练；学习率是0.1；采用了`naive_conv`优化算法，也就是`SGD`。

function_config中还包含哪些配置请参考[function_config API](http://183.81.182.202:8000/html/oneflow.html).

### train or not
function_config里面有好多可以设置的属性，这里着重介绍`train`。通常`train`不会被设置，这种情况下`job function`只能做预测任务。一旦`train`被设置，`job function`就是一个训练任务了，如下面代码所示设置了学习率和模型更新的策略（优化算法）：通常`train`不会被设置，这种情况下`job function`只能做预测任务。一旦`train`被设置，`job function`就是一个训练任务了，如下面代码所示设置了学习率和模型更新的策略（优化算法）：
```python
config.train.primary_lr(0.1)
config.train.model_update_conf({"naive_conf": {}})
```

## 总结
一个OneFlow的全局函数由`@oneflow.global_function`修饰，解耦了网络的搭建过程和任务相关配置（function_config)，`function_config`**采取集中配置的方式，既方便任务切换，又方便集群调度配置。**</strong>
