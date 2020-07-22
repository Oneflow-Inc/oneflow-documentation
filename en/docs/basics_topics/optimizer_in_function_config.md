# Using of flow.function_config() to config optimization algorithm and parameters
After neural network was set up, normally need be training before use for prediction.The process of training is optimize the parameters(Variable) of network. Usually use the back propagation algorithm and special optimizer/ optimization strategy to update the network model parameters. In this article, we will focus on how to set optimizer and hyperparameters in OneFlow.

We can directly use `prediction the configuration` or `training of configuration` when we not familiar with the design and concept of OneFlow. If have any further requirements, you can reference `how to config model_update_conf`. The end of this article, we will explain step by step to introduce some concepts of OneFlow when design it. And explain in detail about how to config  optimizer and hyperparameters when training.

## Configuration of prediction
In OneFlow, no matter training, evaluation or prediction all need use @flow.global_function to specified. But the parameters of optimizer and hyperparameters will send to @flow.global_function as parameter by custom function.In this way, we achieve **parameters and configuration separate from main job</0>. </p>

For example, we define a job for evaluating: `eval_job`()

We use get_eval_config() to define the configurations of eval_job() and use get_eval_config() as the parameter of @flow.global_function to send to eval_job() function.

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config

@flow.global_function(get_eval_config())
def eval_job():
  # build up NN here
```
Of course, the `get_eval_config` above just config the basic parameters in the network.In the example below, we will introduce a training job and config learning rate and sgd optimizer.

## Configuration of training
As same, just following the instructions below and give `train_job` to add a decorator then `@flow.global_function(get_train_config())` are able to achieve a network which can use for training.
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
The `get_train_config` inside is defined the parameters of train_job. The main parameters:
1. Use `config.train.primary_lr` to set learning rate；
2. In `config.train.model_update_conf` define the optimizer and optimization algorithm.(naive_conf is using the default sgd optimization algorithm)

## Config `model_update_conf`
Because there are lots of data frame is defined by protobuf in OneFlow. The dictionary in python can convert to protobuf easier. The input of `model_update_conf` is expecting a dictionary of python. We need to reference the definition of protobuf below to create a good python dictionary and pass that dictionary to `model_update_conf` as input.
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
From the definition above, we can see OneFlow is support 6 types of optimization algorithm. They are:
- `naive_conf` represent SGD
- `momentum_conf`
- `rmsprop_conf`
- `lars_conf`
- `adam_conf`
- `lazy_adam_conf`

We must choose one of these algorithms. For example the previous code is using `naive_conf` and the syntax is:
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
