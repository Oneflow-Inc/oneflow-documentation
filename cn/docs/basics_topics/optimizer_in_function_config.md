# 配置优化算法和超参

当一个神经网络被搭建好之后，通常是需要经过训练之后才能够被拿来做预测/推理。而训练的过程就是网络模型参数(Variable)被优化的过程，通常采用反向传播算法和指定的优化器/优化策略(Optimizer)更新网络模型参数，本文重点介绍在 OneFlow 中如何设置 **优化策略(Optimizer)** 和 **超参(Hyperparameters)** 。

可以在不了解 OneFlow 设计和概念的情况下，直接采用下面的预测配置或训练配置；如果有更进一步的需要，可以参考：

文章主要内容如下：

-  **OneFlow的全局函数和配置** 

   介绍 OneFlow 的全局函数Global Function以及设计概念

-  **配置示例** 

   训练作业和推理/预测作业下的配置示例；

-  **Optimizer和优化算法**

   介绍OneFlow中常用的优化器/优化算法

-  **学习率和超参数**

   介绍学习率的设定，学习率衰减策略，一些超参数设定



## OneFlow的全局函数和配置

此章节将介绍 OneFlow 全局函数的概念，函数配置的概念以及如何在函数配置中区分训练或预测/推理配置。

### 全局函数(Global Function)

在介绍优化策略和超参的设置之前，需要先提到 `OneFlow Global Function` 这个概念，被       `oneflow.global_function` 修饰的函数就是 `OneFlow Global Function` ，通常也可以被称作 `job function` 作业函数，下面就是一个简单的例子：

```python
import oneflow as flow
@flow.global_function(type="test", function_config = flow.function_config())
def test_job():
  # build up NN here
```

其中 `@flow.global_function` 有两个基本参数： `type` 指定了作业的类型， `type = "train"` 为训练；`type="predict"` 为验证或推理。`type` 默认为"predict"，`function_config` 默认为None。

`test_job`被`@flow.global_function`修饰后就成为了能被OneFlow识别的作业。

换句话说：在 OneFlow 中，无论是训练还是验证、预测/推理的作业，都需要通过装饰器 `@flow.global_function` 来指定，之后，OneFlow将根据 `function_config` 参数指定的配置(或默认配置)运行此作业。通过这种方式，做到了 **参数配置和任务的分离** 。

`job function `  作业函数主体包括两部分信息：使用算子搭建神经网络(NN)，以及运行这个网络需要的配置信息(function_config)。

下文专注介绍如何设置优化算法等配置信息，网络的搭建请参考[如何使用OneFlow搭建网络](build_nn_with_op_and_layer.md)。

### 函数配置(function_config)

前面的例子中，你也可能注意到 `@flow.global_function` 装饰器接受 `flow.function_config()`  的返回对象作为参数。这个参数就是设置作业函数配置信息的入口。比如下面这个略微复杂一点的例子：

```python
def get_train_config():
  config = flow.function_config()
  # 设置默认数据类型
  config.default_data_type(flow.float)
  # 设置自动混合精度
  config.enable_auto_mixed_precision(True)
  return config

@flow.global_function("train", get_train_config())
def train_job():
  # build up NN here
```

在这个例子中，通过 `function_config` 设置了网络的默认数据类型为 float；并设置了允许自动混合精度训练模型。

还可以通过 `function_config` 进行一些其他的设置，如：`config.default_logical_view(flow.scope.consistent_view())` 设置作业的默认逻辑视图为 `consistent_view`。

`function_config` 相关的设置通常和计算资源、设备、集群调度有关，而具体的优化器、学习率和超参数，我们在 `job function `作业函数主体内设置。



## 配置示例

### 预测/推理配置
下面我们定义了一个用于验证的作业函数(job function)：`eval_job`

我们通过 `get_eval_config()`  定义了 `eval_job()` 的配置，并将 `get_eval_config()` 作为 `@flow.global_function` 的参数，应用到 `eval_job()` 函数。同时，通过设置参数 `type="predict"` 来表明该job function的类型—用于模型验证任务。

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config
  
@flow.global_function("predict", get_eval_config())
def eval_job() -> tp.Numpy:
  # build up NN here
```


### 训练配置
同样，只要按照下面的方式给 `train_job` 函数配上一个装饰器 `@flow.global_function())` 就能够实现一个用于训练的网络。优化器，学习率和超参数的设定，我们可以直接在job function函数主体里定义：

```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss

```
学习率和优化器的设置如下：

1. 利用 `flow.optimizer.PiecewiseConstantScheduler` 设置了学习率 learning rate的策略—初始学习率为0.1的分段缩放策略。当然，你也可以使用其他的学习率策略如：`flow.optimizer.CosineScheduler`(余弦策略)

2. 在 `flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)` 设置了 optimizer 优化器/优化算法。

## Optimizer和优化算法
目前 OneFlow 支持6种 Optimizer /优化算法，分别是：

-  SGD
-  Adam
-  AdamW
-  LazyAdam
-  LARS
-  RMSProp

这六种优化算法必须选择其一，可以通过`flow.optimizer`来调用，譬如：

```
flow.optimizer.SGD(lr_scheduler, momentum=0.9, grad_clipping=flow.optimizer.grad_clipping.by_global_norm(1))
  .minimize(loss)
```

这里不对每个优化器做详细说明，详细请参考[optimizer api](https://oneflow-api.readthedocs.io/en/latest/optimizer.html)

## 学习率和超参数

#### 学习率(learning rate)

学习率相关的设定是通过类：LrScheduler完成的，Optimizer类的构造方法接收一个 LrScheduler 对象来设定学习率。

你可以通过：`lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01])`实现固定大小的学习率；

也可以使用`flow.optimizer.PiecewiseConstantScheduler`设置分段缩放策略的学习率：

```python
lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(0.001, [30, 60, 90], 0.1)
```

你还可以通过`flow.optimizer.CosineSchedule`设定cosine余弦策略的学习率。



例如你希望训练的初始学习率为 0.01，以 cosin策略降低学习率，迭代 10000 个 iteration，并且训练最开始时额外有 100 个 iteration 进行 warmup，这 100 个 iteration 里学习率从 0 逐渐上升到初始学习率

```
lr_scheduler = flow.optimizer.CosineScheduler(10000, 0.01, warmup=flow.optimizer.warmup.linear(100, 0))
```

#### 超参数(Hyperparameter)

除了优化算法和学习率等常用设置，OneFlow中也支持一些其他的优化选项和超参数设定。

像常用的：**clip梯度截取**、**weight decay**、**L2正则化**

和lr_scheduler一样，clip和weight decay都作为optimizer类的参数可以直接设定：

```python
class Optimizer:
    def __init__(
        self,
        lr_scheduler: LrScheduler,
        loss_scale_factor: Optional[int] = None,
        grad_clipping: Optional[ClipGradientConf] = None,
        train_step_lbn: Optional[Text] = None,
    ):
        self.lr_scheduler = lr_scheduler
        self.loss_scale_factor = loss_scale_factor
        self.grad_clipping = grad_clipping
        self.train_step_lbn = train_step_lbn
        ...
```

> 目前，我们只在AdamW类型的优化器中提供weight decay方法，在其余优化器中，可以通过L2正则化实现"weight decay"的效果（通过设置variable的regularizer）

## 总结

一个 OneFlow 的全局函数由 `@oneflow.global_function` 修饰，解耦了网络的搭建过程和任务相关配置(function_config)，`function_config` **采取集中配置的方式，既方便任务切换，又方便集群调度配置。**

job function中，可以通过`flow.optimizer`方便地设置Optimizer优化器、学习率、已经超参数。当然，目前还不够全面，我们会不断完善，以支持更多的优化算法及参数设定。