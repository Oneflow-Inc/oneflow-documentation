# 配置优化算法和超参

当搭建好神经网络模型后，需要经过训练才能用来做预测。而训练的过程就是网络模型中的参数被优化的过程，通常采用反向传播算法和指定的 Optimizer 更新参数，本文重点介绍在 OneFlow 中如何设置 **Optimizer** 和 **超参(Hyperparameters)** 。


文章主要内容如下：

-  用于训练的作业函数和用于预测的作业函数的配置示例；

-  optimizer 及 学习策略的使用；

-  由于错误配置导致的常见错误及解决方法


可以在不了解 OneFlow 设计和概念的情况下，直接采用 **配置示例** 部分的训练或预测配置；更详细的说明请参考[optimizer API 文档](https://oneflow.readthedocs.io/en/master/optimizer.html)


## 作业函数配置的基本概念
在 [识别 MNIST 手写体数字](../quick_start/lenet_mnist.md#global_function) 一文中，我们已经了解了 `oneflow.global_function` 装饰器及作业函数的概念，本文的配置，建立在此基础上。

我们可以通过向该装饰器传递 `function_config` 参数达到配置作业函数的目的。

如果对于 `oneflow.global_function` 还不了解，请先参阅 [识别 MNIST 手写体数字](../quick_start/lenet_mnist.md#global_function) 及 [作业函数的定义与调用](../extended_topics/job_function_define_call.md)。

## 配置示例

### 预测配置
以下代码中我们定义了一个用于预测的作业函数：`eval_job`。

我们通过 `get_eval_config()`  定义了 `eval_job()` 的配置，并将 `get_eval_config()` 作为 `@flow.global_function` 的参数，应用到 `eval_job()` 函数。同时，通过设置参数 `type="predict"` 来表明该作业函数用于预测，这样，OneFlow 不会在这个作业函数中进行反向传播。

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config

@flow.global_function(type="predict", get_eval_config())
def eval_job() -> tp.Numpy:
  # build up neural network here
```


### 训练配置
如果指定 `@flow.global_function` 的 `type` 参数为 `train`，就能够得到一个用于训练的作业函数。

以下代码中，`train_job` 为用于训练的作业函数，采用默认的 `function_config` 配置（因此没有向 `function_config` 传参)。

因为 OneFlow 会为 `train` 类型的作业函数进行反向传播，因此需要在作业函数中指定 optimizer、学习率等超参数的设定：

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

以上代码中：

1. `flow.optimizer.PiecewiseConstantScheduler` 设置了学习率（0.1）及学习策略（PiecewiseConstantScheduler，分段缩放策略），OneFlow 中还内置了其它学习策略，如：[CosineScheduler](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.CosineScheduler)、[CustomScheduler](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.CustomScheduler)、[InverseTimeScheduler](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.InverseTimeScheduler) 等。

2. 在 `flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)` 设置 optimizer 为 SGD，并指定优化目标为 `loss`。OneFlow 中内置了多种 optimizer，它们分别是：[SGD](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.SGD)、[Adam](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.Adam)、[AdamW](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.AdamW)、[LazyAdam](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.LazyAdam)、[LARS](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.LARS)、[RMSProp](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.RMSProp)，可参阅 API 文档获取使用方法及算法细节。


## FAQ

- 报错 `Check failed: job().job_conf().train_conf().has_model_update_conf()`
> 如果作业函数的 `type` 为 `"train"`，但是没有设置 `optimizer` 及优化目标，那么在反向传播时，OneFlow 会因为不知道如何更新参数而报错。解决方法：为训练作业函数配置 `optimizer`并指定优化目标。

- 报错 `Check failed: NeedBackwardOp`
> 如果作业函数的 `type` 为 `"predict"`，却（错误地）配置了 `optimizer` 时，因为 OneFlow 不会为 `predict` 类型的作业函数生成反向图，所以此时 `optimizer` 无法拿到反向的数据。解决方法：去掉 `predict` 类型的作业函数中的 `optimizer` 相关语句。
