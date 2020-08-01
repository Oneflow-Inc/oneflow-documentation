# Using of flow.function_config() to config optimization algorithm and hyperparameters
After neural network is set up, It usually requires training before it can be used for predict and inference. The process of training is optimize the parameters(Variable) of network. Network model parameters are usually updated using the back propagation algorithm and a specified Optimizer. In this section, we will focus on how to set **optimizer** and **hyperparameters** in OneFlow.

You can use the following configuration in training or predicting without understanding the design and concept about OneFlow. If there is further need, please refer to：

The main content is as follows：

- The Global Function and configuration in OneFlow

  we will introduce the Global Function and concept of design in OneFlow. 

- The configuration example

  Example of configuration under training assignment and inference/prediction assignment. 

- The optimizer and optimization algorithm

  We will introduce the optimizer and algorithm in OneFlow. 

- The learning rate and hyperparameters

  We will introduce the setting of learning rate, learning rate decay schedule and other hyperparameters. 

## The Global Function and configuration in OneFlow

In this section, we will introduce the concept of Global Function, function configuration and how to distinguish between training or prediction/inference configuration in function configuration. It also known as 'job function', there is a simple example：

```python
import oneflow as flow
@flow.global_function(type="test", function_config = flow.function_config())
def test_job():
  # build up NN here
```

There are two parameters in `@flow.global_function`：`type` assign the type of job, `type = "train"` is training, `type="predict"` is predicting or inference. The default type is "predict", the default of function_config is None. 

`test_job` is decorated by `@flow.global_function`, and it can be recognized by OneFlow. 

In another words, whether it's training, validation or  predicting/inference job in OneFlow, it need to be assigned by the decorator `@flow.global_function`, after that, OneFlow will run the job according to the function_config. In this way, **parameter configuration and task are separated.** 

The body of the `job function` consists of two parts of information: Use operators to build a neural network(NN)，and the configuration information needed to run the network. 

We will focus on how to set the optimization algorithm and other configuration information in the following section. The network construction please refer to [How to use OneFlow to build a network](build_nn_with_op_and_layer.md)

## function_config

In the following example, you may also notice that the `@flow.global_function` decorator takes the return object of `flow.function_config()` as parameter.  This paramter is the entry point for setting the job function's configuration. Here is a more complicated example：

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

In this example, we set the default data type as float through `function_config`, and the training model is allowed to use automatic mixed precision. 

There are also some settings you can make from `function_config`, for example：

`config.default_logical_view(flow.scope.consistent_view())` sets the default logical view of the job as `consistent_view`. 

The settings in `function_config` are usually related to the compute resource, device and cluster scheduling. We will set the optimizer, learning rate and hyperparameters in the body of the `job function`. 

## The example of configuration

### The configuration of predict/inference

Here we define a job function to evaluate the model: `eval_job`

We use `get_eval_config()` to define the configurations of `eval_job()` and use `get_eval_config()` as the parameter of `@flow.global_function` to send to` eval_job()` function. At the same time, we set the parameter `type = "predict"` to indicate that the job function is used for model evaluation tasks. 

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config

@flow.global_function(get_eval_config())
def eval_job():
  # build up NN here
```
### The configuration of training

As same, just following the instructions below and  add a decorator `@flow.global_function()) ` to `train_job()`, we can get a network to train. Otherwise, we can set the optimizer, learning rate and hyperparameters in the body of job function. 
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
The settings of learning rate and optimizer are as follow:

1. We use `flow.optimizer.PiecewiseConstantScheduler` to set the strategy of learning rate, we set it as the Piecewise scheduler with the initial learning rate = 0.1. You can also use other strategy, like: `flow.optimizer.CosineScheduler`. 
2. We set the optimizer in `flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)`

## The optimizer

Now OneFlow supports six types of Optimizer as follow：
- SGD
- Adam
- AdamW
- LazyAdam
- LARS
- RMSProp

We must choose one of these algorithms and call it by `flow.optimizer`, for example：
```
flow.optimizer.SGD(lr_scheduler, momentum=0.9, grad_clipping=flow.optimizer.grad_clipping.by_global_norm(1))
  .minimize(loss)
```
We will not explain all optimizer, more details please refer to [optimizer api](http://183.81.182.202:8000/html/train.html#).

## The learning rate and hyperparameters

#### Learning rate

Learning rate is set by the class：LrScheduler, The Constructor of Optimizer class accepts an LrScheduler object to set the learning rate. 

You can use ``lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01])`` to set a regular learning rate；

You can also use `flow.optimizer.PiecewiseConstantScheduler` to set a piece wise scaling scheduler. 

```python
lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(0.001, [30, 60, 90], 0.1)
```

In addition, you can set a cosine decayed learning rate by `flow.optimizer.CosineSchedule`



For example, you want the initial learning rate of the training to be 0.01, the learning rate decayed with the cosine strategy, with 10000 iterations and there are 100 iterations for warm up at the beginning,  the learning rate increases gradually from 0 to the initial learning rate. 

```python
lr_scheduler = flow.optimizer.CosineScheduler(10000, 0.01, warmup=flow.optimizer.warmup.linear(100, 0))
```

#### Hyperparameter

In addition to common settings such as optimizer and learning rate. OneFlow also supports other optimization options and hyperparameter settings. 

Like : **The gradient clip**, **weight decay**, **L2 Normalization**

like lr_scheduler, both clip and weight decay can be directly set as parameters in Optimizer class. 

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

> Now we only provide the weight decay in AdamW optimizer，In other optimizers，you can use L2 normalization to realize "weight decay"(set the regularizer of the variable)

## Summary

A OneFlow global function is decorated by `@oneflow.global_function`. The network building process and task-related configuration are separated.  `Function_config` **use the  centralized configuration. It is easier for switching task and cluster configuration.**

In job function, we can set the Optimizer, learning rate  and hyperparameters by `flow.optimizer`. It is not comprehensive enough at present, we will continue to imporve it to support more optimizer and parameter settings. 