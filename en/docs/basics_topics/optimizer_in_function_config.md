# Optimization Algorithm and Parameter Configuration
After a neural network is set up, it usually requires training before it can be used for predict and inference. The process of training is to optimize the network model parameters. Network model's parameters are usually updated using the back propagation algorithm and a specified optimizer. In this article, we will focus on how to set **optimizers** and **hyperparameters** in OneFlow.


The main content in this article:

- The `global_function` decorator - we will introduce the concept and design of global function decorator in OneFlow

- The configuration example - we will show example of configurations of job function for training and inferencing

- The commonly used optimizer and optimization algorithm in OneFlow 

- How to set learning rate and other hyperparameters

We can use training and inferencing configuration in **Example of configutraion** section later without knowing the details of OneFlow. For more detials please refer to [optimizer api](https://oneflow-api.readthedocs.io/en/latest/optimizer.html)


## Job function and its configuration

In this section, we will introduce the concept of Job Function and its configuration as well as how to distinguish training and prediction/inference in job function's configuration.

### Job Function
A function decorated by `oneflow.global_function` is a Job Function. Here is a example:
```python
import oneflow as flow
@flow.global_function(type="test", function_config = flow.function_config())
def test_job():
  # build up NN here
```

There are two parameters in `@flow.global_function`: 
- `type`, whose default value is "predict", specifies the type of job. `type = "train"` for training, `type="predict"` for predicting or inference
- `function_config` needs a `function_config` object containig configuration aboue job function. Its default value is `None` which means taking preset default configuration in OneFlow. 

The function `test_job` above is decorated by `@flow.global_function`, and it can be recognized as a job function by OneFlow. 

In another word, whether it's for training, predicting or inference , the job function should be decorated by `@flow.global_function`. After that, OneFlow will run the job using configuration set by `function_config`. In this way, **configuration and jobs are decoupled.** 

Two things need to be determined in the job function: construction of neural netowrk using operators and layers and the configuration information needed to run the network. 

We will focus on how to set the optimization algorithm and other configuration in the following sections. The network construction please refer to [Build a Neural Network](build_nn_with_op_and_layer.md).

## function_config

In the example above, you may have noticed that the `@flow.global_function` decorator takes the return object of `flow.function_config()` as parameter. This paramter is the entry point for setting job function's configuration. Here is a more complex example：

```python
def get_train_config():
  config = flow.function_config()
  # Default data type
  config.default_data_type(flow.float)
  # Automatic mixing precision
  config.enable_auto_mixed_precision(True)
  return config

@flow.global_function("train", get_train_config())
def train_job():
  # Build up NN here
```

In this example, we set the default data type as float through `function_config`, and the training model is allowed to use automatic mixed precision. 

There are also some other settings the `function_config` can be used for. For example：

Use `config.default_logical_view(flow.scope.consistent_view())` to set the default logical view of the job as `consistent_view`. 

The `oneflow.function_config()` usually sets options related to computing resources, devices, and cluster scheduling. In contrast, we should set the optimizer, learning rate and hyperparameters inner the `job function`. 

## Example of configuration

### Configuration for predicting/inference

Here we define a job function to evaluate the model: `eval_job`

We set up the configurations of `eval_job()` in `get_eval_config` and use `get_eval_config()` as the parameter passed to `@flow.global_function`. At the same time, we specify the type of the job function for model evaluation by setting the parameter of `@flow.global_function` `type="predict"`. 

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config

@flow.global_function(type="predict", get_eval_config())
def eval_job():
  # build up NN here
```
### Configuration for training

Same as above, we decorate the `train_job()` by `@flow.global_function` in following way then we get a job function for training. We can set the optimizer, learning rate and hyperparameters inner the job function. 
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
The way to set up learning rate and optimizer are as follows:

1. We use `flow.optimizer.PiecewiseConstantScheduler` to set the strategy of learning rate, we set it as the piecewise constant scheduler with the initial learning rate = 0.1. You can also use other strategy, like `flow.optimizer.CosineScheduler` and so on. 
2. We choose SGD as the optimizer and take `loss` as the optimization gaol by `flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)`

## The optimizer

So far, OneFlow supports six types of optimizer in `oneflow.optimizer` as follow:
- SGD
- Adam
- AdamW
- LazyAdam
- LARS
- RMSProp

We must choose one when we define a job function for training. For example:
```
flow.optimizer.SGD(lr_scheduler, momentum=0.9, grad_clipping=flow.optimizer.grad_clipping.by_global_norm(1))
  .minimize(loss)
```
We will not explain all optimizer here, for more details please refer to [optimizer api](https://oneflow-api.readthedocs.io/en/latest/optimizer.html).

## The learning rate and hyperparameters

#### Learning rate

Learning rate is set by class `LrScheduler`. The constructor of optimizer class accepts an `LrScheduler` object. 

You can use `lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01])` to set a fixed learning rate.

You can also use `flow.optimizer.PiecewiseScalingScheduler` to set a learning rate for scaling strategy:
```python
lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(0.001, [30, 60, 90], 0.1)
```

In addition, you can set a cosine decayed learning rate by `flow.optimizer.CosineSchedule`.

```python
lr_scheduler = flow.optimizer.CosineScheduler(
    10000, 0.01, warmup=flow.optimizer.warmup.linear(100, 0)
)
```

Code above shows that the initial learning rate is 0.01. Cosine strategy is used to reduce the learning rate and iterate it 10000 times. At the beginning of training, there are 100 additional iterations for warmup. In these 100 iterations, the learning rate gradually increases from 0 to the initial learning rate.

#### Hyperparameter

Besides the common settings such as optimizer and learning rate, there are also other optimization options and hyperparameter settings. 

Such as: **gradient clip**, **weight decay**, **L2 normalization** and so on.

Just like `lr_scheduler`, they can be passed directly to constructor of optimizer class. 

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

> So far we only provide the weight decay in AdamW optimizer. For other optimizers, The effect of "weight decay" can be set by L2 regularization when we set the regularizer of the variable.

## Summary

Job Function is the function decorated by `@oneflow.global_function`. Configurations and jobs are decoupled by that decorator. The `function_config` is used for **centralized configuration** . It is convenient for both jobs switching and cluster scheduling configuration.

In the job function, we can choose the optimizer, set learning rate  and hyperparameters by `flow.optimizer`.
