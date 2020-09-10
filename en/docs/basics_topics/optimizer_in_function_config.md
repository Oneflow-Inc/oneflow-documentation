# Configuration of Optimization Algorithms and Hyperparameters 

After a neural network model has been set up, it usually requires training before using for prediction or inference. The training process means to optimize parameters of the nerwork which are usually updated with the back propagation algorithm and a specified optimizer. In this article, we will introduce how to setup **optimizers** and **hyperparameters** in OneFlow to users.


Key point summary of this article:

- The global function configuration in OneFlow - introduces the design concept of the `global_function` decorator

- The configuration example - shows how to configure job function for training and inference

- The commonly used optimizers and optimization algorithms in OneFlow 

- How to setup learning rate and other hyperparameters

Users can directly use the training and inferencing configurations described in **Example of configutraion** section without knowing the design concept of OneFlow. For more detials please refer to [optimizer api](https://oneflow-api.readthedocs.io/en/latest/optimizer.html)

## Job function and its configuration

In this section, we will introduce the concept of Job Function and its configuration as well as how to distinguish training and prediction/inference in job function's configuration.

### Job Function

Functions decorated by `oneflow.global_function` will become Job Functions. Here is a example:

```python
import oneflow as flow
@flow.global_function(type="predict", function_config = flow.function_config())
def test_job():
  # build up NN here
```

The `@flow.global_function` decorator accepts two parameters: 
- `type`, whose default value is "predict", specifies the type of job. `type = "train"` means for training, `type="predict"` means for prediction or inference
- `function_config`, an object contains configurations about the job function. Its default value is `None` which means taking the preset default configurations in OneFlow. 

The `test_job` function above is decorated by `@flow.global_function`, which will be recognized as a job function by OneFlow. 

In another word, job functions must be decorated by `@flow.global_function` no matter it's used for training or inference. Then OneFlow will run the job with configurations set by `function_config`. In this way, we can decouple the **configuration and jobs.** 

Two things need to be determined in the job function: construction of neural netowrk using operators and layers and the configuration information needed to run the network. 

Job function contains two parts of informations: 

- The operators and layers to construct the netowrk
- The needed configuration informations to run the network

We will focus on how to setup the optimization algorithm and other configurations in the following sections. For more details of the network construction please refer to [Build a Neural Network](build_nn_with_op_and_layer.md).

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

There are also some other settings of the `function_config`. For example:

Use `config.default_logical_view(flow.scope.consistent_view())` to set the default logical view of the job as `consistent_view`. 

The `oneflow.function_config()` usually sets options related to computing resources, devices, and cluster scheduling. In contrast, the specific optimizer algorithm, learning rate and others hyperparameters should be set inside the `job function`. 

## Example of configurations

### Configuration for prediction/inference

Here we define a job function to evaluate the model: `eval_job`

We set up the configurations of `eval_job()` in `get_eval_config` fucntion and pass it to `@flow.global_function`. At the same time, we set the `type` parameter of the `@flow.global_function` to "predict" for evaluation task.


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

Same as above, we decorate the `train_job()` function with `@flow.global_function` then we get a job function for training. We can set the optimizer, learning rate and hyperparameters inside the job function. 

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
To set up the optimizer and learning rate, follow the instructions below:

1. We set the strategy of learning rate to `flow.optimizer.PiecewiseConstantScheduler` which means piecewise constant scheduler with the initial learning rate = 0.1. You can also use other strategy, like `flow.optimizer.CosineScheduler` and so on. 

2. We choose SGD as the optimizer and take `loss` as the optimization goal by `flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)`

## The optimizers

So far, OneFlow supports six types of optimizer which can be found in `oneflow.optimizer`:

- SGD
- Adam
- AdamW
- LazyAdam
- LARS
- RMSProp

When defining a job function for training users must choose one of the optimizer algorithm. For example:

```
flow.optimizer.SGD(lr_scheduler, momentum=0.9, grad_clipping=flow.optimizer.grad_clipping.by_global_norm(1))
  .minimize(loss)
```
We will not explain all optimizers here, for more details please refer to [optimizer api](https://oneflow-api.readthedocs.io/en/latest/optimizer.html).

## The learning rate and hyperparameters

#### Learning rate

Users can set the learning rate by class `LrScheduler`. And the constructor of optimizer class accepts an `LrScheduler` object. 

You can use `lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.01])` to set a fixed learning rate.

You can also use `flow.optimizer.PiecewiseScalingScheduler` to set a learning rate for scaling strategy:
```python
lr_scheduler = flow.optimizer.PiecewiseScalingScheduler(0.001, [30, 60, 90], 0.1)
```

In addition, you can also set a cosine decayed learning rate by `flow.optimizer.CosineSchedule`.

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

> So far we only provide the weight decay in AdamW optimizer. For other optimizers, by setting the regularizer of the variable to L2 regularization we can get the same effect of "weight decay".

## Summary

A Job Function is the function decorated by `@oneflow.global_function`. Configurations and jobs are decoupled by that decorator. And the `function_config` is used for **centralized configuration** . It is convenient for both jobs switching and cluster scheduling configuration.

In the job function, we can choose the optimizer, set learning rate and hyperparameters by `flow.optimizer`.
