# Configuration of Optimization Algorithms and Hyperparameters 

After a neural network model has been set up, it usually requires training before using for prediction or inference. The training process means to optimize parameters of the nerwork which are usually updated with the back propagation algorithm and a specified optimizer. In this article, we will introduce how to setup **optimizers** and **hyperparameters** in OneFlow to users.


Key point summary of this article:

- Configuration examples of job functions for training and prediction.

- The use of optimizer and learning strategies.

- Common errors due to misconfiguration and corresponding solutions.

Users can directly use the training and inferencing configurations described in **Example of configutraion** section without knowing the design concept of OneFlow. For more detials please refer to [optimizer api](https://oneflow-api.readthedocs.io/en/latest/optimizer.html)

## Job Function Configuration

In [Recognizing MNIST Handwritten Digits] (... /quick_start/lenet_mnist.md#global_function), we have learned about the concept of the `oneflow.global_function` decorator and the job function. The configuration of this article base on that.

The job function can be configured by passing the `function_config` parameter to the decorator.

If you are not familiar with `oneflow.global_function`, please refer to [Recognizing MNIST Handwritten Digits ](../quick_start/lenet_mnist.md#global_function) and [Job Function Definitions and Calls](../extended_topics/job_function_define_call.md).



## Example of Configurations

### Configuration for prediction/inference

Here we define a job function to evaluate the model: `eval_job`

We set up the configurations of `eval_job()` in `get_eval_config` fucntion and pass it to `@flow.global_function`. At the same time, we set the `type` parameter of the `@flow.global_function` to "predict" for evaluation task. This way, OneFlow does not propagate backwards in this job function.


```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config

@flow.global_function(type="predict", get_eval_config())
def eval_job() -> tp.Numpy:
  # build up neural network here
```

### Configuration for training

If you specify the `type` parameter of `@flow.global_function` to be `train`, you can get a job function for training.

In the following code, `train_job` is the job function used for training and it is configured with the default `function_config` (so no parameter is passed to `function_config`).

Because OneFlow will back propagates for `train` functions. Thus you need to specify the following settings like optimizer, learning rate and other hyperparameters in the job function.

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
In above code:

1. PiecewiseConstantScheduler` sets the learning rate (0.1) and the learning strategy (PiecewiseConstantScheduler, a segment scaling strategy). There are other learning strategies built inside OneFlow. Such as: [CosineScheduler](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.CosineScheduler)、[CustomScheduler](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.CustomScheduler)、[InverseTimeScheduler](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.InverseTimeScheduler) and etc.

2. In  `flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)`, set the optimizer to SGD and specify the optimization target as `loss`. OneFlow contains multiple optimizers such as: [SGD](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.SGD)、[Adam](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.Adam)、[AdamW](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.AdamW)、[LazyAdam](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.LazyAdam)、[LARS](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.LARS)、[RMSProp](https://oneflow.readthedocs.io/en/master/optimizer.html#oneflow.optimizer.RMSProp). More information please refer to API documentation.

## FAQ

- Error `Check failed: job().job_conf().train_conf().has_model_update_conf()`

> If the `type` of the job function is `"train"`, but `optimizer` and optimization target are not configured. OneFlow will report an error during back propagation because OneFlow does not know how to update the parameters. 
Solution: Configure `optimizer` for the job function and specify the optimization target.

- Error `Check failed: NeedBackwardOp`

> If the `type` of the job function is `"predict"` but `optimizer` is incorrectly configured. Then `optimizer` cannot get the reversed data because OneFlow does not generate a reversed map for the `predict` job function. 
Solution: Remove the `optimizer` statement from the `predict` function.
