# Using of flow.function_config() to config optimization algorithm and hyperparameters
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
`naive_conf` does not need extra congregations, thus what is store in the dictionary is `{"naive_conf": {}}`. The key is `"naive_conf"` and value is an empty dictionary `{}`.

If choose other optimizer, then need to config the relevant parameters. For example, the code below config a SGD optimizer which inertia is 0.875. Key is `momentum_conf` and the value is not empty. It is `{'beta': 0.875}`.
```
config.train.model_update_conf({"momentum_conf": {'beta': 0.875}})
```
We will not explain all optimizer, more details reference to [optimizer api](http://183.81.182.202:8000/html/train.html#).
### Other optimizations
The difinition previously have 4 more optional optimisations.
- `learning_rate_decay` - descend learning rate
- `warmup_conf` - preheated learning rate
- `clip_conf` - gradient clip
- `weight_decay_conf` -  weight descend

All these opinions could be selected multiple or no selection. The configuration way is add new key-value is dictionary of python. More details reference to[optimizer api](http://183.81.182.202:8000/html/train.html#). The following examples of two forms for your reference.
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
This chapter will introduce the concept, configuration and how to distinguish between training in function configuration or predict configuration of global function in OneFlow.
### OneFlow Global Function
Before introducing the optimization algorithm and hyperparameters, we need to mention the concept of `OneFlow Global Function` which is been decorated by oneflow.global_function. Normally be called `job function`. A simple example below:
```python
import oneflow as flow
@flow.global_function(flow.function_config())
def test_job():
  # build up NN here
```
`test_job` is an `OneFlow Function`. It can be recognised by OneFlow framework. It can turn function of network into suitable calculation chart according to configuration and put it on server to calculate.

The job function have two sections of information: The networks build by the operator and the information to config this network.Building networks reference to [Use OneFlow build the neural network ](build_nn_with_op_and_layer.md).This article focus on introduce how to config information.

### Function configuration
In the previous example, you may noticed that `@flow.global_function` receive an parameter `flow.function_config()`.This `function_config` is the port of parameters information of `OneFlow Function`.There is a more complex example:

```python
config = flow.function_config()
config.default_data_type(flow.float)
config.train.primary_lr(0.1)
config.train.model_update_conf({"naive_conf": {}})

@flow.global_function(config)
def test_job():
  # build up NN here
```
The example above is use `function_config` to config missing data type as float. It will be use for training and the learning rate is 0.1. It use `naive_conv` optimization algorithm which is `SGD`.

Some other things include in function_config please reference to [function_config API](http://183.81.182.202:8000/html/oneflow.html).

### Train or not
Function_config has a lot of property can be set. We are focuse on `train`.Usually, `train` will not be set. In this situation `job function` only can achieve prediction.Once `train` is setted, it means the `job function` have one more training job. The example below demonstrated set learning rate and optimize algorithms.
```python
config.train.primary_lr(0.1)
config.train.model_update_conf({"naive_conf": {}})
```

## Summary
OneFlow is decorated by `@oneflow.global_function`. Explain how to build the network and the relevant function_config. `Function_config` is **use centralized configuration. It is easier for switching task and cluster configuration.(这里链接可能出错） </p>
