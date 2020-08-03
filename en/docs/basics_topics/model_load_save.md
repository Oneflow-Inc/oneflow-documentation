# Loading and saving of model

For loading and saving for model, the common scenario have:

* Save the model which have been training for a while and make it ready for next train.

* Save the model have completed training and make it ready for deployment.

Strictly speaking, to save a incompleted model is called save `checkpoint` or `snapshot`.It is different with `model saving` of a completed model.

But, no matter is the model have completed the training process, we can use **unified port**. Thus, like the `model`、`checkpoint`、`snapshot` we saw in other framework is no difference in OneFlow framework.In OneFlow, we all use `flow.train.CheckPoint` as port controls.

In this article, we will introduce:

* How to create model parameters

* How to save and load model

* Storage structure of OneFlow model

* Part of the initialization technique of the model

## Use get_variable to create/access object of model parameters

We can use `oneflow.get_variable` to create or obtain an object. This object could used for submitting information with global job function. When calling the port of `OneFlow.CheckPoint`. This object also will be store automatically or recover from storage devices.

Because of these characters, the object create by `get_variable` always used in store model parameters.In fact, there are many high levels ports in OneFlow like `oneflow.layers.conv2d`. We use `get_variable` to create model parameters.

### Process of get_variable get/create object

`Get_variable`  need a specified `name`. This parameter will be the name when create a object.

If the` name` value is existing in the program, then get_variable will get the existing object and return.

If the` name` value is not existing in the program, then get_variable will create a blob object and return.

### Use get_variable create object

The prototype of `oneflow.get_variable`:

```python
def get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
)
```

Next code is an example of in `oneflow.layers.conv2d`, use get_variable to create parameters and keep forward build network:

```python
    #...
    weight = flow.get_variable(
        weight_name if weight_name else name_prefix + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=kernel_initializer
        if kernel_initializer is not None
        else flow.constant_initializer(0),
        regularizer=kernel_regularizer,
        trainable=trainable,
        model_name="weight",
    )

    output = flow.nn.conv2d(
        inputs, weight, strides, padding, data_format, dilation_rate, groups=groups, name=name
    )
    #...

    output = flow.nn.conv2d(
        inputs, weight, strides, padding, data_format, dilation_rate, groups=groups, name=name
    )
    #...
```

### Initializer setting

In the previous chapters, when we calling `get_variable`, we specified the method of initiating the parameters by `initializer`. In OneFlow, we provide many initializers which can be find in `oneflow/python/ops/initializer_util.py`

After config `initializer`, the initialize work is done by OneFlow framework. Exactly time was: when user called the `CheckPoint.init` later on, OneFlow will initialize all data created by get_variable according to `initializer`.

Some common `initializer`:

* constant_initializer

* zeros_initializer

* ones_initializer

* random_uniform_initializer

* random_normal_initializer

* truncated_normal_initializer

* glorot_uniform_initializer

* variance_scaling_initializer

* kaiming_initializer




## The python port of OneFlow

We use `oneflow.train.CheckPoint()` to achieve object of CheckPoint. There are three critical methods in `CheckPoint`:

* `init` : According to method of lacking to initialize parameters.

* `save` : Responsible for save the current model to the specified path.

* `load` : Import the model from `path` and use the model to initialize parameters.

The `init`  work like this. Before you training, we need use  `init` to initialize the parameters in net work.

```python
def init(self)
```

The `save` work like this. It could save the model under a specified  `path`.
```python
def save(self, path)
```

The `load` work like this. Can load the model we train perviously from the specified  `path`.
```python
def load(self, path)
```

### Initialize model
Before training, we need get the object of  `CheckPoint` then called the  `init` to initialize the parameters in network. For example:

```python
check_point = flow.train.CheckPoint() #constructing object of CheckPoint
check_point.init() #initialize network parameters 
```

### Save model

At any step of training process, we can called the `save`  which is the obejct of `CheckPoint`  to save model.
```python
check_point.save('./path_to_save')
```
Attention:

* The path to save must be empty otherwise there will be an error in  `save`.

* Although OneFlow do not have limitation of `save` frequency, but more frequent you save model more duty will push to the disk.

* OneFlow model can save in a certain form stored in the specified path. More details in the example below.

### Load model
We can called the `load` which is the obejct of `CheckPoint` to load model from specificed path. Attention, load model from the disk must match in the model with the current task function. Otherwise will have error message.

There is a example of load model from a specific path and construct  `CheckPoint object` :
```python
check_point = flow.train.CheckPoint() #constructing object 
check_point.load("./path_to_model") #load model
```


## The structure of OneFlow saved model
Model of OneFlow are the **parameters** of network. For now there are no Meta Graph information in OneFlow model. The path to saved model have many sub-directories. Each of them corresponding to a `name` of `job function `in model. For example, we define the model in the first place:

```python
def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv1",
        kernel_initializer=initializer,
    )
    pool1 = flow.nn.max_pool2d(
        conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
    )
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(
        conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
    )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")

```
Assume that in the process of training, we called the following code to save model:
```python
check_point = flow.train.CheckPoint()
check_point.save('./lenet_models_name') 
```
Then  `lenet_models_name` and the subdirectories will look like:
```
lenet_models_name
├── conv1-bias
│   └── out
├── conv1-weight
│   └── out
├── conv2-bias
│   └── out
├── conv2-weight
│   └── out
├── hidden-bias
│   └── out
├── hidden-weight
│   └── out
├── outlayer-bias
│   └── out
├── outlayer-weight
│   └── out
├── snapshot_done
└── System-Train-TrainStep-train_job
    └── out
```

We can see:

* The job function in network, each of the variables have a sub-directory.

* All subdirectories above have a  `out` document. It store the parameters of network in binary.`out` is the default file name. We can change that by  `variable op` in the network.

* `snapshot_done` is an empty folder. If it exists, means the network is completed training.

* Snapshots of the training steps is store in `System-Train-TrainStep-train_job`.


## Q&A

For now, OneFlow frame support the basic functions of processing model. But in real time operation may have some problems. We list some of below.

### Initialize the parameters of model
Befor the network training or inference, model need be initialize. Means initialize variable op in the network. Otherwise the parameters of network will probably does not meet expectations.

There are two methods of filling the parameters in network:

* Calling `init` function. This that case all variable op will initialize according to their own initialize method.

* Using `load` function which can read the initialize value from specificed path.

### Part of model initialization and loading
In real time using, we will meet some situations especially when system adjustment or migration study:

* New network is based on a classic network. Expanding some new network structures. The classic model is been trained and it need to be` loaded` when training a new network. And the new expanding of the network needs to be` initialized` according to the specified way.

* The old network has been trained. It need be train again in new optimize the way. New optimize the way will bring some extra variables like `momentum` or `adam`. The old parameters need be load but extra variables need be initialize.

The situations above is:

* Only part of parameters is input by  `load`

* Other parameters are initialize by  `init`

We advise that:

* Save the extended model first.

* Use  `init` to the extended model then save.

* Combined model directory: repalce the path before extending by new path.

* Finally use  `load` to get model and keep training.
