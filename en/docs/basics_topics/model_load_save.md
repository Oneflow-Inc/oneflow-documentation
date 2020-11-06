# Loading and Saving of Model

For loading and saving for model, the common scences is:

* Save the model that has been trained for a while to facilitate the next training.

* Save trained model for reproduction(Such as Model Serving). 

Strictly speaking, we save the untrained model as `checkpoint` or `snapshot`. It is different from `model saving` of a completed model.

However, no matter the model has been trained or not, we can use the same **interface** to save model. Thus, like the `model`、`checkpoint`、`snapshot` we see in other framework is no difference in OneFlow. We use `flow.train.CheckPoint` as the interface.

In this article, we will introduce:

* How to create model parameters

* How to save and load model

* Storage structure of OneFlow model

* How to finetune and extend model

## Use get_variable to Create/Obtain Model Parameters Object

We can use [oneflow.get_variable](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.get_variable) to create or obtain an object and this object can be used to interact with information in global job functions. When we call the interfaces of `OneFlow.CheckPoint`, this object will also be stored automatically or recovered from storage devices.

Because of this feature, the object created by `get_variable` is used to store model parameters. In fact, there are many high level interface in OneFlow (like `oneflow.layers.conv2d`) use `get_variable` internally to create model parameters internally.

### Process of get_variable Create/Obtain Object

The `get_variable`  requires a specified `name` as the identity of the created object. 

If the `name` value already existed in the program, then get_variable will get the existed object and return.

If the `name` value doesn't exist in the program, `get_variable` will create a blob object internally and return.

### Use get_variable Create Object

The prototype of `oneflow.get_variable` is:

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

The following example use `get_variable` to create parameters and build the network with `oneflow.layers.conv2d`:

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
```

### Initializer Setting

In the previous sections, when we call `get_variable`, we specify the method of initializing the parameters by `initializer`. In OneFlow, we provide many initializers which can be found in `oneflow/python/ops/initializer_util.py`

After we set the `initializer`, the initialization is done by OneFlow framework. The specific time is: when user call the `CheckPoint.init`, OneFlow will **initialize all the data** created by get_variable according to `initializer`.

Some commonly used `initializer`:


* [constant_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.constant_initializer)

* [zeros_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.zeros_initializer)

* [ones_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.ones_initializer)

* [random_uniform_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.random_uniform_initializer)

* [random_normal_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.random_normal_initializer)

* [truncated_normal_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.truncated_normal_initializer)

* [glorot_uniform_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.glorot_uniform_initializer)

* [glorot_normal_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.glorot_normal_initializer)

* [variance_scaling_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.variance_scaling_initializer)

* [kaiming_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.kaiming_initializer)

* [xavier_normal_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.xavier_normal_initializer)

* [xavier_uniform_initializer](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.xavier_uniform_initializer)





## The Python Interface of OneFlow Models

We get the CheckPoint object by instantiating `oneflow.train.CheckPoint()`. There are three critical methods in `CheckPoint` class:

* `init` : Initialize parameters according to the default initialization method.

* `save` : Responsible for saving the current model to the specified path.

* `load` : Import the model parameters from `path` and use them to initialize parameters.

The prototype of `init` is as follows. Before training, we need to use `init` to initialize the parameters in network.

```python
def init(self)
```

The prototype of `save` is as follows. It can save the model under the specified `path`.
```python
def save(self, path)
```

The prototype of `load` is as follows. You can load previously saved models specified by the `path`. 
```python
def load(self, path)
```

### Initialize Model
Before training, we need get the object of  `CheckPoint` and call the `init` to initialize the parameters in network.

For example:

```python
check_point = flow.train.CheckPoint() #constructing object of CheckPoint
check_point.init() #initialize network parameters 

#... call job function etc.
```

### Save Model

At any stage of training process, we can save the model by calling the `CheckPoint` object's `save` method.
```python
check_point.save('./path_to_save')
```
Attention:

* The folder specified by path parameter must be empty, otherwise `save` will raise an error.

* OneFlow model can be saved in a certain form to the specified path. For more details please refer to the example below.

* Although OneFlow do not have limitation of `save` frequency, however, if the storage frequency is too high, it will increase the burden of resources such as disk and bandwidth.

### Load Model
We can call the `CheckPoint` object's `load` method to load model from specified path. 

Here is an example of constructing `CheckPoint` object and loading model from a specified path:
```python
check_point = flow.train.CheckPoint() #constructing object 
check_point.load("./path_to_model") #load model
```

## The Structure of OneFlow Model Saving
OneFlow model are the **parameters** of network. For now there are no meta graph information in OneFlow model. The path to save model have many sub-directories. Each of them is corresponding to the `name` of `job function` in model. For example, we define the model in the first place:

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
Assume that in the process of training, we call the following code to save model:
```python
check_point = flow.train.CheckPoint()
check_point.save('./lenet_models_name') 
```
Then `lenet_models_name` and the subdirectories are as follows:
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

* In the network in job function, each variable is corresponding to a sub-directory.

* All sub-directories above have an `out` document. It stores the parameters of network in binary form. `out` is the default file name. We can change that by `variable op` in the network.

* `Snapshot_done` is an empty folder. If it exists, it means that the network training has been finished. 

* Snapshots of the training steps is stored in `System-Train-TrainStep-train_job`.

## Model Finetune and Transfer Learning

In model finetune and transfer learning, we always need：

- Load some of the parameters from original model
- Initialize the other part of parameters in model 

For this, the following procedures in `flow.train.CheckPoint.load` are as follow:

* According to the model defined in job function, traverse the saved path of the model and try to load the parameters.

* If the corresponding parameter is found, the parameter is loaded.

* If it is not found, it will be automatically initialized, and print the warning to remind that some of the parameters have been automatically initialized. 

In the [BERT](../adv_examples/bert.md) of OneFlow Benchmark, we can see the example of finetune.

Here is a simplified example. 

First we need define a model and save it to `./mlp_models_1` after training:
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="dense1",
        )
        dense2 = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="dense2"
        )

        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, dense2)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

    return loss
```
Then we expand the network and add one more layer (`dense3`) in above model:
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        #... original structure

        dense3 = flow.layers.dense(
            dense2, 10, kernel_initializer=initializer, name="dense3"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, dense3)

    #...
```
Finally, load parameters from original model and start training:
```python
if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.load("./mlp_models_1")

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
    check_point.save("./mlp_ext_models_1")
```

We will get the output:
```text
WARNING! CANNOT find variable path in : ./mlp_models_1/dense3-bias/out. It will be initialized. 
WARNING! CANNOT find variable path in : ./mlp_models_1/dense3-weight/out. It will be initialized. 
2.8365176
0.38763675
0.24882479
0.17603233
...
```
It means all parameters need by `dense3` layer are not found in the original model and initialization starts automatically.
### Codes

The following code is from [mlp_mnist_origin.py](../code/basics_topics/mlp_mnist_origin.py). As the backbone network. Trained model is stored in `./mlp_models_1`.

Run:


```shell
wget https://docs.oneflow.org/code/basics_topics/mlp_mnist_origin.py
python3 mlp_mnist_origin.py
```

When the training is complete, you will get the `mlp_models_1` in the current working directory.

The following code is from [mlp_mnist_finetune.py](../code/basics_topics/mlp_mnist_finetune.py). After finetuning (add one more layer `dense3` in backbone network), we load  `./mlp_models_1` and train it.

Run: 


```shell
wget https://docs.oneflow.org/code/basics_topics/mlp_mnist_finetune.py
python3 mlp_mnist_finetune.py
```

The finely tuned models are saved in `. /mlp_ext_models_1`.
