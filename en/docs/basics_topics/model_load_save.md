# Loading and saving of model

For loading and saving for model, the common scenario have:

* Save the model which have been training for a while and make it ready for next train.

* Save the model have completed training and make it ready for deployment.

Strictly speaking, to save a incompleted model is called `checkpoint` or `snapshot`. It is different with `model saving` of a completed model.

But, no matter is the model have completed the training process, we can use **unified port**. Thus, like the `model`、`checkpoint`、`snapshot` we see in other framework is no difference in OneFlow framework. In OneFlow, we all use `flow.train.CheckPoint` as port controls.

In this article, we will introduce:

* How to create model parameters

* How to save and load model

* Storage structure of OneFlow model

* Part of the initialization technique of the model

## Use get_variable to create/access object of model parameters

We can use `oneflow.get_variable` to create or obtain an object. This object could used for submitting information with global job function. When calling the port of `OneFlow.CheckPoint`. This object also will be store automatically or recover from storage devices.

Because of these characters, the object create by `get_variable` always used in store model parameters.In fact, there are many high levels ports in OneFlow like `oneflow.layers.conv2d`. We use `get_variable` to create model parameters.

### Process of get_variable get/create object

`Get_variable`  need a specified `name`. This parameter will be named when create a object.

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

After config `initializer`, the initialize work is done by OneFlow framework. Exactly time was: when user call the `CheckPoint.init` later on, OneFlow will initialize all data created by get_variable according to `initializer`.

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




## The python port of OneFlow models

We use `oneflow.train.CheckPoint()` to achieve object of CheckPoint. There are three critical methods in `CheckPoint`:

* `init` : According to method of lacking to initialize parameters.

* `save` : Responsible for save the current model to the specified path.

* `load` : Import the model from `path` and use the model to initialize parameters.

The `init`  work like this. Before you training, we need use  `init` to initialize the parameters in network.

```python
def init(self)
```

The `save` work like this. It can save the model under a specified  `path`.
```python
def save(self, path)
```

The `load` work like this. It can load the model we train perviously from the specified  `path`.
```python
def load(self, path)
```

### Initialize model
Before training, we need get the object of  `CheckPoint` then call the  `init` to initialize the parameters in network. For example:

```python
check_point = flow.train.CheckPoint() #constructing object of CheckPoint
check_point.init() #initialize network parameters 
```

### Save model

At any step of training process, we can call the `save`  which is the obejct of `CheckPoint`  to save model.
```python
check_point.save('./path_to_save')
```
Attention:

* The path to save must be empty otherwise there will be an error in  `save`.

* Although OneFlow do not have limitation of `save` frequency, but more frequent you save model more duty will push to the disk and broadband.

* OneFlow model can save in a certain form the specified path. More details in the example below.

### Load model
We can call the `load` which is the obejct of `CheckPoint` to load model from specificed path. Attention, load model from the disk must match the model with the current task function. Otherwise will have error message.

There is a example of load model from a specified path and construct  `CheckPoint object` :
```python
check_point = flow.train.CheckPoint() #constructing object 
check_point.load("./path_to_model") #load model
```


## The structure of OneFlow model saving
Model of OneFlow are the **parameters** of network. For now there are no meta graph information in OneFlow model. The path to save model have many sub-directories. Each of them corresponding to a `name` of `job function `in model. For example, we define the model in the first place:

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

* The job function in network, each of the variable have a sub-directory.

* All subdirectories above have an  `out` document. It store the parameters of network in binary. `Out` is the default file name. We can change that by  `variable op` in the network.

* `Snapshot_done` is an empty folder. If it exists, means the network is completed the training.

* Snapshots of the training steps is store in `System-Train-TrainStep-train_job`.

## Model finetune and transfer learning

We will meet the following scenario when we do the model finetune and transfer learning:

* Finetune: Use a backbone which already completed the training as foundation. Keep training after expand some new network structure. The model of original backbone network need load the parameters we store previously. But the new part of the network need initialization.

* Transfer learning: We will training according to new optimization method base on the original network which have completed the training. The new optimization method bring some extra parameters. Such as `momentum` or `adam`. The original parameters need load the parameters we store previously. But the extra parameters need initialization.


To summarise, the above situation is:

* Part of the parameters in model are load from original model.

* The other part(new) of parameters need initialization.

For this, `flow.train.CheckPoint.load` of OneFlow set the following procedures:

* According the description of network model in job function. The saving path of ergodic model and try to load each parameters.

* If find the corresponding parameter then load it.

* If do not find it then start automatic initialization and print warning at same time indicate already automatically initiated the parameters.

In the [BERT](../adv_examples/bert.md) of OneFlow Benchmark. We can see the example of finetune.

The following is a example to explain the concepts:

First we need define a model in the following shape and save it to `./mlp_models_1`：
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
Then we expand the network and add one more layer(`dense3`) to above model:
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

We will get the following output:
```text
WARNING! CANNOT find variable path in : ./mlp_models_1/dense3-bias/out. It will be initialized. 
WARNING! CANNOT find variable path in : ./mlp_models_1/dense3-weight/out. It will be initialized. 
2.8365176
0.38763675
0.24882479
0.17603233
...
```
Means all parameters need by `dense3` layer did not find in original model and it start initialization.
### Complete code

The following code is from [mlp_mnist_origin.py](../code/basics_topics/mlp_mnist_origin.py). As backbone network. Store the trained model in `./mlp_models_1`。
```python
# mlp_mnist_origin.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


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


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
    check_point.save("./mlp_models_1")
```

The following code is from [mlp_mnist_finetune.py](../code/basics_topics/mlp_mnist_finetune.py). After “finetune”（add one more layer `dense3` in backbone network）. Load  `./mlp_models_1` and keep training.
```python
# mlp_mnist_finetune.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


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

        dense3 = flow.layers.dense(
            dense2, 10, kernel_initializer=initializer, name="dense3"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, dense3)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

    return loss


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
