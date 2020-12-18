# Loading and Saving of Model

For loading and saving for model, the common scences is:

* Save the model that has been trained for a while to facilitate the next training.

* Save trained model for reproduction(Such as Model Serving). 

Strictly speaking, we save the untrained model as `checkpoint` or `snapshot`. It is different from `model saving` of a completed model.

However in Oneflow, no matter the model has been trained or not, we can use the same **interface** to save model. Thus, like the `model`、`checkpoint`、`snapshot` we see in other framework is no difference in OneFlow.

In OneFlow, there are interfaces for module saving and loading under the `flow.checkpoint`.

In this article, we will introduce:

* How to create model parameters

* How to save and load model

* Storage structure of OneFlow model

* How to finetune and extend model

## Use get_variable to Create/Obtain Model Parameters Object

We can use [oneflow.get_variable](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.get_variable) to create or obtain an object and this object can be used to interact with information in global job functions. When we call the interfaces of `oneflow.get_all_variables` and `oneflow.load_variables`, we can get or update the value of the object created by `get_variable`.

Because of this feature, the object created by `get_variable` is used to store model parameters. In fact, there are many high level interface in OneFlow (like `oneflow.layers.conv2d`) use `get_variable` internally to create model parameters internally.

### Process

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

In the previous sections, when we call `get_variable` and specify the method of initializing the parameters by `initializer`. In OneFlow, we provide many initializers which can be found in [oneflow](https://oneflow.readthedocs.io/en/master/oneflow.html).

Under the static graph mechanism, parameter initialization is done automatically by the OneFlow framework after setting the `initializer`.

The `initializers` currently supported by OneFlow are listed below. Click on the links to see the relevant algorithms:


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

We can use the following interfaces to get or update the value of the `variable` object created by `oneflow.get_variable` in job function.

- `oneflow.get_all_variables` : Get the `variable` of all job functions.
- `oneflow.load_variables` : Update the `variable` in  job function.

`oneflow.get_all_variables` returns a dictionary whose key is the `name` specified when creating the `variable` and the value corresponding to the key is a tensor which has `numpy()` converted to a numpy array.

For example, creating an object named `myblob` is in the job function:

```python
@flow.global_function()
def job() -> tp.Numpy:
    ...
    myblob = flow.get_variable("myblob", 
        shape=(3,3), 
        initializer=flow.random_normal_initializer()
        )
    ...
```

If we want to print the value of `myblob`, we can call:

```python
...
for epoch in range(20):
    ...
    job()
    all_variables = flow.get_all_variables()
    print(all_variables["myblob"].numpy())
    ...
```

The `flow.get_all_variables` gets the dictionary and `all_variables["myblob"].numpy()` gets the `myblob` object then converts it to a numpy array.

Instead of `get_all_variables`, we can use `oneflow.load_variables` to update the values of varialbe.

The prototype of `oneflow.load_variables` is as follows:

```python
def load_variables(value_dict, ignore_mismatch = True)
```

Before using `load_variables`, we have to prepare a dictionary whose key is the `name` specified when creating `variable` and value is a numpy array. After passing the dictionary to `load_variables`, `load_variables` will find the variable object in the job function based on the key and update the value.

For example:

```python
@flow.global_function(type="predict")
def job() -> tp.Numpy:
    myblob = flow.get_variable("myblob", 
        shape=(3,3), 
        initializer=flow.random_normal_initializer()
        )
    return myblob

myvardict = {"myblob": np.ones((3,3)).astype(np.float32)}
flow.load_variables(myvardict)
print(flow.get_all_variables()["myblob"].numpy())
```

Although we choose the `random_normal_initializer` initialization method, because `flow.load_variables(myvardict)` updates the value of `myblob`. The final output will be:

```text
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
```

### Model Saving and Loading

We can save or load the model by two methods.

- `oneflow.checkpoint.save` : Responsible for saving the current model to the specified path.
- `oneflow.checkpoint.get` :  Import a model from the specified path.

The prototype of `save` is as follows which saves the model to the path specified by `path`.

```python
def save(path, var_dict=None)
```

The optional parameter `var_dict` saves the object specified in `var_dict` to the specified path if it is not `None`.

The prototype of `get` is as follows which loads the previously saved model specified by the `path`.

```python
def get(path)
```

It will return a dictionary that can be updated into the model using the `load_variables`. 

```python
flow.load_variables(flow.checkpoint.get(save_dir))
```

Attention：

- The path specified by the `save` should either not exist or empty. Otherwise `save` will report an error (to prevent overwriting the original saved model)
- OneFlow models are stored in a specified path in a certain structure. See the storage structure of OneFlow models below for more details.
- Although there is no limit to the frequency of `save` in OneFlow. But excessive saving frequency will increase the load on resources such as disk and bandwidth.

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
flow.checkpoint.save('./lenet_models_name')
```

Then `lenet_models_name` and the subdirectories are as follows:

```
lenet_models_name/
├── conv1-bias
│   ├── meta
│   └── out
├── conv1-weight
│   ├── meta
│   └── out
├── conv2-bias
│   ├── meta
│   └── out
├── conv2-weight
│   ├── meta
│   └── out
├── dense1-bias
│   ├── meta
│   └── out
├── dense1-weight
│   ├── meta
│   └── out
├── dense2-bias
│   ├── meta
│   └── out
├── dense2-weight
│   ├── meta
│   └── out
├── snapshot_done
└── System-Train-TrainStep-train_job
    ├── meta
    └── out
```

We can see:

* In the network in job function, each variable is corresponding to a sub-directory.

* In each of the subdirectories, there are `out` and `meta` files where `out` stores the values of the network parameters in binary form and `meta` stores the network structure information in text form.

* `Snapshot_done` is an empty folder. If it exists, it means that the network training has been finished. 

* Snapshots of the training steps is stored in `System-Train-TrainStep-train_job`.

## Model Finetune and Transfer Learning

In model finetune and transfer learning, we always need：

- Load some of the parameters from original model
- Initialize the other part of parameters in model 

We can use `oneflow.load_variables` to do the operation above. Here is a simple example to illustrate the concept.

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

The parameters of new `dense3` layer do not exist in the original model. They are automatically initialized to their values by OneFlow.

### Codes

The following code is from [mlp_mnist_origin.py](../code/basics_topics/mlp_mnist_origin.py). As the backbone network. Trained model is stored in `./mlp_models_1`.

Run:


```shell
wget https://docs.oneflow.org/code/basics_topics/mlp_mnist_origin.py
python3 mlp_mnist_origin.py
```

When the training is complete, you will get the `mlp_models_1` in the current working directory.

The following code is from [mlp_mnist_finetune.py](../code/basics_topics/mlp_mnist_finetune.py). After finetuning (add one more layer `dense3` in backbone network), we load `./mlp_models_1` and train it.

Run: 


```shell
wget https://docs.oneflow.org/code/basics_topics/mlp_mnist_finetune.py
python3 mlp_mnist_finetune.py
```

The finetuned models are saved in `. /mlp_ext_models_1`.
