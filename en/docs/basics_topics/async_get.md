# Get results from job function

In this article, we will talk about getting the return value of a job function in OneFlow. It covers:

* How to get the return value from a job function synchronously.

* How to get the return value from a job function asynchronously.

In OneFlow, a function decorated by `@flow.global_function` is called "Job Function". Job function can be implmented for training, evaluation or prediction. By specifing the return type of job function, we can get results from job function both synchronously and asynchronously.

## Difference between synchronous and asynchronous

Usually, the process of training the model is synchronous. We will explain the concepts of synchronization and asynchronization, and the advantages of asynchronous execution in OneFlow.

### Synchronization

During synchronous training, the training of the next step cannot be started until the work of the previous step is completed.

### Asynchronization

In asynchronous training, it is equivalent to turning on multi-threading mode. A step does not have to wait for the completion of the previous step, but can carry out data preprocessing and loading in advance.

Through the comparison above, it can be seen that the use of asynchronous execution job function in OneFlow can effectively utilize computer resources, especially in the case of loading huge data, enabling asynchronous execution can effectively shorten the data loading and preprocessing time, and accelerate the model training.

Next, we will explain how to get the results synchronously and asynchronously from job function, and how to write callback functions in asynchronous jobs. At the end of the article, we will provide a complete example.

The main points are：

* Getting results synchronously or asynchronously is determined by the return value type of job function

* The data type of return value is selected in  `oneflow.typing`

* When we call job function, the form of getting results synchronously / asynchronously is slightly different


## Get result synchronously

When we define a job function, if the annotation of the return type of the job function is `oneflow.typing.Numpy`, the job function will be called synchronously.

For example, when we define the job function below:
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

Through the Python annotation, OneFlow will know that the returned type of the job function is `oneflow.typing.Numpy`, which corresponds to ndarray in numpy.

```python
loss = train_job(images, labels)
if i % 20 == 0:
    print(loss.mean())
```

From the example above, it should be noted that:

* When we define the job function, the return value of job function (loss) is just a placeholder without data. It is for graph construction.

* When we specify the return value type as `flow.typing.Numpy`. OneFlow will know that when the job function is called, the real data type returned is `numpy` object.

* By calling the job function `train_job(images, labels)`, we can get the result from job function directly, and the retnred value is a `numpy` object corresponding to `flow.typing.Numpy`.

## Data types in `oneflow.typing`
The `oneflow.typing` contains all the data types that can be returned by the job function. The `oneflow.typing.Numpy` shown above is only one of them. The commonly used types and their corresponding meanings are listed as follows:

* `flow.typing.Numpy`: corresponding to a `numpy.ndarray`

* `flow.typing.ListNumpy`: corresponding to a `list` container. Every element in it is a `numpy.ndarray` object. It is related to the view of OneFlow for distributed training. We will see details in [The consistent and mirrored view in distributed training.](../extended_topics/consistent_mirrored.md)

* `flow.typing.Dict`: corresponding to a `dict` whose key can be string or number and value will be`numpy.ndarray`

* `flow.typing.Callback`: corresponding to a callback function. It is used to call job function asynchronously. We will introduce it below.


## Get result asynchronously

Generally speaking, the efficiency of asynchronous training is higher than that of synchronous training. The following describes how to call job function asynchronously and process the results.

The basic steps include:

* Prepare a callback function to process the return value from the job function. You need to specify the parameters accepted by the callback function through annotation.

* Implementation of the job function, through the way of annotation, specified `oneflow.typing.Callback` as the return type of the job function. As we will see in the following example, we can specify the parameter type of the callback function through `oneflow.typing.Callback`.

* Call the job function and register the callback function prepared in step 1 above.

* The registered callback function will be called by OneFlow at the appropriate time, and the return value of job function will be passed to the callback function as a parameter.

The first three steps are completed by users of OneFlow, and the last step is completed by OneFlow framework.

### Prepare a callback function
Suppose the prototype of the callback function is:

```python
def cb_func(result:T):
    #...
```

Among them, the `result` needs to be annotated to specify its type `T`, that is, numpy, listnumpy, etc. mentioned above, or their composite type. We will have corresponding examples below.

The `result` of callback function is actually the return value of job function. Therefore, it must be consistent with the annotation of the return value of the job function

For example, when we define a job function below:
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Callback[tp.Numpy]:
    # mlp
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="hidden",
    )
    logits = flow.layers.dense(
        hidden, 10, kernel_initializer=initializer, name="output"
    )

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```

Annotation `-> tp.Callback[tp.Numpy]` means that this job function returns a `oneflow.typing.Numpy` type, and need to be called asynchronously.

Thus, the callback function we defined should accept a numpy parameter:
```python
def cb_print_loss(result: tp.Numpy):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1
```

Let's take a look at another example. If the job function is defined as below:

```python
@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]:
    with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)
```

Annotation `-> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]` means that this job function returns a `tuple` in which there are two elements whose type is `oneflow.typing.Numpy`, and the job function need to be called asynchronously.

Thus, the parameter annotation of the corresponding callback function should be:

```python
g_total = 0
g_correct = 0


def acc(arguments: Tuple[tp.Numpy, tp.Numpy]):
    global g_total
    global g_correct

    labels = arguments[0]
    logits = arguments[1]
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count
```
The `arguments` corresponds to the return type of the above job function.

### Registration of callback function
When we call the job function asynchronously, the job function will return a 'callback' object, and we registry the prepared callback function by passing it to that object.

OneFlow will automatically call the registered callback function when it gets the training results.

```python
callbacker = train_job(images, labels)
callbacker(cb_print_loss)
```
But the above code is redundant, we recommend:

```python
train_job(images, labels)(cb_print_loss)
```

## Code

### Get single result synchronously
We use LeNet as an example here to show how to get the return value `loss` synchronously and print the loss every 20 iterations.

[synchronize_single_job.py](../code/basics_topics/synchronize_single_job.py)

```python
# lenet_train.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


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
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding="SAME", name="pool1")
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding="SAME", name="pool2")
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


if __name__ == "__main__":
    flow.config.gpu_device_num(1)
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )

    for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
    check_point.save("./lenet_models_1")  # need remove the existed folder
    print("model saved")
```


Output：

```shell
File mnist.npz already exist, path: ./mnist.npz
7.3258467
2.1435719
1.1712438
0.7531896
...
...
model saved
```

### Get mutiple results synchronously
In this case, the job function returns a `tuple`. We get the resutls `labels` and `logits` in tuple synchronously, and evaluate the trained model in the above example, then output the accuracy rate.

[synchronize_batch_job.py](../code/basics_topics/synchronize_batch_job.py)

```python
#lenet_eval.py
import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp

BATCH_SIZE = 100


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
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding="SAME", name="pool1")
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding="SAME", name="pool2")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")


@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> Tuple[tp.Numpy, tp.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)


g_total = 0
g_correct = 0


def acc(labels, logtis):
    global g_total
    global g_correct

    predictions = np.argmax(logtis, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


if __name__ == "__main__":

    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )

    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            labels, logtis = eval_job(images, labels)
            acc(labels, logtis)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
```

The trained model can be downloaded from [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/lenet_models_1.zip)

### Get single result asynchronously

In this case, we take MLP as example to get the single return result loss from job function asynchronously, and the loss is printed every 20 rounds.

[async_single_job.py](../code/basics_topics/async_single_job.py)

```python
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Callback[tp.Numpy]:
    # mlp
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="hidden",
    )
    logits = flow.layers.dense(
        hidden, 10, kernel_initializer=initializer, name="output"
    )

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


g_i = 0


def cb_print_loss(result: tp.Numpy):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1


def main():
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )
    for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            train_job(images, labels)(cb_print_loss)

    check_point.save("./mlp_models_1")  # need remove the existed folder


if __name__ == "__main__":
    main()
```

Output：

```shell
File mnist.npz already exist, path: ./mnist.npz
3.0865736
0.8949808
0.47858357
0.3486296
...
```

### Get multiple results asynchronously
In the following example, we show how to get multiple return results of the job function asynchronously, evaluate the trained model in the above example, and output the accuracy rate.

[async_batch_job.py](../code/basics_topics/async_batch_job.py)

The trained model can be downloaded from [mlp_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

```python
import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="hidden",
    )
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")


@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]:
    with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)


g_total = 0
g_correct = 0


def acc(arguments: Tuple[tp.Numpy, tp.Numpy]):
    global g_total
    global g_correct

    labels = arguments[0]
    logits = arguments[1]
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


def main():
    check_point = flow.train.CheckPoint()
    check_point.load("./mlp_models_1")
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            eval_job(images, labels)(acc)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))


if __name__ == "__main__":
    main()
```
Output：

```shell
File mnist.npz already exist, path: ./mnist.npz
accuracy: 97.6%
```
