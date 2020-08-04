# Obtain results from job function

In this article, we will mainly introduce how to obtain the return value from job function in OneFlow which includes:

* How to use synchronous method obtain the return value from job function.

* How to use asynchronous method obtain the return value from job function.

In OneFlow, we usually use the decorator called @flow.global_function to define job function. Thus, this task can be training, evaluation or prediction. By define the return value of job function. We can use both synchronous and asynchronous to get result of function.

## Difference between synchronization and asynchronization

Normally, our trainin process is synonymous which means need wait in line. Now we will demonstrate the concept of synchronous and asynchronous and the advantages of asynchronous in OneFlow by a simple example.

#### Synchronization

During the complete process of one iteration, when the data from some step/iter completed the forward and reverse transmission process and completed the updated of weight parameters and the optimizer. Then start the training process in next step. Whereas before next step, usually we need to wait for CPU prepare the training data. Normally it will come up with some times for data preprocessing and loading.

#### Asynchronization

During the iteration with asynchronous process, it basic means open the multithreaded mode. One step does not need to wait for the previous step to finish, but it can directly process data and loading. When device is not full loading, it can start training directly. When the device is full loading, then it can start preparing work for other step.

From the contrast mentioned above, OneFlow preform more efficient of using the computing resources when using the asynchronous mode. Especially when enormous amount of dataset are apply. **Open asynchronous process could narrow the time of data loading and data preparation and boost training model**.



Next, we will introduce how to obtain result in synchronous and asynchronous task and the coding of call function in asynchronous task. The complete source code will be provide in the end of page.

The important things are：

* When define the job function, tell OneFlow synchronous or asynchronous by return value.

* The data type of return value is select in  `oneflow.typing` (We called it `flow.typing`).

* When call the job functions，there are some difference between synchronous and asynchronous.


## Obtain result in synchronization

When calling the job function, we will get an OneFlow object. The method `get` of this object can get result by synchronous.

For example, we defined the function below:
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

In above code, by using citation in python to tell OneFlow system. The return value is `tp.Numpy` which is the `ndarray` in `numpy`.

```python
loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
```

From the example above, we should notice that:

* When define the job function, the return value of job function (loss) is just a place holder. It is for constructing map. Do not have data in it.

* When define the return value as `flow.typing.Numpy`. It can tell OneFlow that when calling the job function, the real return value is `numpy` object.

* When calling the job function `train_job(images, labels)`, We can directly get the results of job function and it is a `flow.typing.Numpy` corresponding to `numpy` objects.

## Data type of `oneflow.typing` 
`flow.typing` include all the data type can be return by the job function.  `flow.typing.Numpy` we mention before is one of them. The common data type is in the list below:

* `flow.typing.Numpy`：corresponding to `numpy.ndarray`
* `flow.typing.ListNumpy`：corresponding to a `list`  container. Every elements in it is  `numpy.ndarray` object. It have connections with distributed training in OneFlow. We will see the function in [The consistent and mirrored view in distributed training.](../extended_topics/consistent_mirrored.md)
* `flow.typing.Dict`：corresponding to`Dict`，key is `str`，value is`numpy.ndarray`
* `flow.typing.Callback`：corresponding to a callback function. It is use to call job function synchronous. We will introduce below.


## Obtain result in asynchronization

Normally, the efficiency of asynchronization is better than synchronization. The following is introduced how to obtain the result from a job function which is asynchronous training.

Basic steps include:

* Prepare callback function and achieve return the result of logic in  function of processing.

* To achieve job function. We use the citation to specify `flow.typing.Callback` is the return value data type. We can see in following example `Callback` can define the return data type.

* When calling job function, regist callback on step one.

* OneFlow find the suitable time to regist the callback and job function return the result to the callback.

The first three step is done by the user and the final step is done by OneFlow framework.

### Coding of callback function
Prototype of callback function:

```python
def cb_func(result):
    #...
```

The result is the return value of job function. We define the data type is T which is `Numpy`、`ListNumpy` and etc... Them can be different type. We have example in below.

`result` is the return value of job function. It must be same as the registration made by the job function.

For example, in the job function below, the return is loss.

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

Attention`-> tp.Callback[oft.Numpy]` means this job function. It return a `tp.Numpy` object and need use asynchronous calling.

Thus, the callback function we defined need receive a `Numpy` data：
```python
def cb_print_loss(result: tp.Numpy):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1
```

Another example, if the job function define as below:

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


The`-> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]`represent the job function. Return a  `tuple` which have two elements. Every element is `tp.Numpy` and need use asynchronous calling.

Thus，the parameters in callback function should be：

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
`arguments` corresponding to the data type of the above jon function.

### Registration of callback function
When call the job function, return a `Callback` object. We send the callback function to it. That is register.

OneFlow will automatically register callback function when obtain the return value.

```python
callbacker = train_job(images, labels)
callbacker(cb_print_loss)
```
But the above code is redundant, we recommend:

```python
train_job(images, labels)(cb_print_loss)
```

## The relevant code

### Synchronous obtain a result
In this example, use a simple Multilayer perceptron(mlp), use synchronization to obtain the only return value `loss` and print the average of loss in each 20 iterations.

Name：[synchronize_single_job.py](../code/basics_topics/synchronize_single_job.py)

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

### Synchronous obtain multiple results
In this example, the return object of job function is a `list `. We can use synchronization to obtain the elements like `labels` and `logits` in the `list`. And evaluate the model we trained before then print the accuracy.

Name：[synchronize_batch_job.py](../code/basics_topics/synchronize_batch_job.py)

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

The model have already trained can be downloaded in: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/lenet_models_1.zip)

### Asynchronous obtain a result

In this example, using mlp training,  obtain the only return value `loss` by asynchronous way and print the average of loss in each 20 times of iterations.

Name：[async_single_job.py](../code/basics_topics/async_single_job.py)

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

The model have already trained can be downloaded in: [mlp_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

### Asynchronous obtain multiple results

In this example, the return object of job function is a `dict `. We can use asynchronization to obtain multiple return objects in `dict`. And evaluate the model we trained before then print the accuracy.

Name：[async_batch_job.py](../code/basics_topics/async_batch_job.py)

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


