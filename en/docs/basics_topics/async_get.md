# 获取作业函数的结果

本文主要介绍如何在 OneFlow 中获取作业函数的返回结果，主要包括：

* 如何同步方式获取作业函数的结果

* 如何异步方式获取作业函数的结果

在 OneFlow 中，通常将用 `@flow.global_function` 装饰器修饰的函数定义为作业函数(Job)，此任务可能是训练、验证或预测任务。通过指定作业函数的返回值类型，可以使用同步或者异步的方式获取作业函数的运算结果。

## Difference between synchronous and asynchronous

通常，我们训练模型的过程都是同步的，同步即意味着排队，下面我们以一个简单的例子，说明同步和异步的概念，以及在 OneFlow 中异步执行的优势。

#### Synchronous

在一轮完整的迭代过程中，当某个 step/iter 的数据完成了前向和反向传播过程，并且完成了权重参数和优化器参数的更新后，才能开始下一个 step 的训练。而开始下一 step 之前，还往往需要等 cpu 准备好训练数据，这通常又伴随着一定的数据预处理和加载时间。

#### Asynchronous

当在迭代过程中采用异步执行时，相当于开启了多线程模式，某个 step 不必等上一个 step 的作业结束，而是可以提前进行数据预处理和加载过程，当 gpu 资源有空闲时，可以直接开始训练。当 gpu 资源占用满了，则可以开启其它 step 数据的准备工作。

通过以上对比可知，在 OneFlow 中使用异步执行作业函数，有效利用了计算机资源，尤其是在数据集规模巨大的情况下，**开启异步执行能有效缩短数据的加载和准备时间，加快模型训练**。

接下来，我们将讲解同步、异步作业中的结果的获取，异步作业中回调函数的编写，并在文章的最后提供完整的代码示例。

它们的要点在于：

* 定义作业函数时，通过返回值类型来告之 OneFlow 是同步还是异步模式

* 作业函数的返回值类型在 `oneflow.typing` (下文简称为`flow.typing`)中选择

* 调用作业函数时，同步/异步调用作业函数的形式略有不同

## Obtain result in synchronous

定义作业函数时，通过注解指定作业函数的返回结果为 `oneflow.typing.Numpy` 时，作业函数为一个同步作业函数。

比如，如果我们定义了如下的作业函数：
```python
@flow.global_function(type="train")
def train_job(images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```

以上代码，通过 python 注解的方式告之 OneFlow 系统，返回的是 `tp.Numpy` 类型，即对应了 `numpy` 中的 `ndarray`。

那么，当我们调用作业函数时，作业函数会直接返回 `ndarray` 对象：

```python
loss = train_job(images, labels)
if i % 20 == 0: print(loss.mean())
```

从以上示例中，应该注意到：

* 定义作业函数时，作业函数返回的对象(上文中的 `loss`) 只是数据占位符，用于构建计算图，并没有真实数据。

* 通过指定作业函数的返回值类型为 `flow.typing.Numpy`，可以告之 OneFlow 此作业函数调用时，返回的真实数据类型为 `numpy` 对象

* 通过调用作业函数 `train_job(images, labels)` 可以直接获取作业函数的运行计算结果，类型为 `flow.typing.Numpy` 对应的 `numpy` 对象。

## `oneflow.typing` 中的数据类型
`flow.typing` 中包含了作业函数可以返回的数据类型，上文中出现的 `flow.typing.Numpy` 只是其中一种，现将其中常用的几种类型及对应意义罗列如下：

* `flow.typing.Numpy`：对应了 `numpy.ndarray`
* `flow.typing.ListNumpy`：对应了一个 `list` 容器，其中每个元素都是一个 `numpy.ndarray` 对象。与 OneFlow 进行分布式训练的视角有关，将在[分布式训练的consistent与mirrored视角](../extended_topics/consistent_mirrored.md)中看到其作用
* `flow.typing.Dict`：对应了`Dict`字典，键为`str`类型，值为`numpy.ndarray`
* `flow.typing.Callback`：对应了一个回调函数，用于异步调用作业函数，下文会介绍


## 异步获取结果

一般而言，采用异步方式获取训练结果的效率高于同步方式。 以下介绍如何异步调用作业函数并处理训练结果。

其基本步骤包括：

* 准备回调函数，需要通过注解的方式指定回调函数所接受的参数，回调函数的内部实现处理作业函数返回值结果的逻辑

* 实现作业函数，通过注解的方式，指定 `flow.typing.Callback` 为作业函数的返回类型。我们将在下文例子中看到，我们通过 `Callback` 可以指定回调函数的参数类型

* 调用作业函数，并注册以上第一步准备的回调函数

* 准备的回调函数，会被 OneFlow 调用，并将定义作业函数的返回值作为传输传递给回调函数

以上工作的前三步由 OneFlow 用户完成，最后一步由 OneFlow 框架完成。

### Coding of callback function
回调函数的原型如下：

```python
def cb_func(result: T):
    #...
```

其中的 `result` ，需要通过注解，指定其类型 `T`，即上文中提到的 `Numpy`、`ListNumpy`等，也可以是它们的复合类型，下文我们将有对应的实例。

参数 `result` 对应了作业函数的返回值，因此必须与作业函数返回值所注解的一致。

比如，我们定义了一个作业函数：
```python
@flow.global_function(type="train")
def train_job(images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Callback[tp.Numpy]:
    # mlp
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```
注解`-> tp.Callback[oft.Numpy]` 表示此作业函数，返回一个 `tp.Numpy` 类型的对象，并且需要异步调用。

那么，我们定义的回调函数，就应该接受一个 `Numpy` 类型的参数：
```python
def cb_print_loss(result:tp.Numpy):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1
```

类似的，如果作业函数的定义为：
```python
@flow.global_function(type="predict")
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> oft.Callback[Tuple[tp.Numpy, tp.Numpy]]:
    with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    return (labels, logits)
```

其中`-> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]`表示此作业函数，返回一个包含2个元素的 `tuple`，且每个元素都是 `tp.Numpy` 类型，并且需要异步调用。

那么，对应的回调函数的参数注解应该为：
```python
g_total = 0
g_correct = 0
def acc(arguments:Tuple[tp.Numpy, tp.Numpy]):
    global g_total
    global g_correct

    labels = arguments[0]
    logits = arguments[1]
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count
```
`arguments` 对应了以上作业函数的返回类型。


### Registration of callback function
当我们异步调用作业函数时，返回一个 `Callback` 对象，我们将准备好的回调函数传递给它，就完成了注册。

OneFlow 会在获取到训练结果时，自动调用注册的回调。
```python
callbacker = train_job(images, labels)
callbacker(cb_print_loss)
```

不过以上的写法比较冗余，推荐使用：

```python
train_job(images, labels)(cb_print_loss)
```


## 相关完整代码

### Synchronised obtain a result
在本例中，使用一个 `lenet` 网络，通过同步方式获取唯一的返回结果 `loss` ，并每隔20轮打印一次 loss 平均值。

代码下载：[synchronize_single_job.py](../code/basics_topics/synchronize_single_job.py)

```python
# lenet_train.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2')
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')


@flow.global_function(type="train")
def train_job(images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == "__main__":
    flow.config.gpu_device_num(1)
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)

    for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0: print(loss.mean())
    check_point.save('./lenet_models_1')  # need remove the existed folder
    print("model saved")
```

输出：

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



### Synchronised obtain multiple results

在本例中，作业函数返回一个 `tuple` ，我们通过同步方式获取 `tuple` 中 `labels` 与 `logits` ，并对上例中训练好的模型进行评估，输出准确率。

代码下载：[synchronize_batch_job.py](../code/basics_topics/synchronize_batch_job.py)

```python
#lenet_eval.py
import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')


@flow.global_function(type="predict")
def eval_job(images:tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> Tuple[tp.Numpy, tp.Numpy]:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

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


if __name__ == '__main__':

    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE, BATCH_SIZE)

    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            labels, logtis = eval_job(images, labels)
            acc(labels, logtis)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
```

其中，预训练模型文件可以点此处下载：[lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/lenet_models_1.zip)

### Asynchronously obtain a result

在本例中，使用 mlp 训练，通过异步方式获取唯一的返回结果 `loss` ，并每隔20轮打印一次 loss 平均值。

代码下载：[async_single_job.py](../code/basics_topics/async_single_job.py)

```python
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


@flow.global_function(type="train")
def train_job(images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Callback[tp.Numpy]:
    # mlp
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
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

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
    for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            train_job(images, labels)(cb_print_loss)

    check_point.save('./mlp_models_1')  # need remove the existed folder


if __name__ == '__main__':
    main()
```

输出：

```shell
File mnist.npz already exist, path: ./mnist.npz
3.0865736
0.8949808
0.47858357
0.3486296
...
```



### Asynchronously obtain multiple results

在以下的例子中，我们展示了如何异步方式获取作业函数的多个返回结果，并对上例中训练好的模型进行评估，输出准确率。

其中，预训练模型文件可以点此处下载：[mlp_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

代码下载：[async_batch_job.py](../code/basics_topics/async_batch_job.py)

```python
import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")


@flow.global_function(type="predict")
def eval_job(images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]:
    with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    return (labels, logits)


g_total = 0
g_correct = 0
def acc(arguments:Tuple[tp.Numpy, tp.Numpy]):
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
    check_point.load('./mlp_models_1')
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            eval_job(images, labels)(acc)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))


if __name__ == '__main__':
    main()
```

输出：

```shell
File mnist.npz already exist, path: ./mnist.npz
accuracy: 97.6%
```

