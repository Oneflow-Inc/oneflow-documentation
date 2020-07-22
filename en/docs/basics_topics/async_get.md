# Obtain results from job function

本文主要介绍如何在 OneFlow 中获取任务函数的返回结果，主要包括：

* How to use synchronously method obtained the return value from job function.

* How to use asynchronous method obtained the return value from job function.

在 OneFlow 中，通常将用 `@flow.global_function` 装饰器修饰的函数定义为任务函数(Job)，此任务可能是训练、验证或预测任务。可以通过 `get()` 方法和 `async_get()` 方法来获取任务函数被执行后的返回对象/结果，`get` 和 `async_get` 则分别对应表示同步和异步获取结果。

## Difference between synchronous and asynchronous

通常，我们训练模型的过程都是同步的，同步即意味着排队，下面我们以一个简单的例子，说明同步和异步的概念，以及在 OneFlow 中异步执行的优势。

#### Synchronous

在一轮完整的迭代过程中，当某个 step/iter 的数据完成了前向和反向传播过程，并且完成了权重参数和优化器参数的更新后，才能开始下一个 step 的训练。而开始下一 step 之前，还往往需要等 cpu 准备好训练数据，这通常又伴随着一定的数据预处理和加载时间。

#### Asynchronous

当在迭代过程中采用异步执行时，相当于开启了多线程模式，某个 step 不必等上一个 step 的任务结束，而是可以提前进行数据预处理和加载过程，当 gpu 资源有空闲时，可以直接开始训练。当 gpu 资源占用满了，则可以开启其它 step 数据的准备工作。

通过以上对比可知，在 OneFlow 中使用异步执行任务，有效利用了计算机资源，尤其是在数据集规模巨大的情况下，**开启异步执行能有效缩短数据的加载和准备时间，加快模型训练**。



Next, we will introduce how to obtain result in synchronous and asynchronous task and the coding of call function in asynchronous task. The complete source code will be provide in the end of page.

## Obtain result in synchronous

调用任务函数，得到一个 OneFlow 对象，该对象的 `get` 方法，可以同步方式结果。

For example, we defined the function below:
```python
@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("cpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss
```

那么，我们可以使用以下代码，通过调用 `get` 方法，获取任务函数所返回的 loss ，并打印平均值。

```python
loss = train_job(images, labels).get()
print(loss.mean())
```

From the example above, we should notice that:

因为 OneFlow 框架的特点，定义任务函数时所 `return` 的对象，在调用任务函数时并 **不是** 直接得到，而需要进一步调用 `get` （及下文介绍的 `async_get` ）方法获取。


## Obtain result in asynchronous

Normally, the efficiency of asynchronous is better than synchronous. 以下介绍如何通过调用任务函数的 `async_get` 方法，异步获取训练结果。

Basic steps include:

* Prepare callback function and achieve return the result of logic in  function of processing.

* 通过 async_get 方法注册回调

* OneFlow 在合适时机调用注册好的回调，并将任务函数的训练结果传递给该回调

以上工作的前两步由 OneFlow 用户完成，最后一步由 OneFlow 框架完成。

### Coding of callback function
Prototype of callback function:

```python
def cb_func(result):
    #...
```

其中的 result ，就是任务函数的返回值

比如，在以下的任务函数中，返回了 loss 。

```python
@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
  #mlp
  #... code not shown
  logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)

  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  flow.losses.add_loss(loss)
  return loss
```

对应的回调函数，简单打印平均的 loss 值：

```python
g_i = 0
def cb_print_loss(result):
  global g_i
  if g_i % 20 == 0:
    print(result.mean())
  g_i+=1
```

Another example, the job function below:

```python
@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.scope.placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  return {"labels":labels, "logits":logits}
```

返回了一个字典，分别存储了 labels 和 logits 两个对象。 We can use the callback function below, handle both of them and calculate the accuracy:

```python
def acc(eval_result):
  global g_total
  global g_correct

  labels = eval_result["labels"]
  logits = eval_result["logits"]

  predictions = np.argmax(logits.ndarray(), 1)
  right_count = np.sum(predictions == labels)
  g_total += labels.shape[0]
  g_correct += right_count
```

### Registration of callback function
调用任务函数，会返回 `blob` 对象，调用该对象的 `async_get` 方法，可以注册我们实现好的回调函数。

```python
train_job(images,labels).async_get(cb_print_loss)
```

OneFlow 会在获取到训练结果时，自动调用注册的回调。


## The relevant code

### Synchronised obtain a result
在本例中，使用一个简单的多层感知机(mlp)训练，通过同步方式获取唯一的返回结果 `loss` ，并每隔20轮打印一次 loss 平均值。

Name：[synchronize_single_job.py](../code/basics_topics/synchronize_single_job.py)

```python
import oneflow as flow
from mnist_util import load_data
import oneflow.typing as oft

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer, name="conv1")
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name="pool1")
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer, name="conv2")
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name="pool2")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="outlayer")


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss


if __name__ == '__main__':

    flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels).get().mean()
            if i % 20 == 0: print(loss)
    check_point.save('./lenet_models_1')  # need remove the existed folder
```

### Synchronised obtain multiple results
在本例中，任务函数返回一个 `list` ，我们通过同步方式获取 `list` 中 `labels` 与 `logits` ，并对上例中训练好的模型进行评估，输出准确率。

Name：[synchronize_batch_job.py](../code/basics_topics/synchronize_batch_job.py)

```python
import numpy as np
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer, name="conv1")
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name="pool1")
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer, name="conv2")
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name="pool2")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="outlayer")


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    return [labels, logits]


g_total = 0
g_correct = 0


def acc(labels, logits):
    global g_total
    global g_correct

    predictions = np.argmax(logits.ndarray(), 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


if __name__ == '__main__':

    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")

    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE, BATCH_SIZE)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            labels, logits = eval_job(images, labels).get()
            acc(labels, logits)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
```

The model have already trained can be downloaded in: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/lenet_models_1.zip)

### Asynchronously obtain a result

在本例中，使用 mlp 训练，通过异步方式获取唯一的返回结果 `loss` ，并每隔20轮打印一次 loss 平均值。

Name：[async_single_job.py](../code/basics_topics/async_single_job.py)

```python
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    # mlp
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    flow.losses.add_loss(loss)
    return loss


g_i = 0


def cb_print_loss(result):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1


def main_train():
    # flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
    for epoch in range(50):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            train_job(images, labels).async_get(cb_print_loss)

    check_point.save('./mlp_models_1')  # need remove the existed folder


if __name__ == '__main__':
    main_train()
```

The model have already trained can be downloaded in: [mlp_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

### Asynchronously obtain multiple results

在以下的例子中，任务函数返回一个 `dict` ，我们展示了如何异步方式获取 `dict` 中的多个返回结果。 And evaluate the model we trained before then print the accuracy.

Name：[async_batch_job.py](../code/basics_topics/async_batch_job.py)

```python
import numpy as np
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer)


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    return {"labels": labels, "logits": logits}


g_total = 0
g_correct = 0


def acc(eval_result):
    global g_total
    global g_correct

    labels = eval_result["labels"]
    logits = eval_result["logits"]

    predictions = np.argmax(logits.ndarray(), 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


def main_eval():
    # flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.load('./mlp_models_1')
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            eval_job(images, labels).async_get(acc)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))


if __name__ == '__main__':
    main_eval()
```
