# 获取任务函数的结果

本文主要介绍如何在OneFlow中获取任务函数的返回结果，主要包括：

* 如何同步方式获取任务函数的结果

* 如何异步方式获取任务函数的结果

在OneFlow中，通常将用@flow.global_function装饰器修饰的函数定义为任务函数(Job)，此任务可能是训练、验证或预测任务。可以通过`get()`方法和`async_get()`方法来获取任务函数被执行后的返回对象/结果，`get`和`async_get`则分别对应表示同步和异步获取结果。

## 同步/异步对比

通常，我们训练模型的过程都是同步的，同步即意味着排队，下面我们以一个简单的例子，说明同步和异步的概念，以及在OneFlow中异步执行的优势。

#### 同步

在一轮完整的迭代过程中，当某个step/iter的数据完成了前向和反向传播过程，并且完成了权重参数和优化器参数的更新后，才能开始下一个step的训练。而开始下一step之前，还往往需要等cpu准备好训练数据，这通常又伴随着一定的数据预处理和加载时间。

#### 异步

当在迭代过程中采用异步执行时，相当于开启了多线程模式，某个step不必等上一个step的任务结束，而是可以提前进行数据预处理和加载过程，当gpu资源有空闲时，可以直接开始训练。当gpu资源占用满了，则可以开启其它step数据的准备工作。

通过以上对比可知，在OneFlow中使用异步执行任务，有效利用了计算机资源，尤其是在数据集规模巨大的情况下，**开启异步执行能有效缩短数据的加载和准备时间，加快模型训练**。



接下来，我们将讲解同步、异步任务中的结果的获取，异步任务中回调函数的编写，并在文章的最后提供完整的代码示例。

## 同步获取结果

调用任务函数，得到一个OneFlow对象，该对象的`get`方法，可以同步方式结果。

比如，如果我们定义了如下的任务函数：
```python
@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("cpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss
```

那么，我们可以使用以下代码，通过调用`get`方法，获取任务函数所返回的loss，并打印平均值。

```python
loss = train_job(images, labels).get()
print(loss.mean())
```

从以上示例中，应该注意到：

因为OneFlow框架的特点，定义任务函数时所`return`的对象，在调用任务函数时并 **不是** 直接得到，而需要进一步调用`get`（及下文介绍的`async_get`）方法获取。


## 异步获取结果

一般而言，采用异步方式获取训练结果的效率高于同步方式。
以下介绍如何通过调用任务函数的`async_get`方法，异步获取训练结果。

其基本步骤包括：

* 准备回调函数，在回调函数中实现处理任务函数的返回结果的逻辑

* 通过async_get方法注册回调

* OneFlow在合适时机调用注册好的回调，并将任务函数的训练结果传递给该回调

以上工作的前两步由OneFlow用户完成，最后一步由OneFlow框架完成。

### 编写回调函数
回调函数的原型如下：

```python
def cb_func(result):
    #...
```

其中的result，就是任务函数的返回值

比如，在以下的任务函数中，返回了loss。

```python
@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  #mlp
  #... code not shown
  logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
  
  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
  
  flow.losses.add_loss(loss)
  return loss
```

对应的回调函数，简单打印平均的loss值：

```python
g_i = 0
def cb_print_loss(result):
  global g_i
  if g_i % 20 == 0:
    print(result.mean())
  g_i+=1
```

再比如，以下的任务函数：

```python
@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.fixed_placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  return {"labels":labels, "logits":logits}
```

返回了一个字典，分别存储了labels和logits两个对象。
我们可以实现以下的回调函数，处理两者，计算准确率：

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

### 注册回调函数
调用任务函数，会返回`blob`对象，调用该对象的`async_get`方法，可以注册我们实现好的回调函数。

```python
train_job(images,labels).async_get(cb_print_loss)
```

OneFlow会在获取到训练结果时，自动调用注册的回调。


## 相关完整代码

### 同步获取一个结果
在本例中，使用一个简单的多层感知机(mlp)训练，通过同步方式获取唯一的返回结果`loss`，并每隔20轮打印一次loss平均值。

代码下载：[synchronize_single_job.py](../code/basics_topics/synchronize_single_job.py)

```python
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


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.fixed_placement("gpu", "0:0"):
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

### 同步获取多个返回结果
在本例中，任务函数返回一个`list`，我们通过同步方式获取`list`中`labels`与`logits`，并对上例中训练好的模型进行评估，输出准确率。

代码下载：[synchronize_batch_job.py](../code/basics_topics/synchronize_batch_job.py)

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
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.fixed_placement("gpu", "0:0"):
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

其中，预训练模型文件可以点此处下载：[lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/lenet_models_1.zip)

### 异步获取一个返回结果

在本例中，使用mlp训练，通过异步方式获取唯一的返回结果`loss`，并每隔20轮打印一次loss平均值。

代码下载：[async_single_job.py](../code/basics_topics/async_single_job.py)

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
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
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

其中，预训练模型文件可以点此处下载：[mlp_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

### 异步获取多个返回结果

在以下的例子中，任务函数返回一个`dict`，我们展示了如何异步方式获取`dict`中的多个返回结果。
并对上例中训练好的模型进行评估，输出准确率。

代码下载：[async_batch_job.py](../code/basics_topics/async_batch_job.py)

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
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.fixed_placement("cpu", "0:0"):
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
