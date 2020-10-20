# 获取作业函数的结果

本文主要介绍如何在 OneFlow 中获取作业函数的返回结果，主要包括：

* 如何同步方式获取作业函数的结果

* 如何异步方式获取作业函数的结果

在 OneFlow 中，被 `@flow.global_function` 装饰器修饰的函数定义称为作业函数，作业函数可以用于训练或预测。通过指定作业函数的返回值类型，可以使用同步或者异步的方式获取作业函数的运算结果。

## 同步/异步对比

**同步**

在同步训练中，只有上一个 step 的工作完成后，才能开始下一个 step 的训练。

**异步**

在异步训练中，作业函数的执行是并发的，某个 step 不必等上一个 step 的作业结束，而是可以提前进行数据加载和预处理。

通过以上对比可知，在 OneFlow 中使用异步执行作业函数，有效利用了计算机资源，尤其是在数据集规模巨大的情况下，**开启异步执行能有效缩短数据的加载和准备时间，加快模型训练**。

接下来，我们将分别讲解如何用同步、异步的方式获取作业函数的计算结果，并在文章的最后提供完整代码的链接。

它们的要点在于：

* 定义作业函数时，通过注解返回值类型，告之 OneFlow 是同步还是异步模式

* 作业函数的返回值类型在 `oneflow.typing` 模块中选择

* 调用作业函数时，同步/异步调用作业函数的形式略有不同

## 同步获取结果

定义作业函数时，通过注解指定作业函数的返回结果为 `oneflow.typing.Numpy` 时，作业函数为一个同步作业函数。

比如，如果我们定义了如下的作业函数：
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

以上代码，通过 python 注解的方式告之 OneFlow 系统，返回的是 `tp.Numpy` （ `tp` 是 脚本中 `oneflow.typing` 的别名）类型，它对应了 `NumPy` 中的 `ndarray`。

那么，当我们调用作业函数时，作业函数会直接返回 `ndarray` 对象：

```python
loss = train_job(images, labels)
if i % 20 == 0:
    print(loss.mean())
```

从以上示例中，应该注意到：

* 定义作业函数时，作业函数返回的对象(上文中的 `loss`) 只是数据占位符，用于构建计算图，并没有真实数据。

* 通过指定作业函数的返回值类型为 `oneflow.typing.Numpy`，可以告之 OneFlow 此作业函数调用时，返回的真实数据类型为 `NumPy ndarray` 对象

* 通过调用作业函数 `train_job(images, labels)` 可以直接获取作业函数的运行计算结果，类型为 `oneflow.typing.Numpy` 对应的 `ndarray` 对象。

## `oneflow.typing` 中的数据类型
`oneflow.typing` 中包含了作业函数可以返回的数据类型，上文中出现的 `flow.typing.Numpy` 只是其中一种，现将其中常用的几种类型及对应意义罗列如下：

- `oneflow.typing.Numpy`：对应了 `numpy.ndarray`，本文主要以 `oneflow.typing.Numpy` 举例
- `oneflow.typing.ListNumpy`：对应了一个 `list` 容器，其中每个元素都是一个 `numpy.ndarray` 对象。与 OneFlow 进行分布式训练的视角有关，将在[分布式训练的consistent与mirrored视角](../extended_topics/consistent_mirrored.md)中看到其作用
- `oneflow.typing.ListListNumpy`：对应了一个 `list` 容器，其中每个元素都是一个 `TensorList` 对象，OneFlow 的某些接口需要处理或者返回多个 `TensorList` 对象。具体可以参阅 [概念清单](./concept_explanation.md#3tensorbuffer-tensorlist) 及相关 [API 文档](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=ListListNumpy)
- `oneflow.typing.Callback`：对应了一个回调函数，用于异步调用作业函数，下文会介绍

此外，OneFlow 还允许作业函数以字典的形式传出数据，有关 `ListNumpy`、`ListNumpy`、`ListListNumpy` 以及如何用字典方式传出数据的示例，可以参考 [OneFlow 的测试案例](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/python/test/ops/test_global_function_signature.py)。

## 异步获取结果

一般而言，采用异步方式获取训练结果的效率高于同步方式。
以下介绍如何异步调用作业函数并处理训练结果。

其基本步骤包括：

* 准备回调函数：需要通过注解的方式指定回调函数所接受的参数，在回调函数的内部，实现处理作业函数返回值结果的逻辑

* 实现作业函数：通过注解的方式，指定 `flow.typing.Callback` 为作业函数的返回类型。我们将在下文例子中看到，我们通过 `Callback` 可以指定回调函数的参数类型

* 调用作业函数：并注册以上第一步准备的回调函数

以上工作三个步骤由 OneFlow 的用户完成，在程序运行时，注册的回调函数会被 OneFlow 调用，并将作业函数的返回值作为参数传递给回调函数。

### 编写回调函数
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
注解`-> tp.Callback[tp.Numpy]` 表示此作业函数，返回一个 `tp.Numpy` 类型的对象，并且需要异步调用。

那么，我们定义的回调函数，就应该接受一个 `Numpy` 类型的参数：
```python
def cb_print_loss(result: tp.Numpy):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1
```

类似的，如果作业函数的定义为：
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

其中 `-> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]` 表示此作业函数，返回一个包含2个元素的 `tuple`，且每个元素都是 `tp.Numpy` 类型，并且作业函数需要异步调用。

那么，对应的回调函数的参数注解应该为：
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
`arguments` 对应了以上作业函数的返回类型。


### 注册回调函数
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


## 完整代码

### 同步获取一个结果
在本例中，使用一个 LeNet 网络，通过同步方式获取唯一的返回结果 `loss` ，并每隔20轮打印一次 `loss`。

代码链接：[synchronize_single_job.py](../code/basics_topics/synchronize_single_job.py)

运行：
```shell
wget https://docs.oneflow.org/code/basics_topics/synchronize_single_job.py
python3 synchronize_single_job.py
```

会有类似输出：

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



### 同步获取多个返回结果

在本例中，作业函数返回一个 `tuple` ，我们通过同步方式获取 `tuple` 中 `labels` 与 `logits` ，并对上例中训练好的模型进行评估，输出准确率。

代码链接：[synchronize_batch_job.py](../code/basics_topics/synchronize_batch_job.py)

其中，预训练模型文件可以点此处下载：[lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/lenet_models_1.zip)

运行：
```shell
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip
unzip lenet_models_1.zip
wget https://docs.oneflow.org/code/basics_topics/synchronize_batch_job.py
python3 synchronize_batch_job.py
```

会有输出：
```text
accuracy: 99.3%
```

### 异步获取一个返回结果

在本例中，使用 mlp 训练，通过异步方式获取唯一的返回结果 `loss` ，并每隔20轮打印一次 `loss`。

代码下载：[async_single_job.py](../code/basics_topics/async_single_job.py)

运行：
```shell
wget https://docs.oneflow.org/code/basics_topics/async_single_job.py
python3 async_single_job.py
```

会有类似输出：

```text
File mnist.npz already exist, path: ./mnist.npz
3.0865736
0.8949808
0.47858357
0.3486296
...
```



### 异步获取多个返回结果

在以下的例子中，我们展示了如何异步方式获取作业函数的多个返回结果，并对上例中训练好的模型进行评估，输出准确率。

代码下载：[async_batch_job.py](../code/basics_topics/async_batch_job.py)

其中，预训练模型文件可以点此处下载：[mlp_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

```shell
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip
unzip mlp_models_1.zip
wget https://docs.oneflow.org/code/basics_topics/async_batch_job.py
python3 async_batch_job.py
```

输出：

```text
File mnist.npz already exist, path: ./mnist.npz
accuracy: 97.6%
```
