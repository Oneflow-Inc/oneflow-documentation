# 如何从网络中获取中间层的数据

OneFlow 提供了 `oneflow.watch` 与 `oneflow.watch_diff` 接口，我们可以通过他们注册回调函数，以方便在作业函数运行过程中获取张量数据或梯度。

## 使用流程

想要获取作业函数运行时的数据或者梯度，其基本流程如下：

* 编写回调函数，回调函数的参数需要用注解方式表明数据类型，回调函数内部逻辑由用户自己实现

* 在定义作业函数时，通过 `oneflow.watch` 或 `oneflow.watch_diff`注册回调函数，前者获取张量数据本身，后者获取对应的梯度

* 在作业函数运行时，OneFlow 框架会在适当的时机，调用之前注册的回调，执行回调函数中的逻辑

以 `oneflow.watch` 为例，以下伪代码展示了使用过程：

```python
def MyWatch(x: T):
    #处理x

@global_function()
def foo() -> T:
    #定义网络等 ...
    oneflow.watch(x, MyWatch)
    #...
```

以上的 T 即 `oneflow.typing` 中的数据类型，如 `oneflow.typing.Numpy`，具体可以参考 [作业函数的定义与调用](job_function_define_call.md) 一文。

以下我们将用实际例子展示 `watch` 与 `watch_diff` 的使用方法

## 使用 `watch` 获取运行时中间层的数据

下面是一段完整的例子，用于展示如何使用 OneFlow 的 `oneflow.watch` 功能获取网络中间层的数据。
```python
#test_watch.py
import numpy as np
import oneflow as flow
import oneflow.typing as oft

def watch_handler(y:oft.Numpy):
    print("out:", y)

@flow.global_function()
def ReluJob(x:oft.Numpy.Placeholder((5,))) -> None:
    y = flow.nn.relu(x)
    flow.watch(y, watch_handler)

flow.config.gpu_device_num(1)
data = np.random.uniform(-1, 1, 5).astype(np.float32)
print("in:", data)
ReluJob(data)
```

运行[以上代码](../code/extended_topics/test_watch.py)：
```
python3 test_watch.py
```

能够得到类似下面的输出：
```
in: [ 0.15727027  0.45887455  0.10939325  0.66666406 -0.62354755]
out: [0.15727027 0.45887455 0.10939325 0.66666406 0.        ]
```

### 代码解析
在例子中，我们关注的是 `ReluJob` 里面的 `y`，所以调用 `flow.watch(y, watch_handler)`去监控 `y`。`oneflow.watch` 需要两个参数:

* 第一个参数就是我们关注的对象 `y`；

* 第二个参数是一个回调函数，OneFlow 在调用设备资源执行 `ReluJob` 的时候会将 `y` 的计算结果作为参数传递给这个回调函数。而我们定义的回调函数 `watch_handler` 的逻辑函数，是将得到的参数打印出来。

用户通过自定义回调函数，在回调函数中按照自己的需求处理 OneFlow 运行时从设备中拿到的数据。

## 使用 `watch_diff` 获取运行时的梯度
### `test_watch_diff.py`
下面是一段完整的例子，用于展示如何使用OneFlow的`oneflow.watch_diff`功能获取网络中间层的梯度。
```python
# test_watch_diff.py
import oneflow as flow
import oneflow.typing as oft

BATCH_SIZE = 100

def watch_diff_handler(blob: oft.Numpy):
    print("watch_diff_handler:", blob, blob.shape, blob.dtype)

def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="train", function_config=get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> oft.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
        logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    flow.watch_diff(logits, watch_diff_handler)
    return loss


if __name__ == '__main__':
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0: print(loss.mean())
```

运行[以上代码](../code/extended_topics/test_watch_diff.py)：
```
python3 test_watch_diff.py
```
然后能够得到类似下面的输出：
```
[ ...
 [ 1.39966095e-03  3.49164731e-03  3.31605263e-02  4.50417027e-03
   7.73609674e-04  4.89911772e-02  2.47627571e-02  7.65468649e-05
  -1.18361652e-01  1.20161276e-03]] (100, 10) float32
```
### 代码解析
以上通过 `oneflow.watch_diff` 获取梯度的例子，其流程与 通过 `oneflow.watch` 获取张量数据的例子是类似的。

首先，定义了回调函数：
```python
def watch_diff_handler(blob: oft.Numpy):
    print("watch_diff_handler:", blob, blob.shape, blob.dtype)
```

然后，在作业函数中使用 `oneflow.watch_diff` 注册以上的回调函数：
```python
flow.watch_diff(logits, watch_diff_handler)
```

在 OneFlow 运行时， OneFlow 框架就会调用 `watch_diff_handler`，并且将以上的 `logits` 对应的梯度传递给 `watch_diff_handler`。
