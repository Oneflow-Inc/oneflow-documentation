# 获取运行时数据

OneFlow 提供了 `oneflow.watch` 与 `oneflow.watch_diff` 接口，我们可以通过他们注册回调函数，以方便在作业函数运行过程中获取张量数据或梯度。

## 使用流程

想要获取作业函数运行时的数据或者梯度，其基本流程如下：

* 编写回调函数，回调函数的参数需要用注解方式表明监控的数据类型，回调函数内部逻辑由用户自己实现

* 在定义作业函数时，通过 `oneflow.watch` 或 `oneflow.watch_diff` 注册回调函数，前者获取张量数据本身，后者获取对应的梯度

* 在作业函数运行时，OneFlow 框架会在适当的时机，调用之前注册的回调，将监控的数据传递给回调函数，并执行回调函数中的逻辑

以 `oneflow.watch` 为例，以下伪代码展示了使用过程：

```python
def my_watch(x: T):
    #处理x

@global_function()
def foo() -> T:
    #定义网络等 ...
    oneflow.watch(x, my_watch)
    #...
```

以上的 `T` 即 `oneflow.typing` 中的数据类型，如 `oneflow.typing.Numpy`，具体可以参考 [此文](../basics_topics/async_get.md#oneflowtyping)。

以下我们将用实际例子展示 `watch` 与 `watch_diff` 的使用方法

## `watch` 使用例子

下面是一段完整的例子，用于展示如何使用 OneFlow 的 `oneflow.watch` 功能获取网络中间层的数据。

### 代码
代码：[test_watch.py](../code/extended_topics/test_watch.py)

运行该程序：
```shell
python3 test_watch.py
```

能够得到类似下面的输出：
```
in: [ 0.15727027  0.45887455  0.10939325  0.66666406 -0.62354755]
out: [0.15727027 0.45887455 0.10939325 0.66666406 0.        ]
```

### 代码解读
在例子中，我们关注的是 `ReluJob` 里面的 `y`，所以调用 `flow.watch(y, watch_handler)`去监控 `y`。`oneflow.watch` 需要两个参数:

* 第一个参数就是我们关注的对象 `y`；

* 第二个参数是一个回调函数，OneFlow 在调用设备资源执行 `ReluJob` 的时候会将 `y` 的计算结果作为参数传递给这个回调函数。而我们定义的回调函数 `watch_handler` 的逻辑函数，是将得到的参数打印出来。

用户通过自定义回调函数，在回调函数中按照自己的需求处理 OneFlow 运行时从设备中拿到的数据。

## `watch_diff` 使用例子
下面是一段完整的例子，用于展示如何使用 OneFlow 的 `oneflow.watch_diff` 功能获取网络中间层的梯度。

### 代码
代码：[test_watch_diff.py](../code/extended_topics/test_watch_diff.py)

运行该程序：
```shell
python3 test_watch_diff.py
```

能够得到类似下面的输出：
```
[ ...
 [ 1.39966095e-03  3.49164731e-03  3.31605263e-02  4.50417027e-03
   7.73609674e-04  4.89911772e-02  2.47627571e-02  7.65468649e-05
  -1.18361652e-01  1.20161276e-03]] (100, 10) float32
```

### 代码解读
以上通过 `oneflow.watch_diff` 获取梯度的例子，其流程与 通过 `oneflow.watch` 获取张量数据的例子是类似的。

首先，定义了回调函数：
```python
def watch_diff_handler(blob: tp.Numpy):
    print("watch_diff_handler:", blob, blob.shape, blob.dtype)
```

然后，在作业函数中使用 `oneflow.watch_diff` 注册以上的回调函数：
```python
flow.watch_diff(logits, watch_diff_handler)
```

在 OneFlow 运行时， OneFlow 框架就会调用 `watch_diff_handler`，并且将以上的 `logits` 对应的梯度传递给 `watch_diff_handler`。
