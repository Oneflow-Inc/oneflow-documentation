# 如何从网络中获取中间层的数据
[TOC]

OneFlow提供了`watch`, `watch_diff`和`watch_scope`三种接口，帮助获取网络运行过程中指定中间层的数据。

## `watch` - 获取指定中间层输出的数据
### `test_watch.py`
下面是一段完整的例子，用于展示如何使用OneFlow的`watch`功能获取网络中间层的数据。
```python
# test_watch.py
import numpy as np
import oneflow as flow

def cb(y):
    print("out", y.ndarray())

@flow.global_function()
def ReluJob(x=flow.FixedTensorDef((10,))):
    y = flow.nn.relu(x)
    flow.watch(y, cb)

flow.config.gpu_device_num(1)
data = np.random.uniform(-1, 1, 10).astype(np.float32)
print("in: ", data)
ReluJob(data)
```
### 运行`test_watch.py`
您可以用python的方式运行上面[这段代码](test_watch.py)，比如
```
python3 test_watch.py
```
然后能够得到类似下面的输出：
```
in:  [-0.33018136  0.27108115  0.08992404  0.5222855  -0.507921    0.32096234
 -0.3682254   0.9071676   0.81585795  0.36498776]
out [0.         0.27108115 0.08992404 0.5222855  0.         0.32096234
 0.         0.9071676  0.81585795 0.36498776]
```

### 代码分析
在例子中，我们关注的是`ReluJob`里面的`y`，所以调用`flow.watch`去关注`y`。`flow.watch`需要两个参数:
- 第一个参数就是我们关注的那个对象`y`；
- 第二个参数`cb`，是一个回调函数，OneFlow在调用设备资源执行`ReluJob`的时候会将`y`的计算结果作为输入传递给`cb`。`cb`函数会把输入`y`，转换成ndarray对象打印出来。

用户需要自定义回调函数，在回调函数中按照自己的需求处理OneFlow从设备中拿到的数据。

## `watch_diff` - 获取中间层的梯度
### `test_watch_diff.py`
下面是一段完整的例子，用于展示如何使用OneFlow的`watch_diff`功能获取网络中间层的梯度。
```
import numpy as np
import oneflow as flow

def get_cb(bn):
    def cb(x):
        blob = x.ndarray()
        print(bn, blob.shape, blob.dtype)
        # print(blob)
    return cb

def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config

@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((8, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((8,), dtype=flow.int32)):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    flow.losses.add_loss(loss)

    flow.watch(logits, get_cb("logits"))
    flow.watch_diff(logits, get_cb("logits_grad"))

    return loss

flow.train.CheckPoint().init()
images = np.random.uniform(-10, 10, (8, 1, 28, 28)).astype(np.float32)
labels = np.random.randint(-10, 10, (8,)).astype(np.int32)
loss = train_job(images, labels).get().mean()
```
### 运行`test_watch_diff.py`
您可以用python的方式运行上面[这段代码](test_watch_diff.py)，比如:
```
python3 test_watch_diff.py
```
然后能够得到类似下面的输出：
```
logits (8, 10) float32
logits_grad (8, 10) float32
```
### 代码分析
上面的例子比前面的例子略微复杂一点，目的是构造一个能够训练的网络，其中：
- `flow.watch(logits, get_cb("logits"))`用来获得`logits`的值
- `flow.watch_diff(logits, get_cb("logits_grad"))`用来获得`logits`梯度的值。

`flow.watch_diff`也是需要两个输入：
- 第一参数就是我们需要关注的那个Tensor，例子中就是`logits`
- 第二个参数也是一个回调函数，需用户自定义处理数据的方式。

这个例子中演示了一个略微复杂一点的回调函数：
```
def get_cb(bn):
    def cb(x):
        blob = x.ndarray()
        print(bn, blob.shape, blob.dtype)
        # print(blob)
    return cb
```
通过调用`get_cb`能够得到`watch`所需的回调函数。这么做的目的，是为了给回调函数提供更多信息，比如这个例子中，通过`get_cb`的参数`bn`，`cb`函数得以得到并使用这个`bn`。

## `watch_scope` - 获取作用域内所有tensor的值或其梯度值
### `test_watch_scope.py`
下面是一段完整的例子，用于展示如何通过作用域`watch_scope`的方式获取网络中间层的数据和梯度。
```python
import numpy as np
import oneflow as flow

def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config

tensor_watched = {}
tensor_grad_watched = {}

@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((8, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((8,), dtype=flow.int32)):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    with flow.watch_scope(tensor_watched, tensor_grad_watched):
        logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    flow.losses.add_loss(loss)

    return loss

flow.train.CheckPoint().init()
images = np.random.uniform(-10, 10, (8, 1, 28, 28)).astype(np.float32)
labels = np.random.randint(-10, 10, (8,)).astype(np.int32)
loss = train_job(images, labels).get().mean()

print("view watched tensors")
for lbn, tensor_data in tensor_watched.items():
    print(lbn)
    #print(tensor_data["blob"].ndarray())
    #print(tensor_data["blob_def"])

print("view watched grad tensors")
for lbn, tensor_data in tensor_grad_watched.items():
    print(lbn)
    #print(tensor_data["blob"].ndarray())
    #print(tensor_data["blob_def"])
```
### 运行`test_watch_scope.py`
您可以用python的方式运行上面[这段代码](test_watch_scope.py)，比如
```
python3 test_watch_scope.py
```
然后能够得到类似下面的输出：
```
view watched blobs
Dense_6-bias/out
Dense_6-weight/out
Dense_6_matmul/out_0
Dense_6_bias_add/out_0
view watched grad blobs
Dense_6_matmul/out_0
Dense_6_bias_add/out_0
Dense_6-bias/out
Dense_6-weight/out
```
该例子输出中只是打印了，watch作用域里面所有tensor及其梯度在OneFlow运行时的logical blob name（lbn）。
### 代码分析
例子中首先定义了两个空的字典：
```
tensor_watched = {}
tensor_grad_watched = {}
```
将分别用于保存watch_scope中的tensor和grad。

然后定义观察区`with flow.watch_scope(tensor_watched, tensor_grad_watched):`，把`tensor_watched`和`tensor_grad_watched`作为参数传入，在观察区所有的tensor和grad都将被写入这个字典，key是logical blob name（lbn），value包含了这个lbn的定义信息（`blob_def`)和blob对象（`blob`)。

`train_job`被执行一次之后，通过遍历`tensor_watched`和`tensor_grad_watched`查看其中的内容。

另外如果不关注`tensor_grad_watched`，可以在`watch_scope`中只传入一个参数即可。

