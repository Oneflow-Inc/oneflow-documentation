# The Definition and Call of Job Function

In OneFlow, We can encapsulate the train, predict, inference and some other tasks into a function, which is called job function. The job function is used to connect the user's business logic and the computing resource managed by OneFlow

In OneFlow, the function decorated by @oneflow.global_function decorator is the OneFlow's job function

We mainly define the structure of the model and choose the optimization in job function.Otherwise, we can also pass some hyperparameters about training and the environment configuration to the job function(like the following example: get_train_config()), OneFlow will manage the memory, GPU and some other computing resource according to our config.

In this section, we will specifically learn about:

* how to define and call the job function

* how to get the return value of job function

## The relationship between the job function and the running process of OneFlow
The Job function is divided into two phases: definition and call.

It's related to OneFlow's operating mechanism.Briefly, The OneFlow Python layer Api simply describes the configuration and the training environment of the model.These information will pass to the C++ backend.After compilation, composition and so on, the calculation diagram is obtained.Finally, it will be executed by OneFlow runtime.

The definition of the job function, is actually doing the description of network model and the configuration of training environment in Python. In this phase, there's no data here, we can only define the shape, data type of the model's node, we call it as placeholder, which is convenient to model inference in the compilation of OneFlow.

The job function will be called after the OneFlow runtime has started. We can pass the data by calling job function and get the results

The definition and calling method of job functions are described in detail as below

## 作业函数的定义
我们将模型封装在 Python 中，再使用`oneflow.global_function`修饰符进行修饰。就完成了作业函数的定义。

作业函数主要描述两方面的事情：

* 模型结构

* 训练过程中的优化目标

以下代码示例中，我们构建了一个 mlp 模型。并且将由 `flow.nn.sparse_softmax_cross_entropy_with_logits` 计算得到交叉熵损失结果作为优化目标。

```python
@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> oft.Numpy:
    # mlp
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss
```

### 数据占位符
注意，以上的 `images`、`logtis`、`labels`、`loss`等对象，在我们定义作业函数时，并没有实际的数据。它们的作用只是 **描述数据的形状和属性** ，起到 **占位符** 的作用。

在作业函数的参数中的数据占位符，使用`oneflow.typing`下的`Numpy.Placeholder`、`ListNumpy.Placeholder`、`ListListNumpy.Placeholder`，注解作业函数参数的类型，对应作业函数调用时，传递 `numpy` 数据对象。

除了`oneflow.typing`下的几种类型外，不出现在参数中，而由 OneFlow 的算子或层产生的变量，如以上代码中的`reshape`、`hidden`、`logits`、`loss`等，也都起到了数据占位符的作用。

不管是以上提及的哪种变量，它们都直接或间接继承自 OneFlow 的 `BlobDef`基类，OneFlow 中把这种对象类型统称为 **Blob**。

**Blob** 在作业函数定义时，均无真实数据，均只起到数据占位方便框架推理的作用。

### 作业函数的返回值
之所以在上文中强调数据占位符 **Blob** 的概念，是因为作业函数的返回值是不能任意指定的，必须是 `Blob` 类型的对象，或者仅存有 `Blob` 对象的容器。

如以上代码的中所返回的 `loss`，它就是 `Blob` 类型。

作业函数的返回值，需要通过注解声明，比如以上代码中的 `-> oft.Numpy`，表示返回1个 `Blob`。

再比如，可以通过注解声明返回值类型为 `-> Tuple[oft.Numpy, oft.Numpy]`，表示返回1个 `tuple`，该 `tuple` 中有2个 `Blob` 对象。

具体的使用例子，可以参考[获取作业函数的结果](../basics_topics/async_get.md)

## 作业函数的调用
OneFlow 利用函数修饰符将普通 Python 函数转变为 OneFlow 特有的作业函数的过程，对于用户而言是无感、透明的。

我们可以像调用普通的 Python 函数一样调用作业函数。每一次调用，OneFlow 都会在框架内部完成正向传播、反向传播、参数更新等一系列事情。

以下代码，获取数据之后，会向 `train_job` 作业函数传递参数并调用，打印平均损失值。

```python
(train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
for i, (images, labels) in enumerate(zip(train_images, train_labels)):
    loss = train_job(images, labels)
    if i % 20 == 0:
        print(loss.mean())
```

可以看到，通过调用调用作业函数 `train_job` 直接返回了 `numpy` 数据。

以上展示的调用方式是同步方式， OneFlow 还支持异步调用，具体可以参阅专题[获取作业函数的结果](../basics_topics/async_get.md)。

