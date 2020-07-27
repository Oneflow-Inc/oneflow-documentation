# 作业函数的定义与调用

在 OneFlow 中，我们将训练、预测、推理等任务封装在一个函数中，统称为作业函数(job function)，作业函数联系用户的业务逻辑与 OneFlow 管理的计算资源。

在 OneFlow 中，被 `@oneflow.global_function` 装饰器所修饰的 python 函数，就成为了 OneFlow 作业函数。

我们主要在作业函数中定义网络模型的结构、选择优化指标；此外，还可以将训练有关的超参及环境配置当做参数传递给作业函数(如:下面例子中的：`get_train_config()`)，OneFlow 会根据设置为我们管理内存、GPU等硬件资源。

本文中我们将具体学习：

* 如何定义和调用作业函数

* 如何获取作业函数的返回值

## 作业函数与 OneFlow 运行流程的关系
作业函数分为定义和调用两个阶段。

这与 OneFlow 本身的运行机制有关，简化地说，OneFlow Python 层接口，只是在描述网络模型和训练环境的配置信息，这些信息将传递给底层的 C++ 代码，经过编译、构图等得到计算图，最终交给 OneFlow 运行时，由 OneFlow 运行时(runtime)执行。

作业函数的定义，其实是在做 Python 层的描述网络模型和训练环境的配置工作，在这个阶段，并没有实际的数据，而只能通过规定网络节点的形状、数据类型等信息，起到 **数据占位符** 的作用，方便 OneFlow 的编译构图过程进行模型推理。

作业函数的调用，发生在 OneFlow runtime 已经启动后，我们可以通过调用作业函数，向其传递真实的数据，并获取返回结果。

以下将具体介绍作业函数的定义与调用方法。

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

