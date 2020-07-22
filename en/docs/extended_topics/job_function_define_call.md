在 OneFlow 中，我们将训练、预测/推理等任务统称为任务函数(job function)，任务函数联系用户的业务逻辑与 OneFlow 管理的计算资源。

在 OneFlow 中，任何被定义为任务函数的方法体都需要用 `@oneflow.global_function` 修饰，通过此注解，我们不仅能定义任务的模型结构，优化方式等业务逻辑，同时可以将任务运行时所需的配置当做参数传递给任务函数(如:下面例子中的：get_train_config())，使得 OneFlow 能方便地为我们管理内存、GPU等计算资源。

本文中我们将具体学习：

* 如何定义和调用任务函数

* 如何获取任务函数的返回值

## 任务函数的定义与调用
任务函数分为定义和调用两个阶段。
### 任务函数的定义
我们将模型封装在Python中，再使用`oneflow.global_function`修饰符进行修饰。就完成了任务函数的定义。 任务函数主要描述两方面的事情：就完成了任务函数的定义。 任务函数主要描述两方面的事情：

* 模型结构

* 训练过程中的优化目标

以下代码示例中，我们构建了一个 mlp 模型。并且将由 `flow.nn.sparse_softmax_cross_entropy_with_logits` 计算得到交叉熵损失结果作为优化目标。
```python
@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.scope.placement("cpu", "0:0"):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  flow.losses.add_loss(loss)
  return loss
```
注意，以上的logtis、labels、loss等对象，它们的作用只是 **描述模型中的数据对象** ，起到 **占位符** 的作用。在定义函数时并 **没有** 真正的数据。在OneFlow中把这种对象类型统称为`blob`。 任务函数的返回值 **必须是blob对象类型或者包含blob对象的容器** 。在定义函数时并 **没有** 真正的数据。在 OneFlow 中把这种对象类型统称为 `blob`。 任务函数的返回值 **必须是blob对象类型或者包含blob对象的容器** 。

### 任务函数的调用
OneFlow 对任务函数的处理，对于用户而言是无感、透明的，我们可以像调用普通的 Python 函数一样调用任务函数。每一次调用，OneFlow 都会在框架内部完成正向传播、反向传播、参数更新等一系列事情。

以下代码，获取数据之后，会向 `train_job` 任务函数传递参数并调用，打印平均损失值。

```python
  # 简单的共50 epochs的训练任务
  (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
  for epoch in range(50):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels).get()
      if i % 20 == 0: print(loss.mean())
```

要注意，直接调用任务函数 `tran_job` 其实得到的是 OneFlow 的 `blob`对象，需要进一步通过该对象的 `get` 或者 `async_get` 方法，来获取任务函数的返回结果。详情可以参阅专题[获取任务函数的结果](../basics_topics/async_get.md)。

