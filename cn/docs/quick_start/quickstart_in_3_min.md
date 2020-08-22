这篇文章介绍了如何快速上手 OneFlow ，我们可以在3分钟内完成一个完整的神经网络训练过程。

## 运行一个例子
如果已经安装好了 OneFlow ，可以使用以下命令下载[文档仓库](https://github.com/Oneflow-Inc/oneflow-documentation.git)中的[mlp_mnist.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/cn/docs/code/quick_start/mlp_mnist.py)脚本，并运行。

```shell
wget https://docs.oneflow.org/code/quick_start/mlp_mnist.py #下载脚本
python3 mlp_mnist.py #运行脚本
```

我们将得到类似以下输出：
```
Epoch [1/20], Loss: 2.3155
Epoch [1/20], Loss: 0.7955
Epoch [1/20], Loss: 0.4653
Epoch [1/20], Loss: 0.2064
Epoch [1/20], Loss: 0.2683
Epoch [1/20], Loss: 0.3167
...
```

输出的是一串数字，每个数字代表了训练的损失值，训练的目标是损失值越小越好。到此您已经用 OneFlow 完成了一个完整的神经网络的训练。

## 代码解读
下面是完整代码。
```python
# mlp_mnist.py
import oneflow as flow
import oneflow.typing as tp
import numpy as np

BATCH_SIZE = 100

@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        reshape = flow.reshape(images, [images.shape[0], -1])
        initializer1 = flow.random_uniform_initializer(-1/28.0, 1/28.0)
        hidden = flow.layers.dense(
            reshape,
            500,
            activation=flow.nn.relu,
            kernel_initializer=initializer1,
            bias_initializer=initializer1,
            name="dense1",
        )
        initializer2 = flow.random_uniform_initializer(-np.sqrt(1/500.0), np.sqrt(1/500.0))
        logits = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer2, bias_initializer=initializer2, name="dense2"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, 20, loss.mean()))
```

接下来让我们简单介绍下这段代码。

OneFlow 相对其他深度学习框架较特殊的地方是这里：
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
```
`train_job` 是一个被 `@flow.global_function` 修饰的函数，通常被称为作业函数(job function)。只有被 `@flow.global_function` 修饰的作业函数才能够被 OneFlow 识别，通过type来指定job的类型：type="train"为训练作业；type="predict"为验证或预测作业。

在 OneFlow 中一个神经网络的训练或者预测作业需要两部分信息：

* 一部分是这个神经网络本身的结构和相关参数，这些在上文提到的作业函数里定义；

* 另外一部分是使用什么样的配置去训练这个网络，比如 `learning rate` 、模型优化更新的方法。这些job function里配置如下：
```python
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
```

这段代码里包含了训练一个神经网络的所有元素，除了上面说的作业函数及其配置之外：

- `check_point.init()`: 初始化网络模型参数；

- `flow.data.load_mnist(BATCH_SIZE,BATCH_SIZE)`: 准备并加载训练数据；

- ` train_job(images, labels)`: 返回每一次训练的损失值；

- `print(..., loss.mean())`: 每训练20次，打印一次损失值。



以上只是一个简单网络的示例，在[使用卷积神经网络进行手写体识别](lenet_mnist.md)中，我们对使用OneFlow的流程进行了更加全面和具体的介绍。
另外，还可参考 OneFlow [基础专题](../basics_topics/data_input.md)中对于训练中各类问题的详细介绍。


我们同时还提供了一些经典网络的[样例代码](https://github.com/Oneflow-Inc/OneFlow-Benchmark)及数据供参考。
