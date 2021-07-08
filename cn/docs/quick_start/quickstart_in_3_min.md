这篇文章将介绍如何快速上手 OneFlow ，我们可以在3分钟内完成一个完整的神经网络训练过程。

## 运行例子
如果已经安装好了 OneFlow ，可以使用以下命令下载[文档仓库](https://github.com/Oneflow-Inc/oneflow-documentation.git)中的[mlp_mnist.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/cn/docs/code/quick_start/mlp_mnist.py)脚本，并运行。

```
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

输出的是一串数字，每个数字代表了训练的损失值，训练的目标是损失值越小越好。到此我们已经用 OneFlow 完成了一个完整的神经网络的训练。

## 代码解读

以下是完整代码，我们将对其关键部分进行解读。
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
        initializer2 = flow.random_uniform_initializer(
            -np.sqrt(1/500.0), np.sqrt(1/500.0))
        logits = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer2, bias_initializer=initializer2, name="dense2"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return loss


if __name__ == "__main__":
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
`train_job` 是一个被 `@flow.global_function` 修饰的函数，通常称为 **作业函数** (job function)。只有作业函数才能够被 OneFlow 识别，进行训练或者预测。通过 type 来指定 job 的类型：`type="train"` 为训练作业；`type="predict"` 为预测作业。

在 OneFlow 中，神经网络的训练或者预测需要两部分信息：

* 一部分是这个神经网络本身的结构和相关参数，这些在上文提到的作业函数里定义；

* 另外一部分是使用什么样的配置去训练这个网络，比如 `learning rate` 、模型优化更新的方法。这些在 job function 里配置如下：
```python
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
```

本文例子中包含了训练一个神经网络的所有元素，除了上面说的作业函数及其配置之外，还有：

- `flow.data.load_mnist(BATCH_SIZE,BATCH_SIZE)`: 准备并加载训练数据；

- ` train_job(images, labels)`: 返回每一次训练的损失值；

- `print(..., loss.mean())`: 每训练20次，打印一次损失值。


以上只是一个简单的示例，在[识别 MNIST 手写体数字](lenet_mnist.md)中，我们对使用 OneFlow 的流程进行了更加全面和具体的介绍。
在 OneFlow [基础专题](../basics_topics/data_input.md)中对于训练中各类问题进行了详细介绍。


我们同时还提供了一些经典网络的[样例代码](https://github.com/Oneflow-Inc/OneFlow-Benchmark)及数据供参考。

## FAQ
- 运行本文脚本时，为什么一直卡着不动？
> 可能是环境变量中设置了错误的代理。可以先通过先运行命令取消代理
```
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```
然后再进行尝试

- 我电脑无法联网，运行脚本时一直卡着不动
> 本文脚本会自动从网络下载需要的数据文件，如果电脑无法联网，则需要点击 [这里](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist.npz) 手工下载，并将它放置在脚本 `mlp_mnist.py` 相同路径下，然后再进行尝试
