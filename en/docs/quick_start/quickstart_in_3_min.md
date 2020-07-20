这篇文章介绍了如何快速上手OneFlow，我们可以在3分钟内完成一个完整的神经网络训练过程。

## 运行一个例子
如果已经安装好了OneFlow，可以使用以下命令同步我们的[文档仓库](https://github.com/Oneflow-Inc/oneflow-documentation.git)，并运行其中的[mlp_mnist.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/mlp_mnist.py)脚本。

```shell
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git #同步仓库
cd oneflow-documentation/docs/code/quick_start/ #切换到示例代码路径
```

然后，运行神经网络训练脚本：
```shell
python mlp_mnist.py
```

这样您将得到下面的输出：
```
2.7290366
0.81281316
0.50629824
0.35949975
0.35245502
...
```

输出的是一串数字，每个数字代表了每一轮训练后的损失值，训练的目标是损失值越小越好。到此您已经用OneFlow完成了一个完整的神经网络的训练。

## 代码解读
下面是完整代码。
```python
#mlp_mnist.py
import numpy as np
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
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.scope.placement("cpu", "0:0"):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  flow.losses.add_loss(loss)
  return loss


if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.init()
  (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
  for i, (images, labels) in enumerate(zip(train_images, train_labels)):
    loss = train_job(images, labels).get().mean()
    if i % 20 == 0: print(loss)
```

后面章节是对这段代码的简单介绍。

OneFlow相对其他深度学习框架较特殊的地方是这里：
```
@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
```
`train_job`是一个被`@flow.global_function`修饰的函数，通常被称作任务函数。只有被`@flow.global_function`修饰的任务函数才能够被OneFlow识别成一个神经网络训练或者预测任务。

在OneFlow中一个神经网络的训练或者预测任务需要两部分信息：

* 一部分是这个神经网络本身的结构和相关参数，这些在上文提到的任务函数里定义；

* 另外一部分是使用什么样的配置去训练这个网络，比如`learning rate`、模型优化更新的方法。这些在`@flow.global_function(get_train_config())`中的`get_train_config()`配置。

这段代码里包含了训练一个神经网络的所有元素，除了上面说的任务函数及其配置之外：

- `check_point.init()`: 初始化网络模型参数；

- `load_data(BATCH_SIZE)`: 准备并加载训练数据；

- `train_job(images, labels).get().mean()`: 返回每一次训练的损失值；

- `if i % 20 == 0: print(loss)`: 每训练20次，打印一次损失值。


  

以上只是一个简单网络的示例，在[使用卷积神经网络进行手写体识别](lenet_mnist.md)中，我们对使用OneFlow的流程进行了更加全面和具体的介绍。
另外，还可参考OneFlow[基础专题](../basics_topics/data_input.md)中对于训练中各类问题的详细介绍。


我们同时还提供了一些经典网络的[样例代码](https://github.com/Oneflow-Inc/OneFlow-Benchmark)及数据供参考。




