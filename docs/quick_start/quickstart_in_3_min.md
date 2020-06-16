这篇文章介绍了如何快速的用OneFlow训练一个神经网络，也许只用3分钟的时间您就能够完成一个完整的神经网络训练过程。

如果您已经安装好了OneFlow，请从[这里](mlp_mnist.py)下载到您自己的机器上，或者把后面的完整代码拷贝到一个python文件中（如mlp_mnist.py）。

然后在文件所在目录运行`python mlp_mnist.py`。

这样您将得到下面的输出：
```
2.7290366
0.81281316
0.50629824
0.35949975
0.35245502
...
```

输出的是一串数字，每个数字代表了每一轮训练后损失值，训练的目标是损失值越小越好。到此您已经用OneFlow完成了一个完整的神经网络的训练。

下面是完整代码。
```
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


@flow.function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.fixed_placement("cpu", "0:0"):
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
后面章节是是对这段代码的简单介绍。

OneFlow相对其他深度学习框架较特殊的地方是这里：
```
@flow.function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
```
`train_job`是一个被`@flow.function`修饰的函数，通常被称作任务函数。只有被`@flow.function`修饰的任务函数才能够被OneFlow识别成一个神经网络训练或者预测任务。

在OneFlow中一个神经网络的训练或者预测任务需要两部分信息，一部分就是这个神经网络本身的结构和相关参数，这是在刚才提到任务函数里定义的；另外一部分就是使用什么样的配置去训练这个网络，`@flow.function(get_train_config())`中的`get_train_config()`定义了这些配置信息，比如这里用的是`naive_conf`作为模型优化更新的方法，也就是通常说的`SGD`。

这段代码里包含了训练一个神经网络的所有元素，除了上面说的任务函数及其配置之外：
- `check_point.init()`是用来初始化网络模型参数的；
- `load_data(BATCH_SIZE)`是准备训练数据的；
- `job(images, labels).get().mean()`则进行一次训练，并返回损失值；
- `if i % 20 == 0: print(loss)`每20次训练打印看看损失值大小。

这里是一个简单网络的示例，还有一篇文档[使用卷积神经网络进行手写体识别](lenet_mnist.md)进行了更加全面和细节的介绍，另外如果您希望了解更多，可以参考OneFlow使用的[基础专题](link)，另外我们还提供了一些经典网络的样例代码及数据供参考。




