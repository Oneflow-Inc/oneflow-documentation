This article introduces how to quickly get started with OneFlow. We can complete a full neural network training process just in 3 minutes.

## Example
If you already have one flow installed, you can run the following command to clone our [repository](https://github.com/Oneflow-Inc/oneflow-documentation.git) and run the script called [mlp_mnist.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/mlp_mnist.py).

```shell
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git #clone repository
cd oneflow-documentation/docs/code/quick_start/ #Switch to the sample code path
```

Then run the neural network training script:
```shell
python mlp_mnist.py
```

You will get following output:
```
2.7290366
0.81281316
0.50629824
0.35949975
0.35245502
...
```

The output is a string of number, each number represents the loss value of each round of training. The target of training is make loss value as small as possibleThus far, you have completed a full neural network training by using OneFlow.

## Code interpretation
The following is the full code
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

The next chapter is a brief description of this code.

The special feature of OneFlow compare to other deep learning framework:
```
@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
```
` train_job` function is been modify by `@flow.global_function`. Only the job function which been modify by `@flow.global_function` can be identify by OneFlow as a neural network training or forecasting task.

在OneFlow中一个神经网络的训练或者预测任务需要两部分信息：

* 一部分是这个神经网络本身的结构和相关参数，这些在上文提到的任务函数里定义；

* 另外一部分是使用什么样的配置去训练这个网络，比如`learning rate`、模型优化更新的方法。这些在`@flow.global_function(get_train_config())`中的`get_train_config()`配置。

这段代码里包含了训练一个神经网络的所有元素，除了上面说的任务函数及其配置之外：

- `check_point.init()`: 初始化网络模型参数；

- `load_data(BATCH_SIZE)`: 准备并加载训练数据；

- `train_job(images, labels).get().mean()`: 返回每一次训练的损失值；

- `if i % 20 == 0: print(loss)`: 每训练20次，打印一次损失值。




以上只是一个简单网络的示例，在[使用卷积神经网络进行手写体识别](lenet_mnist.md)中，我们对使用OneFlow的流程进行了更加全面和具体的介绍。 另外，还可参考OneFlow[基础专题](../basics_topics/data_input.md)中对于训练中各类问题的详细介绍。


我们同时还提供了一些经典网络的[样例代码](https://github.com/Oneflow-Inc/OneFlow-Benchmark)及数据供参考。




