# 分布式训练

在 OneFlow 中您只需要简单的几行配置，OneFlow 框架内部会自动处理任务调度、资源并行等问题，因此，您并不需要特别改动网络结构和业务逻辑代码，就可以方便地使用分布式训练。

OneFlow 的分布式训练能力独树一帜，是 OneFlow 区别于其它框架的 **最重要特性**。

本文将介绍：

* 如何将单机程序修改为分布式程序

* OneFlow 中节点概念及分工

## OneFlow 分布式优势

* 采用去中心化的流水架构，而非 `master` 与 `worker` 架构，最大程度优化节点网络通信效率

* 提供 `consistent strategy` ，整个节点网络中只需要逻辑上唯一的输入与输出

* 提供兼容其它框架的`mirrored strategy`，熟悉其它框架分布式训练的用户可直接上手

* 极简配置，由单一节点的训练程序转变为分布式训练程序，只需要几行配置代码

## 配置分布式训练网络

通过 OneFlow 提供的分布式配置的接口，您只需要简单的几行配置(指定分布式计算的节点 ip 以及每个节点使用 gpu 的数量)即可实现分布式的训练网络。

换句话说，这使得单机训练程序与分布式训练程序几乎是一样的，作为 OneFlow 用户，只需要专注于程序的 **业务逻辑** 及 **模型结构本身** ，而不用操心分布式执行问题。**OneFlow 框架会自动帮您处理复杂的任务调度、资源并行等问题。**

下面，我们会介绍一个例子：将单机版的训练任务，通过添加几行配置代码后将其改造为分布式训练任务。

### 单机训练程序
以下是单机训练程序的框架，因为各个函数的代码会在下文分布式程序中呈现，在此就未呈现。
```python
import numpy as np
import oneflow as flow

BATCH_SIZE = 100
DATA_DIRECTORY = '/dataset/mnist_kaggle/60/train'
IMG_SIZE = 28
NUM_CHANNELS = 1

def _data_load_layer(data_dir=DATA_DIRECTORY, arg_data_part_num=1, fromat="NHWC"):
    #加载数据 ...

def lenet(data, train=False):
    #构建网络 ...


def get_train_config():
    #配置训练环境


@flow.global_function(get_train_config())
def train_job():
    #任务函数的实现

def main():
  check_point = flow.train.CheckPoint()
  check_point.init()

  for step in range(50):
      losses = train_job().get()
      print("{:12} {:>12.10f}".format(step, losses.mean()))

  check_point.save('./lenet_models_1') # need remove the existed folder
  print("model saved")

if __name__ == '__main__':
  main()
```

### GPU及端口配置

在 `oneflow.config` 模块中，提供了分布式相关的设置接口，我们主要使用其中两个：

* `oneflow.config.gpu_device_num` : 设置所使用的 GPU 的数目，这个参数会应用到所有的机器中；

* `oneflow.config.ctrl_port` : 设置用于通信的端口号，所有机器上都将使用相同的端口号进行通信。

以下代码中，我们设置每台主机使用的 GPU 数目为1，采用9988端口通信。大家可以根据自身环境的具体情况进行修改。
```python
#每个节点的 gpu 使用数目
flow.config.gpu_device_num(1)
#通信端口
flow.env.ctrl_port(9988)
```

注意，即使是单机的训练，只要有多张 GPU 卡，我们也可以通过 `flow.config.gpu_device_num` 将单机程序，设置为单机多卡的分布式程序，如以下代码，设置1台(每台)机器上，2张 GPU 卡参与分布式训练：
```python
flow.config.gpu_device_num(2)
```

### 节点配置

接着，我们需要配置网络中的主机关系，需要提前说明的是，OneFlow 中，将分布式中的主机称为节点(`node`)。

每个节点的组网信息，由一个 `dict` 类型存放，其中的 "addr" 这个 key 对应了节点的 IP 。
所有的节点放置在一个 `list` 中，经接口 `flow.env.machine` 告之 OneFlow ，OneFlow 内部会自动建立各个节点之间的连接。

```python
nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
flow.env.machine(nodes)
```

如以上代码中，我们的分布式系统中有2个节点，IP分别为"192.168.1.12"与"192.168.1.11"。

注意，节点list中的第0个节点(以上代码中的"192.168.1.12")，又称为`master node`，整个分布式训练系统启动后，由它完成构图，其它节点等待；当构图完成后，所有节点会收到通知，知晓各自联系的其它节点，去中心化地协同运行。

在训练过程中，由 `master node` 保留标准输出及保存模型，其它节点只负责计算。

我们可以将针对分布式的配置代码封装为函数，方便调用：

```python
def config_distributed():
    print("distributed config")
    #每个节点的gpu使用数目
    flow.config.gpu_device_num(1)
    #通信端口
    flow.env.ctrl_port(9988)

    #节点配置
    nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
    flow.env.machine(nodes)
```

### 分布式训练完整脚本
单机程序加入 OneFlow 的分布式配置代码后，就成为了分布式程序，在所有的节点运行一样的程序即可。

我们可以将分布式训练程序与上文的 **单机训练程序** 比较，会发现仅仅只是增加了 `config_distributed` 函数并调用，我们之前的单机训练脚本，就成为了分布式训练脚本。

以下是完整代码：[distributed_train.py](../code/basics_topics/distributed_train.py)

```python
import numpy as np
import oneflow as flow

BATCH_SIZE = 100
DATA_DIRECTORY = '/dataset/mnist_kaggle/60/train'
IMG_SIZE = 28
NUM_CHANNELS = 1

def _data_load_layer(data_dir=DATA_DIRECTORY, arg_data_part_num=1, fromat="NHWC"):
  if fromat == "NHWC":
    image_shape = (IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
  else:
    image_shape = (NUM_CHANNELS, IMG_SIZE, IMG_SIZE)
  image_blob_conf = flow.data.BlobConf("img_raw", shape=image_shape,
                                       dtype=flow.float32, codec=flow.data.RawCodec())
  label_blob_conf = flow.data.BlobConf("label", shape=(1,1), dtype=flow.int32,
                                       codec=flow.data.RawCodec())
  return flow.data.decode_ofrecord(data_dir, (label_blob_conf, image_blob_conf),
                                   data_part_num=arg_data_part_num, name="decode", batch_size=BATCH_SIZE)

def lenet(data, train=False):
  initializer = flow.truncated_normal(0.1)
  conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu,
                             kernel_initializer=initializer)
  pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME')
  conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu,
                             kernel_initializer=initializer)
  pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME')
  reshape = flow.reshape(pool2, [pool2.shape[0], -1])
  hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
  if train: hidden = flow.nn.dropout(hidden, rate=0.5)
  return flow.layers.dense(hidden, 10, kernel_initializer=initializer)

def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config

@flow.global_function(get_train_config())
def train_job():
  (labels, images) = _data_load_layer(arg_data_part_num=60)

  logits = lenet(images, train=True)
  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
  flow.losses.add_loss(loss)
  return loss

def config_distributed():
    #每个节点的gpu使用数目
    flow.config.gpu_device_num(1)
    #通信端口
    flow.env.ctrl_port(9988)

    #节点配置
    nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
    flow.env.machine(nodes)

def main():
  config_distributed()
  check_point = flow.train.CheckPoint()
  check_point.init()

  for step in range(50):
      losses = train_job().get()
      print("{:12} {:>12.10f}".format(step, losses.mean()))

  check_point.save('./lenet_models_1') # need remove the existed folder
  print("model saved")

if __name__ == '__main__':
  main()
```
