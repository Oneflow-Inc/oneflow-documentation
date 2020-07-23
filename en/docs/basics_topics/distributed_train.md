# Distributed training

In OneFlow, you only need few simple configuration and the frame of OneFlow will automatically deal with calling job function, resources paralleling  and other issue. Thus, we do not need change network structure and logic of code. Then we can easily use distributed training.

The ability of distributed training in OneFlow is very outstanding. This is the **main characters **distinguished between other framework.

In this article, we will introduce:

* How to change a solo program to distributing program.

* The concept and job division of node in OneFlow.

## The distribution advantage of OneFlow.

* OneFlow use decentralized and flow framework. Not like  `master` and `worker` frame, it can maximum optimize the network speed between nodes.

* Support for  `consistent strategy`, the whole network only need only logic input and output.

* Also support to adapt with `mirrored strategy` from other framework. User who familiar with the distributed training in other frame can easily use OneFlow.

* The minimalist configuration, only need few line of code can change a single node of the training program into a distributed training program.

## Configuration of the distributed training

By the distribued training port of OneFlow, you only need few configuration(Specify the distributed computing nodes IP and the number GPU used of each node ) to achieve distribued training network.

In another word, it make solo training as same as distribued training. As the user of OneFlow, just need to focus on **job logic ** and **structures of model**. No need to worry anout distribution execution.**Frame of OneFlow will automatically deal with calling job function, resources paralleling  and other issue. (链接可能有问题）**

This is a example for change the solo training to a distributed training by adding few code:

### Solo training
This is solo training framework, code of function will show in distributed training later on.
```python
import numpy as np
import oneflow as flow

BATCH_SIZE = 100
DATA_DIRECTORY = '/dataset/mnist_kaggle/60/train'
IMG_SIZE = 28
NUM_CHANNELS = 1

def _data_load_layer(data_dir=DATA_DIRECTORY, arg_data_part_num=1, fromat="NHWC"):
    #loading data...

def lenet(data, train=False):
    #constructed network...


def get_train_config():
    #config training env


@flow.global_function(get_train_config())
def train_job():
    #achieve job function

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

### Configuration of ports and GPU

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

每个节点的组网信息，由一个 `dict` 类型存放，其中的 "addr" 这个 key 对应了节点的 IP 。 所有的节点放置在一个 `list` 中，经接口 `flow.env.machine` 告之 OneFlow ，OneFlow 内部会自动建立各个节点之间的连接。

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
