# 分布式训练

深度学习中，越来越多的场景需要分布式训练。由于分布式系统面临单机单卡所没有的分布式任务调度、复杂的资源并行等问题，因此，通常情况下，分布式训练对用户有一定的技术门槛。

在 OneFlow 中，通过顶层设计与工程创新，做到了 [分布式最易用](./essentials_of_oneflow.md#oneflow_2)，用户不需要特别改动网络结构和业务逻辑代码，就可以方便地使用 OneFlow 进行分布式训练。这是 OneFlow 区别于其它框架的 **最重要特性**。

本文将介绍：

* 如何将单机程序修改为分布式程序

* OneFlow 中节点概念及分工

## OneFlow 分布式优势

* 采用去中心化的流式架构，而非 `master` 与 `worker` 架构，最大程度优化节点网络通信效率

* 提供 `consistent view`，使得用户可以像编写单机单卡程序那样编写分布式程序

* 提供 `mirrored view`，熟悉其它框架分布式训练的用户可直接上手

* 极简配置，由单机单卡的训练程序转变为分布式训练程序，只需要几行配置代码

## 配置分布式训练网络

只需要增加几行简单的配置代码，指定分布式计算的节点 IP 以及每个节点使用 GPU 的数量，即可实现分布式的训练网络。

换句话说，这使得单机训练程序与分布式训练程序几乎是一样的，作为 OneFlow 用户，只需要专注于程序的 **业务逻辑** 及 **模型结构本身** ，而不用操心分布式执行问题。分布式的一切问题，都由 OneFlow 处理。

下面，我们会介绍一个例子：将单机版的训练程序，通过添加几行配置代码后将其改造为分布式训练程序。

### 单机训练程序
以下是单机训练程序的框架，因为其网络结构及业务逻辑与文末的分布式程序完全一样，因此函数实现未详细列出。

```python
import numpy as np
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100

def mlp(data):
  #构建网络...


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
  #作业函数实现...
  #配置训练优化方法和参数


if __name__ == '__main__':
  #调用作业函数，开始训练...
      loss = train_job(images, labels)
  #...
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

如以上代码中，我们的分布式系统中有2个节点，IP 分别为"192.168.1.12"与"192.168.1.11"。

注意，节点 list 中的第0个节点(以上代码中的"192.168.1.12")，又称为 `master node`，整个分布式训练系统启动后，由它完成构图，其它节点等待；当构图完成后，所有节点会收到通知，知晓各自联系的其它节点，去中心化地协同运行。

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

### 分布式训练及代码
单机程序加入 OneFlow 的分布式配置代码后，就成为了分布式程序，在所有的节点运行一样的程序即可。

我们可以将分布式训练程序与上文的 **单机训练程序** 比较，会发现仅仅只是增加了 `config_distributed` 函数并调用，我们之前的单机训练脚本，就成为了分布式训练脚本。

分布式脚本代码：[distributed_train.py](../code/basics_topics/distributed_train.py)

在 `192.168.1.12` 及 `192.168.1.11` 上 均运行：
```shell
wget https://docs.oneflow.org/code/basics_topics/distributed_train.py
python3 distributed_train.py
```

`192.168.1.12` 机器上将显示程序结果。

## FAQ
- 运行本文分布式代码后，程序长期等待，未显示计算结果
> 1. 请检查 ssh 配置，确保两台机器之间能够免密 ssh 互联
> 2. 请确保两台机器使用了相同版本的 OneFlow、运行的脚本程序完全一样
> 3. 请确保训练使用的端口未被占用，或使用 `oneflow.config.ctrl_port` 更换端口
> 4. 如果在环境变量中设置了代理，请确保代理能够正常工作，或者取消掉代理

- 在 docker 中跑训练，程序长期等待，未显示计算结果
> docker 默认的模式下，物理机与容器中的端口是隔离的，请使用 `--net=host` host 模式，或者启动容器时使用 `-p` 选项进行端口映射。具体请查阅 docker 的手册

- 存在虚拟网卡的情况
> 若存在虚拟网卡，可能因为 nccl 的通信走虚拟网卡而无法通信。此时需要通过 `export NCCL_SOCKET_IFNAME=device_name` 来指定通信网卡，具体可参阅 [nccl 官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html?highlight=nccl_socket_ifname#nccl-socket-ifname)
