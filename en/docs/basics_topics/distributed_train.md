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

下面，我们会介绍一个例子：将单机版的训练作业，通过添加几行配置代码后将其改造为分布式训练作业。

### Solo training
以下是单机训练程序的框架，因为各个函数的代码会在下文分布式程序中呈现，在此就未详细列出。
```python
import numpy as np
import oneflow as flow
import oneflow.typing as oft

BATCH_SIZE = 100

def mlp(data):
  #构建网络...

def get_train_config():
  #配置训练参数及环境...

@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)) -> oft.Numpy:
  #作业函数实现...


if __name__ == '__main__':
  #调用作业函数，开始训练...
      loss = train_job(images, labels)
  #...
```

### Configuration of ports and GPU

In  `oneflow.config` , we provide distributed related port. We mainly use two of them:

* `oneflow.config.gpu_device_num` : set the number of GPU been using. This will apply to all machine.

* `oneflow.config.ctrl_port` :set the port number of communications, also will apply to all mechine.

The following demo we set all machine use one GPU and use port 9988 to communicate.User can change the configuration as well.
```python
#GPU number
flow.config.gpu_device_num(1)
#Port number
flow.env.ctrl_port(9988
```

Attention, even use solo training. If you have multiple GPU, we can use  `flow.config.gpu_device_num`  to change solo process to single machine with multiple GPU distribution process. The code below set two GPU in one machine to do the distribution training:
```python
flow.config.gpu_device_num(2)
```

### Node configuration

Then we need to comfig the connection between the machine in network. In OneFlow, the distributed machine called `node`.

Each node of the network information, is store by a  `dict`. The key "addr" in following example is this corresponding IP of this node. All node is stored in a  `list`, use port  `flow.env.machine` to connect OneFlow. OneFlow will automatically generate the connection between nodes.

```python
nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
flow.env.machine(nodes)
```

The code above, we have two nodes in our distribution system. Their IP are"192.168.1.12" and "192.168.1.11".

注意，节点 list 中的第0个节点(以上代码中的"192.168.1.12")，又称为`master node`，整个分布式训练系统启动后，由它完成构图，其它节点等待；当构图完成后，所有节点会收到通知，知晓各自联系的其它节点，去中心化地协同运行。

During the process of training, `master node`  will remain stander output and stored model. The calculation is done by other node.

We can specific to distribution configuration to package code as function. Then it is easier to use:

```python
def config_distributed():
    print("distributed config")
    #number of GPU used in each node
    flow.config.gpu_device_num(1)
    #communication channel 
    flow.env.ctrl_port(9988)

    #node configuration 
    nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
    flow.env.machine(n
```

### Complete script of distributed training
After solo process join yo OneFlow's configurations code, it will become distribution program. Just need run the same program in all nodes.

We can compare distributed training with  **solo training**. We will find that we turn the solo training script to distributed training script only by adding `config_distributed`  function and called it.

Name: [distributed_train.py](../code/basics_topics/distributed_train.py)

```python
import numpy as np
import oneflow as flow
import oneflow.typing as oft

BATCH_SIZE = 100

def mlp(data):
  initializer = flow.truncated_normal(0.1)
  reshape = flow.reshape(data, [data.shape[0], -1])
  hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
  return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output-weight")

def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.default_logical_view(flow.scope.consistent_view())
  config.train.model_update_conf({"naive_conf": {}})
  return config

def config_distributed():
  print("distributed config")
  #每个节点的gpu使用数目
  flow.config.gpu_device_num(1)
  #通信端口
  flow.env.ctrl_port(9988)

  #节点配置
  nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
  flow.env.machine(nodes)

@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)) -> oft.Numpy:
  logits = mlp(images)
  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
  flow.losses.add_loss(loss)
  return loss


if __name__ == '__main__':
  config_distributed()
  flow.config.enable_debug_mode(True)
  check_point = flow.train.CheckPoint()
  check_point.init()
  (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
  for epoch in range(1):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels)
      if i % 20 == 0: print(loss.mean())
```
