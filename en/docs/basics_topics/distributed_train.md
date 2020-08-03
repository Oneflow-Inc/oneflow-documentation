# Distributed training

In OneFlow, you only need few simple configuration and the frame of OneFlow will automatically deal with calling job function, resources paralleling  and other issue. Thus, we do not need change network structure and logic of code. Then we can easily use distributed training.

The ability of distributed training in OneFlow is very outstanding. This is the **main characters **distinguished between other framework.

In this article, we will introduce:

* How to change a solo program to distributing program.

* The concept and job division of node in OneFlow.

## The distribution advantage of OneFlow.

* OneFlow use decentralized and flow framework. Not like  `master` and `worker` frame, it can maximum optimize the network speed between nodes.

* Support for  `consistent view`, the whole network only need only logic input and output.

* Also support to adapt with `mirrored view` from other framework. User who familiar with the distributed training in other frame can easily use OneFlow.

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
import oneflow.typing as tp

BATCH_SIZE = 100

def mlp(data):
  #build network...


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
  #achieve job function ...
  #Optimization method and parameters configuration training


if __name__ == '__main__':
  #call job function and start training...
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

Attention, the number zero node in list(192.168.1.12) is called `master node`. After the whole distribution system active, it will do the mapping and other node is waiting. We the mapping is done, all node will get a message. Know the connection between other nodes and itself. Then working together decentralized.

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
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="hidden",
    )
    return flow.layers.dense(
        hidden, 10, kernel_initializer=initializer, name="output-weight"
    )


def config_distributed():
    print("distributed config")
    # GPU number in each node
    flow.config.gpu_device_num(1)
    # communications channel
    flow.env.ctrl_port(9988)

    # node configuration 
    nodes = [{"addr": "192.168.1.12"}, {"addr": "192.168.1.11"}]
    flow.env.machine(nodes)


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    logits = mlp(images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == "__main__":
    config_distributed()
    flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())

```

