# Distributed training

In OneFlow, you only need a few simple lines of configuration and OneFlow will automatically deal with tasks scheduling, resources parallelism and so on. Thus, we do not need to change network structure and logic of code, which makes distributed training easier to use.

OneFlow's unique distributed training capability is the **most important feature** that distinguishs OneFlow from other frameworks.

In this article, we will introduce:

* How to switch a program platform from a single machine to a distributed system.

* The concept and mission of node in OneFlow.

## The distribution advantage of OneFlow.

* OneFlow use decentralized streaming architecture. Not like  `master` and `worker` architecture, it can optimize the communication efficiency of network to the maximum extent.

* Support `consistent view`, the whole network only needs one logic input and output.

* A `mirrored view` compatible with other frameworks is provided. Users who are familiar with the distributed training of other frameworks can learn to use it quickly. 

* Only a few lines of configuration code are needed to switch a program platform from a single machine to a distributed system.

## Configuration of the distributed training

By the distributed training interface of OneFlow, you only need a few configuration to specify the distributed computing nodes IP and the number of devices for performing distributed training network.

In another word, it makes a single machine program and a distributed machine program almost the same in terms of complexity of coding. User just need to focus on **job logic** and **structures of model** without worrying about distribution execution. **OneFlow will automatically deal with tasks scheduling, resources parallelism as well as other issues.**

Here is an example to change a program run on a single machine to be run on a distributed system with few configurations. 

### Single machine program
Here is the framework of single machine training program. Because the code of each function will be presented in the distributed program below, it is not listed in detail here.
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

### Configuration of ports and device

In `oneflow.config`, we provide interfaces related to distributed program. We mainly use two of them:

* `oneflow.config.gpu_device_num` : set the number of device. This will be applied to all machines.

* `oneflow.config.ctrl_port` : set the port number of communications. All the machines will use the same port. 

In the following demo, we set all machines to use one device and use the port 9988 for communication. User can change the configuration according to their actual situation.
```python
#device number
flow.config.gpu_device_num(1)
#Port number
flow.env.ctrl_port(9988)
```

To be mentioned that, if we only have one single machine with multiple GPU devices in it, we can still use  `flow.config.gpu_device_num`  to change a program from running on a single machine to run on a distributed system. In the code below, we will use two GPU devices in one machine to do the distributed training:
```python
flow.config.gpu_device_num(2)
```

### Node configuration

Then we need to config the connection between the machines in network. In OneFlow, the distributed machine called `node`.

The network information of each node is stored as a `dict`. The key "addr" is corresponding with IP of this node. All nodes are stored in a `list`, which will be informed to Oneflow by `flow.env.machine`. OneFlow will automatically generate the connection between nodes.

```python
nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
flow.env.machine(nodes)
```

In the code above, we have two nodes in our distributed system. Their IP is "192.168.1.12" and "192.168.1.11".

It should be noted that the node 0 in list (in the above code is 192.168.1.12) is called `master node`. After the whole distributed training system starts, it will create the graph while the other nodes are waiting. When construction of graph is finished, all nodes will receive a notice specifying which nodes that they need to contact. Then they will work together in a decentralized way.

During the training process, `master node`  will deal with the standard output and store the model. The other nodes are only responsible for calculation.

We can wrap the configuration code for distributed training as a function, which is easy to be called:

```python
def config_distributed():
    print("distributed config")
    #number of device used in each node
    flow.config.gpu_device_num(1)
    #communication channel 
    flow.env.ctrl_port(9988)

    #node configuration 
    nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
    flow.env.machine(n
```

### Complete code of distributed training
After adding the configurations code, the program becomes a distributed training one. Just follow the same step as we do in a single machine program.

Compared with **single machine training program**, the distributed training program only needs to call one more function named `config_distributed`.

Code: [distributed_train.py](../code/basics_topics/distributed_train.py)

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
    # device number in each node
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

