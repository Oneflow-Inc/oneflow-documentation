# Distributed training

In OneFlow, you only need few simple lines of configuration and OneFlow will automatically deal with tasks scheduling, resources parallelism  and so on. Thus, we do not need change network structure and the logic of code. Then we can easily use distributed training.

OneFlow's unique distributed training capability is the **most important feature** that sets OneFlow apart from other frameworks.

In this article, we will introduce:

* How to switch a program from single machine to distributed system.

* The concept and job of node in OneFlow.

## The distribution advantage of OneFlow.

* OneFlow use decentralized streaming architecture. Not like  `master` and `worker` architecture, it can optimize the communication efficiency of node network to the maximum extent.

* Support for  `consistent view`, the whole network only need a logic input and output.

* Provide a `mirrored view` compatible with other frameworks. Users who are familiar with the distributed training of other frameworks can start it quickly. 

* Only a few lines of configuration code are needed to switch a program from single machine to distributed system.

## Configuration of the distributed training

By the distributed training interface of OneFlow, you only need few configuration(Specify the distributed computing nodes IP and the number of devices) to realize distributed training network.

In another word, it make single machine program and distributed machine program almost the same. User just need to focus on **job logic ** and **structures of model** without worrying about distribution execution. **OneFlow will automatically deal with tasks scheduling, resources parallelism  and other issue.**

Here is an example to change a program running on a single machine to run on a distributed system with few configurations. 

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

In  `oneflow.config` , we provide distributed program related interface. We mainly use two of them:

* `oneflow.config.gpu_device_num` : set the number of device. This will apply to all machine.

* `oneflow.config.ctrl_port` :set the port number of communications, all the machine will apply the same port. 

The following demo we set all machine use one device and use port 9988 to communicate. User can change the configuration according to the environment.
```python
#device number
flow.config.gpu_device_num(1)
#Port number
flow.env.ctrl_port(9988
```

Be careful, even use single machine training. If we have multiple devices, we can use  `flow.config.gpu_device_num`  to change a program running on a single machine to run on a distributed system. In the code below, we set two devices in one machine to do the distributed training:
```python
flow.config.gpu_device_num(2)
```

### Node configuration

Then we need to config the connection between the machines in network. In OneFlow, the distributed machine called `node`.

The network information of each node is stored as a `dict`. The key "addr" is corresponding to IP of this node. All the nodes are stored in a  `list`, use  `flow.env.machine` to connect OneFlow. OneFlow will automatically generate the connection between nodes.

```python
nodes = [{"addr":"192.168.1.12"}, {"addr":"192.168.1.11"}]
flow.env.machine(nodes)
```

In the code above, we have two nodes in our distribution system. The IP is "192.168.1.12" and "192.168.1.11".

It should be noted that the node 0 in list (in the above code is 192.168.1.12) is called `master node`. After the whole distributed training system starting, it will create the graph  and other nodes are waiting. When the graph is created, all nodes will receive a notice to know the other nodes they are in contact with and work together with decentralized way.

During the training process, `master node`  will remain the standard output and store the  model. Other nodes are responsible for calculation.

We can wrap the configuration code for distributed training as a function, which is easy to be call:

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
After adding the configurations code, we can change the single machine program to the distributed training program. Just run the same program in all nodes.

We can compare the distributed training program with **single machine training program**. We will find that we only add `config_distributed`  function and called it.

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

