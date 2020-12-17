# Distributed Training

In deep learning, more and more scenarios require distributed training. Since distributed systems face problems such as distributed task scheduling and complex resource parallelism in multiple cards machines. Thus distributed training usually has a certain technical threshold for users.

In OneFlow, through top-level design and engineering innovation. It is [easiest use distribution system](./essentials_of_oneflow.md#oneflow_2). Users can easily use OneFlow for distributed training without making any special changes to the network structure or job logic. This is the **most important feature** that make OneFlow different from other frameworks.

In this article, we will introduce:

* How to switch a program platform from a single machine to a distributed system.

* The concept and mission of node in OneFlow.

## The Distribution Advantage of OneFlow.

* OneFlow use decentralized streaming architecture. Not like  `master` and `worker` architecture, it can optimize the communication efficiency of network to the maximum extent.

* Support `consistent view`, the whole network only needs one logic input and output.

* A `mirrored view` compatible with other frameworks is provided. Users who are familiar with the distributed training of other frameworks can learn to use it quickly.

* Only a few lines of configuration code are needed to switch a program platform from a single machine to a distributed system.

## Configuration of the Distributed Training

By the distributed training interface of OneFlow, you only need a few configuration to specify the distributed computing nodes IP and the number of devices for performing distributed training network.

In another word, it makes a single machine program and a distributed machine program almost the same in terms of complexity of coding. User just need to focus on **job logic** and **structures of model** without worrying about distribution execution. Everything related to distribution is handled by OneFlow.

Here is an example to change a program run on a single machine to be run on a distributed system with few configurations.

### Single Machine Program
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

### Configuration of Ports and Device

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

### Node Configuration

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

### Complete Code of Distributed Training
After adding the configurations code, the program becomes a distributed training one. Just follow the same step as we do in a single machine program.

Compared with **single machine training program**, the distributed training program only needs to call one more function named `config_distributed`.

Distribution script: [distributed_train.py](../code/basics_topics/distributed_train.py)

Running on both `192.168.1.12` and `192.168.1.12`:

```shell
wget https://docs.oneflow.org/code/basics_topics/distributed_train.py
python3 distributed_train.py
```
The result of the program will be displayed on `192.168.1.12`.

## FAQ

- After running this distribution code, the program waits for a long time and does not display the calculation results。

> 1. Please check the ssh configuration to ensure that the two machines can be interconnected with each other ssh-free.
> 2. Make sure that both machines are using the same version of OneFlow and are running the exact same script program.
> 3. Make sure the port used for training is unoccupied or replace the port with `oneflow.config.ctrl_port`.
> 4. If a proxy is set in an environment variable, make sure the proxy works or disable it.

- Run training in docker, program waits for a long time and does not show calculation results.

> In default mode of docker, the machine is isolated from the ports in the container. Then use `--net=host` (host mode) or use the `-v` option for port mapping when starting the container. For details information please refer to the docker manual.

- The communications library was not installed correctly

> Make sure the version of the communication library (nccl) is the same on each machine during distributed training.

- Using virtual network cards

> If there are virtual network cards, you may not be able to communicate with nccl. In this case, you need to specify the communication network cards by `export NCCL_SOCKET_IFNAME=device_name`. More details please refer to [nccl official documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs).
