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
    #number of GPU used in each node
    flow.config.gpu_device_num(1)
    #communication channel 
    flow.env.ctrl_port(9988)

    #node configuration 
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
