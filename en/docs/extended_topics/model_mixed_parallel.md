# Characteristics of parallelism in OneFlow

In [Consistent and Mirrored view](consistent_mirrored.md), we have already known OneFlow provides mirrored and consistent two point of view and learned about the  `consistent` in OneFlow have some special characteristics.

Because in `consistent_view`, OneFlow gives the  unified view on logical side. When doing the distributed training, user can choose use data parallelism, model parallelism or mixed parallelism.

In this section, we will keep going through the special ` consistent` view in OneFlow. Which includes:

* Data only parallelism in `consistent_view` flow chart.

* Mixed parallelism in `consistent_view` flow chart.

* The advantages of mixed parallelism and the applicable scenario.

* Example of mixed parallelism.

## Network logical diagram in model training

We need to set up a simple multi-layer network first and use this network to discuss parallelism methods. The structure like the figure shows:

![多层网络逻辑图](imgs/para_logical.png)

In each layer, we have **samples**(in grey), **models**(in blue) and **operators**(circles) which operating on both of them. To simplify our discussion, we can limit the sample and model as **matrix**. The operator applying on them we call it **matrix multiplication**.

Compare the figure above, we can easily get the logic of the network:

* The input of layer 0 is `Data 0` matrix and `Model 0 ` matrix. Then apply `op` (matrix multiplication) and output `Data 1`.

* The input of layer 1 is `Data 1` matrix and `Model 1` matrix. Then apply `op` and get `output`.

* The layer 2 is `output` layer and `Data 2` is the output of network. Of course, it can play as input in a deeper network.

In `consistent` view, OneFlow supports the data parallelism, model parallelism and mixed parallelism. We will introduce them in order but mixed parallelism is the key point.

## The characteristics of parallelism in consistent view

### Pure data parallelism

We have already known that in consistent view. The default parallelism method is data parallelism. If we choose mirrored view, we can only use data parallelism. Compare passing data in `numpy` when calling the job function (without using `flow.data.xxx_reader` in OneFlow). The difference between them is:

* In mirrored view, when we use pure data parallelism. We need to cut and reorganize data according to the number of device and use `list` to pass and receive data.

* But in consistent view we have the consistency on logic. Cutting data and reorganizing data will be completed by OneFlow framework.

The following figure is in consistent view, using pure data parallelism to achieve original logical network process:

![纯数据并行](imgs/para_consistent_data.png)

In pure data parallelism, we use two devices for training. Because we use **pure data parallelism**. We can see that for each original logical layer, the sample is divided in average to each device. We have complete **training model** in each device. The data after cutting processed by `op`. Finally we combine the data in each GPU and get the full complete data.

### Pure model parallelism

In `consistent` view, we can choose pure model parallelism (the configuration details we will talk about it later). The flow diagram is as follows:

![纯模型并行](imgs/para_consistent_model.png)

In pure model parallelism example, we still use two devices for training. In each layer of original logic model is processed by `op  `on **part of model** and **complete data**. Then combine the output and get whole results.

One thing we need to mention is in above figure. The output from each device on layer 0 **cannot** use as the input in layer 1: Because in model parallelism, in order to complete the operation. We need partial model and **complete** data. To solve this problem, OneFlow use `boxing` mechanism.

`boxing` will count the data in each nodes in distributed training and divide or assemble data properly then send to corresponding GPU. Except the model assembling in model parallelism. The reverse gradient synchronization in data parallelism also will use  `boxing`  to solve problem.

The algorithm in `boxing` is complex. But it is open to users. The reason of adding  for adding `boxing` is for keep user from confused. In this article, we only need to remember that OneFlow will automatically solve the data distribution issue.

## Choose the optimal parallelism method

The difference between data parallelism and model parallelism is not constant. The sample, model size and model structure decide the performance in distributed training. We need analysis  particular case.

To be concluded:

* In data parallelism case, the information need to synced is **gradient** in backpropagation. Thus, we need to make sure the synchronization's speed in different nodes is faster than the calculation's speed in side nodes. For example, the **Convolution Layer** has fewer parameters, but it need large scale of calculation. It is suitable for data parallelism.

* In model parallelism, we can send the complete model in logical to **each device**. It can deal with the oversize model problem. Thus it is suitable for the neural network with massive parameters (like full connection layer) to use model parallelism.

In fact, we can use **mix parallelism**. That means OneFlow uses different parallelism in different parts of training process. For example, at the beginning of the neural network, which have few parameters and need large calculation. We better user data parallelism. But the layer like full connection layer which have many parameters we should use model parallelism. The following is the demonstration figure for the neural network in begin of the section which use **mixed parallelism**.

![混合并行](imgs/para_consistent_mixed.png)

For now, all other popular framework didn’t support the mixed parallelism otherwise need be deep customizing. But in OneFlow, we can use it very simple. We also can use mixed parallelism distributed training with network relay to deep optimize distributed systems.

## Mixed parallelism example:

### Code example 

In `consistent`  view, we use mixed parallelism to MLP model: the input layer and hidden layer use data parallelism, output layer use model parallelism.

Complete Code: [mixed_parallelism_mlp.py](../code/extended_topics/mixed_parallelism_mlp.py)

More details explanations in later "code explanations"

```python
# mixed_parallelism_mlp.py
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
        name="dense1",
    )
    return flow.layers.dense(
        hidden,
        10,
        kernel_initializer=initializer,
        # dense为列存储，进行split(0)切分
        model_distribute=flow.distribute.split(axis=0),
        name="dense2",
    )


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="train", function_config=get_train_config())
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
    flow.config.gpu_device_num(2)
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )

    for epoch in range(3):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
```

### Code explanation

The above script is modified from the demo in [3 min quick start](../quick_start/quickstart_in_3_min.md). Compare two version of script, we can see how easy to configure the parallelism method in `consistent_view`. Only need modify on code of single machine.

The crucial parts are:

* Use  `oneflow.config.gpu_device_num`  to set the device number in training:

```python
  flow.config.gpu_device_num(2)
```

* `reshape` and `hidden` using data parallelism as default. The output layer can set `model_distribute` as `flow.distribute.split(axis=0)` to change to model parallelism:

```python
def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    return flow.layers.dense(hidden,
                             10,
                             kernel_initializer=initializer,
                             # dense is columns storing，process split(0) cutting 
                             model_distribute=flow.distribute.split(axis=0),
                             name="output"
                             )
```

You may curious about why `split(axis=0)` is column cutting. What we need to explain is in OneFlow  `dense` is column storing. Thus the `flow.distribute.split(axis=0)` in above code is column splitting.

In addition, `flow.layers.dense`  use `model_distribute`  to set parallelism method. It use the common  `get_variable` to create `blob` in basic level from inner.  Use `get_variable` to config parallelism method called  `distribute`.

We can see that we only modify just few things. Then change parallelism method to mixed parallelism in distributed training. It is the main difference between OneFlow and other framework.

## Flow parallelism example

Besides the model parallelism, OneFlow also provides a more flexible parallelism method which is flow parallelism. It can allow user use  `scope.placement` to specify the device of the operator.

In flow parallelism, the part of layers of whole network are on one device and other layers are on other devices. They work as relay, switch between devices in different phase.

In the following example, we change few code in "Using consistent view in OneFlow" of  [Consistent and Mirrored view](consistent_mirrored.md) and demonstrate flow parallelism.

### Code Example

Complete Code: [mixed_parallelism_lenet.py](../code/extended_topics/mixed_parallelism_lenet.py)

More details please refer to code explanation later.

```python
# mixed_parallelism_lenet.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="conv1",
    )
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding="SAME", name="pool1")
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="conv2",
    )
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding="SAME", name="pool2")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    with flow.scope.placement("gpu", "0:0"):
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="hidden",
        )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5)

    with flow.scope.placement("gpu", "0:1"):
        output = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="outlayer"
        )
    return output


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="train", function_config=get_train_config())
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == "__main__":
    flow.config.gpu_device_num(2)
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )

    for epoch in range(50):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
```

### Code explanation

There are only two line of code is important and they have same effect:

* Use  `oneflow.scope.placement` to specify the operator run on number 0 device in  `hidden` layer.

```python
  with flow.scope.placement("gpu", "0:0"):
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="hidden",
        
```

* Use  `oneflow.scope.placement` to specify the operator in  ` output ` layer run on number 1 device. 

```python
  with flow.scope.placement("gpu", "0:1"):
        output = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="outlayer"
        )
```

The first parameter in  `scope.placement` is to specify `cpu` or `gpu`. The second parameter is to specify machine number and device. Like we use the second device on first machine should be:

```python
  with flow.scope.placement("gpu", "1:2"):
    # ...
```

Flow parallelism can allow user to specify device for each op. It is very useful for user who master the distributed training to **optimize deeply**.

In addition, OneFlow also provides `oneflow.unpack`, `oneflow.pack`. Combine those with the characteristics of task scheduling in OneFlow. It will make the flow parallelism easier to use and more efficient. We will introduce these in other article.

