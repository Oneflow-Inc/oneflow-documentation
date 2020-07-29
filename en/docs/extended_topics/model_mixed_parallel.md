
# Characteristics of parallel in OneFlow

In [Consistent and Mirrored view](consistent_mirrored.md), we already know OneFlow provide mirrored and consistent two point of view. And be aware of  `consistent` in OneFlow have some special characteristics.

Thus, in `consistent_view`, OneFlow give the  unified view on logical side. When doing the distributed training, use can choose use data parallel, model parallel or mix parallel.

In this article, we will keep go through the ` consistent` view in OneFlow. Which includes:

* Process demo of pure data parallel in `consistent_view`.

* Process demo of mixed parallel in `consistent_view`.

* The advantages of mixed parallel and the scenario.

* Example of mixed parallel.

## Network logic diagram of model training
We need to set up a simple multi-layer network first and use this network to discuss parallel methods. The structure like the figure shows:

![多层网络逻辑图](imgs/para_logical.png)

In each layers, we have **samples**(in grey), **models**(in blue) and **operators**(circles) which operating on both of them. To simplify our discussion, we can limiting the samples and model as** matrixes**. The operator applying on them we called it **Matrix multiplication**.

Compare the figure above, we can easily get the logic of the network:

* The input of layer 0 is `Data 0` matrix and `Model 0`matrix. Then apply `operator`(matrix multiplication) and give output `Data 1`.

* The input of layer 1 is `Data 1` matrix and `Model 1`matrix. Then apply `operator` and get `output`.

* The layer 2 is `output layer` and `Data 2` is the output of network. Of course, it can play as input in deeper network.

In `consistent` view, it supports the data parallel, model parallel and mixed parallel. We will introduce those in order but mixed parallel is the key thing.

## The characteristics of parallel in consistent view

### Pure data parallel

We already know that in consistent view. The default parallel method is data parallel. If we choose mirrored view, we only can use data parallel. Compare passing data in `numpy` when calling the job function with use `flow.data.xxx_reader` in OneFlow. The difference between them is:

* In mirrored view, when we use pure data parallel. We need to cut assembly data according to the number of GPU and use `list` to pass and receive data.

* But in consistent view we have the consistency on logic. Cutting data and assembly data will complete by OneFlow framework.

The following figure is in consistent view, using pure data parallel to achieve original logical network process:

![纯数据并行](imgs/para_consistent_data.png)

In pure data parallel, we use two GPU for training. Because we use **pure data parallel**. We can see that for each original logical layer, the sample is divided in average to each GPU. We have complete **training model** in each GPU. The data after cut process by `operator`. Finally combined the data in each GPU and get the full complete data.

### Pure model parallel
In `consistent` view, we can choose pure model parallel (the configuration details will talk about later). The process schematic diagram:

![纯模型并行](imgs/para_consistent_model.png)

In pure model parallel example, we still use two GPU for training. In each layer of original logic model is process by `operator `on **part of model** and **complete data**. Then combine the output and get whole results.

One thing we need to mention is in above figure. The output from each GPU on layer 0 **cannot** use as the input in layer 1: Beacuse in model parallel, in order to run the operator. We need part of model and **complete** data. To solve this problem, OneFlow use `boxing` function.

`boxing` will count the data in each nodes in distributed training and divide or assemble data properly then send to corresponding GPU. Except the model assembling in model parallel. The reverse gradient synchronization in data parallel also will use  `boxing`  to solve problem.

The algorithm in `boxing` is complex. But it is open to users. The reason of adding  for adding `boxing` is for keep user from confused. In this article, we only need to remember that OneFlow will automatically solve the data distribution issue.

## Choose the optimal parallel method
The difference between data parallel and model parallel is constant. The sample scale, model scale and model structure decide the performance in distributed training. We need analysis  particular case.

To be concluded:

* In data parallel case, the information need to synced is ** gradient** need to be pass reversely. Thus, we need to make sure the synchronization speed in different nodes is faster than the calculations speed in side nodes. Such as there is not much parameters in **Convolutional Layer**. But need large scale of calculation. It is suitable for data parallel.

* In model parallel, we can send the complete model in logical to **each GPU**. Can deal with the oversize model issue. Thus it is suitable for the neural network which have massive parameters (like full connection layer).

In fact, we can use **mix parallel**. That means use different parallel in different part of training process.Such as at the beginning of the neural network, just have few parameters and need for large calculation. We better user data parallel. But layer like full connection layer which have many parameters we should use model parallel. The following is the demonstration figure for the neural network in begin of the article which use **mixed parallel**.

![混合并行](imgs/para_consistent_mixed.png)

For now, all other popular framework didn’t support the mixed parallel otherwise need be deep customizing. But in OneFlow, we can use it very simple. We aslo can use mixed parallel distributed training with network relay to deep optimize distributed systems.

## Mixed parallel example:
### Demo script
In `consistent`  view, we use mixed parallel to MLP model: data input layer and hidden layer is data parallel, output layer use model parallel.

Name: [mixed_parallel_mlp.py](../code/extended_topics/mixed_parallel_mlp.py)

More details explanations in "script explanations"

```python
#mixed_parallel_mlp.py
import oneflow as flow
import oneflow.typing as oft

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="dense1")
    return flow.layers.dense(hidden,
                             10,
                             kernel_initializer=initializer,
                             # dense为列存储，进行split(0)切分
                             model_distribute=flow.distribute.split(axis=0),
                             name="dense2"
                             )


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="train", function_config=get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> oft.Numpy:
    logits = mlp(images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == '__main__':
    flow.config.gpu_device_num(2)
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)

    for epoch in range(3):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0: print(loss.mean())
```

### Script explanation
The above script is modified from the demo in [3 min quick start](../quick_start/quickstart_in_3_min.md). Compare two version of script, we can see how easy to configure the parallel method in `consistent_view`. Only need modify on code of solo machine.

The crucial parts are:

* Use  `oneflow.config.gpu_device_num`  to set the GPU number in training:
```python
  flow.config.gpu_device_num(2)
```

* `reshape` and `hidden` is default using data parallel. The output layer can set `model_distribute` as `flow.distribute.split(axis=0)` to change to model parallel:
```python
def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="dense1")
    return flow.layers.dense(hidden,
                             10,
                             kernel_initializer=initializer,
                             # dense为列存储，进行split(0)切分
                             model_distribute=flow.distribute.split(axis=0),
                             name="dense2"
                             )
```
You may curious about why `split(axis=0)` is column cutting.What we need to explain is in OneFlow  `dense` is column storing. Thus the `flow.distribute.split(axis=0)` in above script is column cutting.

In addition, `flow.layers.dense`  use `model_distribute`  to set parallel method. It use the common  `get_variable` to creates `blob` in basic level from inner.  Use `get_variable` to config parallel method called  `distribute`.

We can see that we only modify just few things. Then change parallel method to mixed parallel in distributed training. It is the main difference between OneFlow and other framework.

## Flow parallel example
Besides the model parallel, OneFlow also provides a more flexible parallel method which is flow parallel. It can let user use  `scope.placement` to display specify hardware of the operator.

In flow parallel, the part of layer of whole network is on one hardware and other layer is on other hardware. They work as relay, switch between hardware in different step.

In following example, we change few code in "Using consistent view in OneFlow" of  [Consistent and Mirrored view](consistent_mirrored.md) and demonstrate flow parallel.

### Example

Name: [mixed_parallel_lenet.py](../code/extended_topics/mixed_parallel_lenet.py)

More details in script explanation.

```python
#mixed_parallel_lenet.py
import oneflow as flow
import oneflow.typing as oft

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer, name="conv1")
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name="pool1")
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer, name="conv2")
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name="pool2")
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    with flow.scope.placement("gpu", "0:0"):
        hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)

    with flow.scope.placement("gpu", "0:1"):
        output = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="outlayer")
    return output


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="train", function_config=get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> oft.Numpy:
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


if __name__ == '__main__':
    flow.config.gpu_device_num(2)
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)

    for epoch in range(50):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0: print(loss.mean())
```
### Script explanation

There are only two line of code is important and they have same nature:

* Use  `oneflow.scope.placement` to specify the operator run on number 0 GPU in  `hidden` layer.
```python
  with flow.scope.placement("gpu", "0:0"):
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
```

* Use  `oneflow.scope.placement` to specify the operator run on number 1 GPU in  ` output ` layer.
```python
  with flow.scope.placement("gpu", "0:1"):
    output = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="outlayer")
```

The first parameter in  `scope.placement` is to specify `cpu` or `gpu`. The second parameter is to specify machine number and device. Like use the second GPU on first machine should be:
```python
  with flow.scope.placement("gpu", "1:2"):
```

Flow parallel can let user to specify device for each operator. It is very useful for user who master the distributed training to **deep optimize**.

In addition, OneFlow also provide `oneflow.unpack`, `oneflow.pack`. Combine those with the characteristics of task scheduling in OneFlow. That will make the flow parallel easier to use and more efficient. We will introduce those in other article.

