# Term/concept explanation

In this article, we will explain some common terms and concepts in OneFlow. The main content is divided for algorithm engineer and framework developers:

-  **Algorithm development**
-  **Framework development**

In algorithms development part, we will explain some common terms and concepts used in the process of deep learning algorithms development and model training. But in framework development part, we will focus on the introduction of the inner design concepts of OneFlow and some relevant basic concepts.



## Algorithms developing

### 1.Placeholder

Placeholder is **data Placeholder**, this concept is used to define the shape of input or output data. There is no data in Placeholder. 

For example:

```python
import oneflow.typing as tp
def test_job(
    images: tp.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((32,), dtype=flow.int32),
) -> Tuple[tp.Numpy, tp.Numpy]:
    # do something with images or labels
    return (images, labels)
```

It define the input image shape is (32, 1, 28, 28) and data type is `flow.float32`, the input label shape is (32,) and data type is `flow.int32` in job function.



### 2.Tensor and Blob

Tensor is a common concept in other framework. In pytorch, Tensor contain the data, data type, grad, storing device and other attributes. Tensor can be used to create and describe the computation graph in forward and backward process. 

In OneFlow, the basic level also use the concept of Tensor. But there are some difference  about Tensor between OneFlow and pytorch/tensorflow. In order to provide sufficient support for distributed system and parallelism, the Tensor in OneFlow is more complex and have more types and attributes (Such as logical, physical, devices and attributes of distribution). The Tensor unified on logic level, could be divided to different devices. In order to simplify description, OneFlow hides the different types of Tensor, all the things are defined by a higher level concept named Blob.



In OneFlow, Blob has a corresponding base class `BlobDef`. You can print the attributes of  `Blob` when building network. As the following code, we can print  `conv1`'s `shape` and `dtype`:

```python
print(conv1.shape, conv1.dtype)
```

Blob can only be Placeholder, but can also be a specific unit that contains values. 



### 3.Job Function

In OneFlow, we call the training, evaluating, predicting and inferential tasks as job function. Job function connects logic of user and  computing resources that managed by  OneFlow.

In OneFlow, we can use decorator `@oneflow.global_function` to change a function to a job function. By this decorator, we can not only define the type of job function(such as: `type="train"`), but also bind a `FunctionConfig` object to set the configuration of job function. OneFlow can manage our memory and device resources more conveniently.



 **Why use global_function?**

The beginning of the OneFlow's design is to solve the task of multiple devices in distributed training. In this situation, we set the global configuration by `global_funtion`. 



### 4.Layer and Operator

#### Layer

The layer concept in OneFlow is basically the same as the layer in tensorflow, pytorch and other popular deep learning framework. It is used to describe a layer in neural network. Like: convolution layer, batch normalization layer, fully connected layer and normalization layer. Layer can simplify the process of building neural network. For example you can use just few line of code to build Lenet:

```python
def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1', data_format='NCHW')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', data_format='NCHW')
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')
```

Layer is composed by many operators. For example: `layers.conv2d` is composed by  `conv2d` and `variable`.

#### Operator

Operator is the **basic calculation unit** in OneFlow. The calculation in above example is completed by operators. `flow.nn.max_pool2d` and `flow.reshape` all both operators.



### 5.Consistent/Mirrored View

OneFlow use two types of view:  **Mirrored View** and **Consistent View** to describe the distribution of data and model under distributed system. Different view is corresponding to different parallelism strategy.

Mirrored View comes from mirrors strategy of MPI distributed calculation. It is used to describe the mirrored model to multiple devices when using data parallelism.

Consistent View regards multi devices as one object in distributed environment. In this strategy, OneFlow will hide the detailed process for user and choose parallelism method in the optimal strategy (data/model/hybrid parallelism).

Basically:

When we set the mirrored view (`flow.scope.mirrored_view`), it means we can only use **data parallelism**. For example, we set four single device nodes in job function, the model will be copied and pasted to all devices, the data will be divided into four parts and send to each device.

When set consistent view (`flow.scope.consistent_view`), OneFlow **can choose data parallelism, model parallelism or hybrid parallelism.**



## Framework developing

### 1.Boxing

The module responsible for converting between different parallelism attributes of logical tensor. We called it  **Boxing**.

Such as: When the op of upstream and downstream has different parallelism feature (such as parallelism number different). OneFlow will use Boxing to automatic process the data conversion and transmission.



### 2.SBP

In fact, all the forward and backward operations in neural network can be calculated by matrix. In matrix calculation, there are operations such as split and broadcast. OneFlow also have same operations, we call it SBP. Of course, the SBP in OneFlow is not only matrix calculation. It also corresponding to divided data into different devices, broadcast and some other operations.

SBP means Split, Broadcast, Partial sum.

#### Split

In parallelism operations, tensor is divided into many sub tensor. Different operators allow tensor to be divided on different axis. Boxing will automatically handle the splitting of tensor on different axis under multiple operations. 

#### Broadcast

In parallelism operator calculation, the tensor will be broadcasted to many devices. Make the same tensor amount in each device.

#### Partial Sum

If an operator has distributive attribute, tensor will add parts of dimensions according to their attribute.

