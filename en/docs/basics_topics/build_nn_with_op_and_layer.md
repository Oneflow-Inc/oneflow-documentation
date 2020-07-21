# Use OneFlow build the neural network

The example in the previous section [ Recognition of MNIST handwritten numbers](http://183.81.182.202:8000/quick_start/lenet_mnist.html), we use flow.layers in the network layer and the operator in flow.nn build up a simple lenet network.Now we will using a simple neural network to introduce the core operator and layer in OneFlow network.

The code below is nerural network which is constructed by convolution layer, pooling layer and all connection layer. The figure show part of the operator of network and shape of result of operator. The shape of `data` is 100x1×28×28 and it is `Blob`. First of all `data` be the input of `conv2d` to take part in convolution, then the output will send to conv. After that conv1 will be the input of `max_pool2d` and keep run like this.(Note: Here isn't accurate, it is easy to understand such a description)

```python
def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')
```

![](imgs/lenet.png)

The code above have two type of elements, one is in brackets represent operation unit which include `op` and `layer`. Such as conv2d、dense、max_pool2d and etc... Another is in arrow represent data block definition. Such as `BlobDef`.

## Operator and layer
Operator is more common concept. It is the basic calculation unit in OneFlow. `reshape` and `nn.max_pool2d` is two type of operator. `layers.conv2d` and `layers.dense` is not operator. They are layer which construct by special operators.For example, `layers.conv2d` is construct by `conv2d` and variable. Layer can simplify the mapping process in front end. More details: [layers.py](https://github.com/Oneflow-Inc/oneflow/oneflow/python/ops/layers.py).

## Blob - `BlobDef` object
The statement we mention before have some part inaccurate. The network construction process and running process is separate in fact. The build process is OneFlow underlying calculation chart based on the description of the network connection relation and the process, but the real calculation happened at run time.What we discussing here is compile process, not run time. ，`data`、`conv1` and etc.. are all the object of [`BlobDef`](https://github.com/Oneflow-Inc/oneflow-documentation/docs/extended_topics/consistent_mirrored.md)(这里链接可能会出错). In OneFlow language it usually be called `Blob`. It play a role in fringe. In or output as a operator.`Blob` could be trade as data placeholder. During the compile, it does not have exact value but it contains all the information in this edge. For example, `shape` and `dtype`.In addition, `lbn` always be mentioned in OneFlow. It actually is abbreviation of `Logical Blob`. It is the `Blob` what user thinking when they constructing the network.Why we need to emphasise it is the `Blob` in user's mind? Because in run time step. The data of  `Blob` may distribute on different device. Such as 16 graphics card install on two servers. One of the card in have 1/16 information. Only construct them together can get the complete data of `Blob`.

When building network, could print of the information of `Blob`. For example, it can print data `shape` and `dtype`.
```
print(conv1.shape, conv1.dtype)
```

### Operator overloading
Operator overloading is define in BlobDef which means object of BlobDef could handle add, subtract, multiply and divide.

Like '+' in the following code:

```
output = output + fc2_biases
```
这句代码等价于：
```
output = flow.broadcast_add(output, fc2_biases)
```

## 总结
使用OneFlow进行神经网络搭建，需要OneFlow提供的算子或层作为计算单元，BlobDef作为算子和层的输入和输出，运算符重载帮助简化了部分语句。
