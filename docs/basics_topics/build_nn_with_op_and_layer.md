# 使用OneFlow搭建神经网络

下面的代码是一个主要由卷积、池化和全连接组成的网络，后面的图示意了该网络的算子（op）和算子输出的形状。
`data`是维度是100x1×28×28的`Blob`，`data`首先作为`conv2d`的输入参与卷积计算，计算结果传给conv1，然后conv1作为输入传给`max_pool2d`，依次类推。（注：这里的说法不准确，只是方便理解这么描述）

```
def lenet(data):
  initializer = flow.truncated_normal(0.1)
  conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu,
                             kernel_initializer=initializer)
  pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', data_format='NCHW')
  conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu,
                             kernel_initializer=initializer)
  pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', data_format='NCHW')
  reshape = flow.reshape(pool2, [pool2.shape[0], -1])
  hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
  return flow.layers.dense(hidden, 10, kernel_initializer=initializer)
```
![image](E07492F444294C4DA1AB4DB81C606326)

上图中有两类元素，一类是方框代表的运算单元，包括`op`和`layer`两类，比如conv2d、dense、max_pool2d等；一类是箭头代表的数据块定义（`BlobDef`）。

## 算子（op）和层（layer）
算子（op）是比较常用的一种概念，是OneFlow中基本的运算单元，前面的`reshape`和`nn.max_pool2d`就是两种算子。
`layers.conv2d`和`layers.dense`不是算子，它们是由算子组合成的特定的

运算层（layer）。比如`layers.conv2d`其实是由`conv2d`算子和variable算子组成的，层的存在简化了前端的构图过程，详细参考[layers.py](oneflow/python/ops/layers.py)。

## BlobDef
前面提到过有说法不准确的地方，网络的构建过程和运行过程其实是分开的，构建过程是OneFlow底层根据描述的网络连接关系构建真正计算图的过程，真正的计算是在运行时发生的。我们这里讨论的是构图过程，不是计算过程，`data`、`conv1`等都是[`BlobDef`对象](TODO)。

搭建网络时可以打印`BlobDef`对象的属性，比如要打印形状`shape`和数据类型`dtype`可以
```
print(conv1.shape, conv1.dtype)
```

### 运算符重载
BlobDef中定义了运算符重载，也就是说，BlobDef对象之间可以进行加减乘除等操作，例如下面这句代码中的加号：
```
output = output + fc2_biases
```
这句代码等价于：
```
output = flow.broadcast_add(output, fc2_biases)
```

## 总结
使用OneFlow进行神经网络搭建，需要OneFlow提供的算子或层作为计算单元，BlobDef作为算子和层的输入和输出，运算符重载帮助简化了部分语句。
