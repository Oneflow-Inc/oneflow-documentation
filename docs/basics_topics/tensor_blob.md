
编写和调用任务函数，是使用OneFlow的重点。而在学习编写任务函数前，需要了解tensor与blob类型。tensor是对多维数组的封装，用于存储计算数据；blob作为 **数据占位符** ，在定义任务函数，构建模型时不可或缺。
在本文中将学习：

* tensor与numpy的转换

* tensor与blob的区别与联系

* tensor与blob的应用场景

## tensor
tensor是OneFlow中的一种内置数据类型，可看作是封装好的 **多维数组** 。
在OneFlow进行训练时，内部都是对tensor对象中的具体数据进行计算。

### 将numpy数据转为tensor
实际上，我们 **不需要** 做额外的工作将numpy数据转为tensor。
在OneFlow中，numpy数据转为tensor对象对于用户而言是透明的。
用户可以直接在调用训练任务函数时，将numpy数据作为参数传递。
```python
import numpy as np
import oneflow as flow

@flow.function(flow.function_config())
def test_job(images=flow.FixedTensorDef((32, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((32, ), dtype=flow.int32)):
  # do something with images or labels
  return images, labels

if __name__ == '__main__':
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
  images, labels = test_job(images_in, labels_in).get()
  print("type1:", type(images_in), type(labels_in))
  print("type2:", type(images), type(labels))
  print("shape1:", images_in.shape, labels_in.shape)
  print("shape2:", images.shape, labels.shape)
```
运行以上代码，将得到以下结果：
```python
type1: <class 'numpy.ndarray'> <class 'numpy.ndarray'>
type2: <class 'oneflow.python.framework.local_blob.LocalFixedTensor'> <class 'oneflow.python.framework.local_blob.LocalFixedTensor'>
shape1: (32, 1, 28, 28) (32,)
shape2: (32, 1, 28, 28) (32,)
```
可见，在任务函数test_job的调用过程中，ndarray类型被自动转为了LocalFixedTensor类型。

### 将tensor对象转为numpy.ndarray
通过调用tensor对象的`ndarray`方法，可以获取tensor对象关联的数组。
将上例稍作修改，得到以下示例代码：
```python
if __name__ == '__main__':
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
  images, labels = test_job(images_in, labels_in).get()
  np_images = images.ndarray()
  np_labels = labels.ndarray()
  print("type:", type(np_images), type(np_labels))
  print(np.sum(np_images != images_in))
  print(np.sum(np_labels != labels_in))
```

输出结果为：

```python
type: <class 'numpy.ndarray'> <class 'numpy.ndarray'>
0
0
```

可见，我们成功将tensor对象转变为了ndarray对象。

## blob
OneFlow中的任务函数中，会构建模型网络。训练网络所需要的数据在构建时是 **没有的** ，在调用任务函数时，才会得到数据，并转化为tensor。
我们可以利用blob数据类型，在没有数据的前提下，起 **占位** 作用，构建网络模型。
如上文代码中：

```python
@flow.function(flow.function_config())
def test_job(images=flow.FixedTensorDef((32, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((32, ), dtype=flow.int32)):
  # do something with images or labels
  return images, labels
```

是通过FixedTensorDef创建了`images`和`labels`两个占位符，创建过程中指定了blob的shape以及元素数据类型。

### tensor与blob的关系

* tensor是OneFlow内置的数据类型，OneFlow使用tensor保存数据并进行计算

* blob是占位符，在定义模型，没有具体的数据时，通过使用blob来表示模型中各数据的关系时。

* 调用训练函数时，传递给训练函数的numpy数据会被自动转变为tensor，numpy数据、blob的shape、dtype等属性必须一致。

我们也可以借用编程语言中的概念，简单理解：

* blob是构建模型中的“ **形参** ”

* tensor是模型训练中的“ **实参** ”

### 创建blob的方法
使用`FixedTensorDef`将构造并返回一个`ConsistentBlob`对象，其原型如下：
```python
 FixedTensorDef(shape, 
            dtype=data_type_util.kFloat, 
            batch_axis=0,
            split_axis='same_with_batch_axis', 
            name=None)
```
如上文的示例代码，通过FixedTensorDef构造blob对象：
```python
images=flow.FixedTensorDef((32, 1, 28, 28), dtype=flow.float)
```

此外，还可以使用`MirroredTensorDef`，构造得到`MirroredBlob`对象。其原型如下
```python
MirroredTensorDef(shape,
            dtype=data_type_util.kFloat, 
            batch_axis=0, 
            name=None)
```
此外，还可以使用`oneflow.get_variable`对象创建blob，该方法一般用于任务函数之间的信息交互，我们会在 **get_variable** 文章中介绍。


