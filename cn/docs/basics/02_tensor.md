# Tensor 张量

神经网络中的数据，都存放在 Tensor 中，Tensor 类似多维数组或者数学上的矩阵。OneFlow 提供了很多用于操作 Tensor 的算子，Tensor 与算子一起构成神经网络。

Tensor 有别于普通的多维数组的地方是：除了可以运行在 CPU 上外，它还可以运行在 其它 AI 芯片（如 NVIDIA GPU）上，因此可以提高运算速度。此外，OneFlow 还为张量提供了 [自动求导](./05_autograd.md) 的功能。

```python
import oneflow as flow
import numpy as np
```

## 创建 Tensor
有多种方法创建 Tensor，包括：

- 直接从数据创建
- 通过 Numpy 数组创建
- 使用算子创建

### 直接从数据创建
可以直接从数据创建 Tensor：

```python
x1 = flow.tensor([[1, 2], [3, 4]])
x2 = flow.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x1)
print(x2)
```

可以看到创建的 `x1`、`x2` Tensor，它们的类型分别是 `int64` 和 `float32`。

```text
tensor([[1, 2],
        [3, 4]], dtype=oneflow.int64)
tensor([[1., 2.],
        [3., 4.]], dtype=oneflow.float32)
```

### 通过 Numpy 数组创建

Tensor 可以通过 Numpy 数组创建，只需要在创建 Tensor 对象时，将 Numpy 数组作为参数传递即可。

```python
x3 = flow.tensor(np.ones((2,3)))
x4 = flow.tensor(np.random.rand(2,3))
print(x3)
print(x4)
```

```text
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=oneflow.float64)
tensor([[0.6213, 0.6142, 0.1592],
        [0.5539, 0.8453, 0.8576]], dtype=oneflow.float64)
```


### 通过算子创建

OneFlow 中还提供了一些算子，可以通过它们创建 Tensor。比如 [ones](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.ones#oneflow.ones)、 [zeros](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.zeros#oneflow.zeros),、[eye](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.eye#oneflow.eye)，它们分别创建全为1的张量、全为0的张量和单位张量。

```python
x5 = flow.ones(2, 3)
x6 = flow.zeros(2, 3)
x7 = flow.eye(3)
print(x5)
print(x6)
print(x7)
```

```text
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=oneflow.float32)
tensor([[0., 0., 0.],
        [0., 0., 0.]], dtype=oneflow.float32)
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]], dtype=oneflow.float32)
```

[randn](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.randn#oneflow.randn) 方法可以创建随机化的张量：

```python
x8 = flow.randn(2,3)
print(x8)
```

## `Tensor` 与 `tensor` 的区别
细心的用户会发现，OneFlow 中有 [oneflow.Tensor](https://oneflow.readthedocs.io/en/master/tensor.html?highlight=oneflow.Tensor#oneflow.Tensor) 和 [oneflow.tensor](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.Tensor#oneflow.tensor) 两个接口，它们都能用来创建张量。那么它们有什么区别呢？

简单而言，大写的 `Tensor` 数据类型默认限定为 `float32`，而小写的 `tensor` 的数据类型可以随着创建时的数据改变。以下代码展示了两者这方面的区别：

```python
print(flow.Tensor([1, 2, 3]))
print(flow.tensor([1, 2, 3]))
print(flow.tensor([1.0, 2.0, 3.0]))
```

数据结果为：

```text
tensor([1., 2., 3.], dtype=oneflow.float32)
tensor([1, 2, 3], dtype=oneflow.int64)
tensor([1., 2., 3.], dtype=oneflow.float32)
```

此外，大写的 `Tensor` 可以在创建时不指定具体数据：

```python
x9 = flow.Tensor(2, 3)
print(x9.shape)
```

```text
flow.Size([2, 3])
```

因此，如果在创建张量的同时不想指定数据，那么常常用 `oneflow.Tensor`，否则，应该使用 `oneflow.tensor`。

## Tensor 的属性

Tensor 的 `shape`、`dtype`、`device` 属性分别描述了 Tensor 的形状、数据类型和所在的设备类型。

```python
x9 = flow.randn(1,4)
print(x9.shape)
print(x9.dtype)
print(x9.device)
```

输出结果分别展示了张量的形状、数据类型和所处的设备（第0号 CPU 上，之所以有编号，是因为 OneFlow 很方便自然地支持分布式，可参考 [Consistent Tensor](../parallelism/03_consistent_tensor.md)）

```text
flow.Size([1, 4])
oneflow.float32
cpu:0
```

可以通过 [reshape](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.reshape#oneflow.reshape) 方法改变 Tensor 的形状，用 [to](https://oneflow.readthedocs.io/en/master/tensor.html?highlight=Tensor.to#oneflow.Tensor.to) 方法改变 Tensor 的数据类型和所处设备：

```
x10 = x9.reshape(2, 2)
x11 = x10.to(dtype=flow.int32, device=flow.device("cuda"))
print(x10.shape)
print(x11.dtype, x11.device)
```

```text
flow.Size([2, 2])
oneflow.int32 cuda:0
```

## 操作 Tensor 的常见算子

OneFlow 中提供了大量的算子，对 Tensor 进行操作，它们大多在 [oneflow](https://oneflow.readthedocs.io/en/master/oneflow.html)、[oneflow.Tensor](https://oneflow.readthedocs.io/en/master/tensor.html)、[oneflow.nn](https://oneflow.readthedocs.io/en/master/nn.html)、[oneflow.nn.functional](https://oneflow.readthedocs.io/en/master/functional.html)这几个名称空间下。

OneFlow 中的 Tensor，与 Numpy 数组一样易用。比如，支持与 Numpy 类似的切片操作：

```python
tensor = flow.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```

```text
First row:  tensor([1., 1., 1., 1.], dtype=oneflow.float32)
First column:  tensor([1., 1., 1., 1.], dtype=oneflow.float32)
Last column: tensor([1., 1., 1., 1.], dtype=oneflow.float32)
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]], dtype=oneflow.float32)
```

此外，OneFlow 中还有很多其它操作，如算数相关操作的 [add](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.add#oneflow.add)、[sub](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.sub#oneflow.sub)、[mul](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.mul#oneflow.mul)、[div](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.div#oneflow.div)等；位置相关操作的 [scatter](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.scatter#oneflow.scatter)、[gather](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.gather#oneflow.gather)、[gather_nd](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.gather_nd#oneflow.gather_nd)等；以及激活函数、卷积等（[relu](https://oneflow.readthedocs.io/en/master/functional.html?highlight=oneflow.relu#oneflow.nn.functional.relu)、[conv2d](https://oneflow.readthedocs.io/en/master/functional.html?highlight=oneflow.conv2d#oneflow.nn.functional.conv2d)），点击它们的链接可以查看更详细的 API 说明，并找到更多的其它算子。
