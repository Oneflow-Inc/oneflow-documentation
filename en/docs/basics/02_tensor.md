# TENSORS

The data in the neural network is stored in tensors, which are similar to arrays and mathematical matrices. OneFlow provides a series of operators on tensors. Tensors, together with operators, build up a neural network.

Tensors differ from regular multidimensional arrays in that they can run on AI chips (such as the Nvidia GPU) and CPU, thus increasing computing speed. In addition, OneFlow provides [Autograd](./05_autograd.md) function, which supports automatic differentiation.

```python
import oneflow as flow
import numpy as np
```

## Creating Tensors

There are several ways to create tensors, including:

- Directly from data
- From a NumPy array
- By an operator

### Directly from data

Tensors can be created directly from data:

```python
x1 = flow.tensor([[1, 2], [3, 4]])
x2 = flow.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x1)
print(x2)
```

Out:

```text
tensor([[1, 2],
        [3, 4]], dtype=oneflow.int64)
tensor([[1., 2.],
        [3., 4.]], dtype=oneflow.float32)
```

We can see that the tensor `x1` and `x2` are created, whose data types are `int64` and `float32`, respectively.

### From a NumPy array

Tensors can be created from NumPy arrays by passing the NumPy array as a parameter when the tensor object is constructed.

```python
x3 = flow.tensor(np.ones((2,3)))
x4 = flow.tensor(np.random.rand(2,3))
print(x3)
print(x4)
```

Out:

```text
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=oneflow.float64)
tensor([[0.6213, 0.6142, 0.1592],
        [0.5539, 0.8453, 0.8576]], dtype=oneflow.float64)
```

### By an operator

There are also many operators available in OneFlow that can be used to create tensors. For example, [ones](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.ones#oneflow.ones), [zeros](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.zeros#oneflow.zeros) and [eye](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.eye#oneflow.eye), which create the all-ones tensor, zero tensor, and identity tensor, respectively.

```python
x5 = flow.ones(2, 3)
x6 = flow.zeros(2, 3)
x7 = flow.eye(3)
print(x5)
print(x6)
print(x7)
```

Out:

```text
tensor([[1., 1., 1.],
        [1., 1., 1.]], dtype=oneflow.float32)
tensor([[0., 0., 0.],
        [0., 0., 0.]], dtype=oneflow.float32)
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]], dtype=oneflow.float32)
```

The [randn](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.randn#oneflow.randn) method creates a random tensor:

```python
x8 = flow.randn(2,3)
```

## Difference Between `Tensor` and `tensor`

There are two interfaces ([oneflow.Tensor](https://oneflow.readthedocs.io/en/master/tensor.html?highlight=oneflow.Tensor#oneflow.Tensor) and [oneflow.tensor](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.Tensor#oneflow.tensor)) in OneFlow, both of which can be used to create tensors. What’s the difference between them?

Briefly speaking, the data type of `oneflow.Tensor` is limited to `float32` by default, while the data type of `oneflow.tensor` can be changed when the data is created. The following code illustrates the difference:

```python
print(flow.Tensor([1, 2, 3]))
print(flow.tensor([1, 2, 3]))
print(flow.tensor([1.0, 2.0, 3.0]))
```

Out:

```text
tensor([1., 2., 3.], dtype=oneflow.float32)
tensor([1, 2, 3], dtype=oneflow.int64)
tensor([1., 2., 3.], dtype=oneflow.float32)
```

Besides, `oneflow.Tensor` can be created without specific data:

```python
x9 = flow.Tensor(2, 3)
print(x9.shape)
```

Out:

```text
flow.Size([2, 3])
```

Therefore, use `oneflow.Tensor` to create a tensor if you do not want to specify an explicit value, otherwise, you should use `oneflow.tensor`.

## Attributes of a Tensor

The `shape`, `dtype`, and `device` attributes of a tensor describe its shape, data type, and device type respectively.

```python
x9 = flow.randn(1,4)
print(x9.shape)
print(x9.dtype)
print(x9.device)
```

Out:

```text
flow.Size([1, 4])
oneflow.float32
cpu:0
```

The output shows the shape, the data type, and the device (on CPU No. 0, CPUs were numbered because OneFlow naturally supports distribution, see [Global Tensor](../parallelism/03_global_tensor.md)).

The shape of the tensor can be changed by the [reshape](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.reshape#oneflow.reshape) method, and the data type and device of the tensor can be changed by the [to](https://oneflow.readthedocs.io/en/master/tensor.html?highlight=Tensor.to#oneflow.Tensor.to) method:

```python
x10 = x9.reshape(2, 2)
x11 = x10.to(dtype=flow.int32, device=flow.device("cuda"))
print(x10.shape)
print(x11.dtype, x11.device)
```

Out:

```text
oneflow.Size([2, 2])
oneflow.int32 cuda:0
```

## Operations on Tensors

A large number of operators are provided in OneFlow, most of which are in the namespaces of [oneflow](https://oneflow.readthedocs.io/en/master/oneflow.html), [oneflow.nn](https://oneflow.readthedocs.io/en/master/nn.html), and [oneflow.nn.functional](https://oneflow.readthedocs.io/en/master/functional.html).

Tensors in OneFlow are as easy to use as the NumPy arrays. For example, slicing in NumPy style is supported:

```python
tensor = flow.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```

Out:

```text
First row:  tensor([1., 1., 1., 1.], dtype=oneflow.float32)
First column:  tensor([1., 1., 1., 1.], dtype=oneflow.float32)
Last column: tensor([1., 1., 1., 1.], dtype=oneflow.float32)
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]], dtype=oneflow.float32)
```

In addition, there are many other operations in OneFlow, such as [add](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.add#oneflow.add), [sub](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.sub#oneflow.sub), [mul](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.mul#oneflow.mul), [div](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.div#oneflow.div) for arithmetic operations; [scatter](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.scatter#oneflow.scatter), [gather](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.gather#oneflow.gather), [gather_nd](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.gather_nd#oneflow.gather_nd) for positional operations; and activation functions ([relu](https://oneflow.readthedocs.io/en/master/functional.html?highlight=oneflow.relu#oneflow.nn.functional.relu)), convolution functions ([conv2d](https://oneflow.readthedocs.io/en/master/functional.html?highlight=oneflow.conv2d#oneflow.nn.functional.conv2d)), etc. Click on their links to see detailed API description and find out more about other operators.