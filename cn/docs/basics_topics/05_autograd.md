# 自动求导

训练完毕和未训练的模型最大的区别就是权重 (weight)。举一个简单的例子：

<img src="https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-documentation/desmos-graph%20(1).png" style="zoom:40%;"/>

如上图所示，与其说我们要让模型识别出青蛙和蚯蚓的区别，其实就是让模型保存 y = x 这条线。而决定这条线最大的因素就是其斜率，也就是权重，同时这条线的权重也是这条线的导数（注意：导数和权重是两个概念，权重存在于正向传播，而导数存在于反向传播）。有正确的权重，这条线就可以完美的区分蚯蚓和青蛙；相反，若没有权重，模型几乎不可能做到正确分类。但是，当面对复杂的分类工程时，坐标系往往有成千上万个维度。也就是说，节点数会大量增加 （每一个节点储存一个权重）。我们不可能对每个节点一一求导。也就是为什么自动求导，是训练神经网络必需品。

实现自动求导的过程就是反向传播。以青蛙和蚯蚓为例，若分类线没有正确分类，我们可以指定一个损失函数 (例如常见的MSE Mean Squared Error) 让电脑意识到与正确答案的差距，并根据差距更新权重。重复以上过程以达到正确的分类。

## `backward()` 自动求导

##### `requires_grad` 与 `backward()`

让我们先看一段简单的代码：

```python
import oneflow as flow
import numpy as np
# 建立一个简单的张量，并做简单的变换
x =  flow.tensor([1.0,2.0,3.0], requires_grad = True)
print(x)
y = x*x
# 用 MSE 来计算 x 与 y 的差距
loss = flow.nn.MSELoss()
out = loss(x, y)
print(out)
# 反向传播，计算导数
out.backward()
print(x.grad)
```

这段代码不难理解。简单来说，我们建立了两个矩阵，并通过 MSE 损失函数来推算两个矩阵的差距，并计算 loss'(x)。只有推导出导数值，机器才能相对应的对每个结点上的权重做出调整。

当建立一个矩阵时，`requires_grad ` 默认为 `false`，也就是为什么当我们需要反向传播时，要在张量后面设置`requires_grad`。而`.backward()` 的作用就是让机器自动进行求导，也就是反向传播 (backward pass)。



### 为什么只能是针对标量求导

若你将上面的代码在本地跑过一遍的话，你可能会发现，loss的输出是比较奇怪的：

```python
out = loss(x,y)
print(out)
```

输出：

```shell
tensor(13.3333, dtype=oneflow.float32, grad_fn=<scalar_mul_backward>)
```

重点在后面的 `grad_fn=<scalar_mul_backward>` 。scalar 的意思是标量，也就是说，在反向传播的过程中，我们只能针对标量 (loss) 求导，而不是一个任意一个向量 (vector, 也可以说是所有非 single-element 的 tensor)。

其实这一法则也很好理解。在反向传播过程中，我们要计算的是 loss 对每一个节点上权重的导数，并根据导数更新权重。而若 loss 是一个向量的话，loss 就不是一个值，导数也就不复存在 (标量对向量求导)。

### 为什么默认保留叶子节点的导数

在 **2.1 快速上手** 中，我们的训练循环结尾会有一步 `.zero_grad()` 的过程。这是因为，所有叶子节点的导数会被保留。每次在我们进行 `.backward()` 操作后，梯度需要被保留以用于更新参数 (也就是后文中的`optimizer.step()`)。

## `grad` 成员及相关操作

### detach

detach 可以让 oneflow 停止对 `requires_grad=True` 的元素进行求导跟踪。例如：

```python
# 建立一个简单的张量，并做简单的变换
x =  flow.tensor([1.0,2.0,3.0], requires_grad = True)
print(x)
y = x*x
z = x**3
a = y+z
# 用 MSE 来计算 x 与 a 的差距
loss = flow.nn.MSELoss()
out = loss(x, a)
print(out)

# 反向传播，计算导数
out.backward()
print(x.grad)
```

其输出为

```shell
tensor([1., 2., 3.], dtype=oneflow.float32, requires_grad=True)
tensor(396.6667, dtype=oneflow.float32, grad_fn=<scalar_mul_backward>)
tensor([  2.6667, 100.    , 704.    ], dtype=oneflow.float32)
```

但若我们将 `z = x**3` 替换成 `z = x.detach()**3`, 输出会变为：

```shell
tensor([1., 2., 3.], dtype=oneflow.float32, requires_grad=True)
tensor(396.6667, dtype=oneflow.float32, grad_fn=<scalar_mul_backward>)
tensor([  0.6667,  20.    , 110.    ], dtype=oneflow.float32)
```

原因很简单，加入 detach 操作后，oneflow 就不会在反向过程中对 z 所参与的节点求导，从而导致 `x.grad` 值的变化。

在实际运用中，假设有两个模型，第一个模型的输出为第二个模型的输入。若你只想训练第二个模型，那么只需在第一个模型后面加入 detach 操作就可以达到目的。

### retain_graph 

上面讲到，非叶子节点的梯度会在更新完被释放。但如果我们想查看被释放的梯度呢？只需要在 autograd 函数中加上 `requires_grad=True` 即可

比如：

```python
import oneflow as flow

x = flow.tensor(2., requires_grad=True)
y = flow.tensor(3., requires_grad=True)
z = x*y 

x_grad = flow.autograd.grad(z,x,retain_graph=True)
y_grad = flow.autograd.grad(z,y)

print(x_grad[0],y_grad[0])
```

输出：

```shell
tensor(3., dtype=oneflow.float32) tensor(2., dtype=oneflow.float32)
```

若没有加入 `requires_grad=True` , oneflow 会默认报错，因为梯度已被释放。
