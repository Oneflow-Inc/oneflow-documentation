# 自动求导

神经网络的训练过程离不开 **反向传播算法**，在反向传播过程中，需要获取 loss 函数对模型参数的梯度，用于更新参数。

OneFlow 提供了自动求导机制，可自动计算神经网络中参数的梯度。

本文将先介绍计算图的基本概念，它有利于理解 OneFlow 自动求导的常见设置及限制，再介绍 OneFlow 中与自动求导有关的常见接口。

## 计算图

张量与算子，共同组成计算图，如以下代码：

```python
import oneflow as flow

def loss(y_pred, y):
    return flow.sum(1/2*(y_pred-y)**2)

x = flow.ones(1, 5)  # 输入
w = flow.randn(5, 3, requires_grad=True)
b = flow.randn(1, 3, requires_grad=True)
z = flow.matmul(x, w) + b

y = flow.zeros(1, 3)  # label
l = loss(z,y)
```

它对应的计算图如下：

![todo](https://todo)

计算图中，像 `x`、`w`、`b`、`y` 这种只有输出，没有输入的节点称为 **叶子节点**；向 `loss` 这种只有输入没有输出的节点，称为 **根**。

反向传播过程中，需要求得 `l` 对 `w`、`b` 的梯度，以更新这两个模型参数。因此，我们在创建它们时，设置 `requires_grad` 为 `True`。


## 自动求导

在反向传播的过程中，需要得到 `l` 分别对 `w`、`b` 的梯度 $\frac{\partial l}{\partial w}$ 和 $\frac{\partial l}{\partial b}$。我们只需要对 `l` 调用 `backward()` 方法，然后 OneFlow 就会自动计算梯度，并且存放到 `w` 与 `b` 的 `grad` 成员中。

```python
l.backwad()
print(w.grad)
print(b.grad)
```

```text
tensor([[0.9397, 2.5428, 2.5377],
        [0.9397, 2.5428, 2.5377],
        [0.9397, 2.5428, 2.5377],
        [0.9397, 2.5428, 2.5377],
        [0.9397, 2.5428, 2.5377]], dtype=oneflow.float32)
tensor([[0.9397, 2.5428, 2.5377]], dtype=oneflow.float32)
```

### 停止对某个 Tensor 求梯度

### 对一个计算图多次 `backward()`

### 对非叶子节点求梯度

## 输出为 Tensor 时如何求导
- scalar implicity only




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
