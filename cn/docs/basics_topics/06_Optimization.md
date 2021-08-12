# 反向传播

再上一章节中，我们讲到了求导对于模型训练的重要性。然而，在求导后，我们还需要让机器根据求出的导数来更新梯度。而这一过程，就被称之为反向传播。

## 利用自动求导手工实现反向传播

为了更方便理解自动求导的作用，我们在这里提供了一份用 numpy 纯手写的简单模型：

```python
import numpy as np

ITER_COUNT = 500
LR = 0.01

# 前向传播
def forward(x, w):
    return np.matmul(x, w)

# 损失函数 (return MSE 的导数)
def loss(y_pred, y):
    return (0.5*(y_pred-y)**2).sum()

# 计算导数
def gradient(x, y, y_pred):
    return np.matmul(x.T, (y_pred-y))

if __name__ == "__main__":
    # 训练目的: Y = 2*X1 + 3*X2
    x = np.array([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=np.float32)
    y = np.array([[8], [13], [26], [9]], dtype=np.float32)

    w = np.random.rand(2, 1)
    # 训练循环
    for i in range(0, ITER_COUNT):
        y_pred = forward(x, w)
        l = loss(y_pred, y)
        if (i+1) % 50 == 0: print(f"{i+1}/{500} loss:{l}")

        grad = gradient(x, y, y_pred)
        w -= LR*grad

    print(f"w:{w}")
```

输出：

```shell
50/500 loss:0.0012162785263114685
100/500 loss:3.11160142374838e-05
150/500 loss:7.960399867959713e-07
200/500 loss:2.0365065260521826e-08
250/500 loss:5.209988065278517e-10
300/500 loss:1.3328695632996161e-11
350/500 loss:3.4098758995283524e-13
400/500 loss:8.723474589862032e-15
450/500 loss:2.231723694177745e-16
500/500 loss:5.7094113647001346e-18
w:[[2.]
 [3.]]
```

可以看到，以上代码的主要目的为训练模型去寻找公式中2，3两个参数 (`Y = 2*X1 + 3*X2`)。具体训练过程及步骤在 **2.1 快速上手** 中已有详细介绍。

在 **2.1 快速上手** 的反向传播时，我们与用了一个简单的 `.backward` 就解决了更新导数的问题。但为了更好的诠释 `.backward` 的过程，我们这里手写了反向传播。`loss() `的返回结果`(0.5*(y_pred-y)**2).sum()` 为 MSE 损失函数的导数。关键在于：

```python
def gradient(x, y, y_pred):
    return np.matmul(x.T, (y_pred-y))
```

以及训练循环中的：

```python
grad = gradient(x, y, y_pred)
w -= LR*grad
```

可以看到，所谓更新参数，就是简单的导数乘以学习率。

## 利用 `flow.optim` 中已有的类进行反向传播

上面手写的模型似乎很麻烦。我们不但要对其导数公式，还需要手写更新过程。在训练稍稍复杂一点的模型的话，工作量会大大提高 (激活函数等等都需要手写)。下面是我们用 oneflow 写出的训练 `Y = 2*X1 + 3*X2` 的模型。

```python
import oneflow as flow

class MyLrModule(flow.nn.Module):
    def __init__(self, lr, iter_count):
        super().__init__()
        self.w = flow.nn.Parameter(flow.tensor([[1], [1]],dtype=flow.float32))
        self.lr = lr
        self.iter_count = iter_count

    def forward(self, x):
        return flow.matmul(x, self.w)

if __name__ == "__main__":
    # train data: Y = 2*X1 + 3*X2
    x = flow.tensor([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=flow.float32)
    y = flow.tensor([[8], [13], [26], [9]], dtype=flow.float32)

    model = MyLrModule(0.01, 500)
    loss = flow.nn.MSELoss(reduction='sum')
    optimizer = flow.optim.SGD(model.parameters(), model.lr)

    for i in range(0, model.iter_count):
        y_pred = model(x)
        l = loss(y_pred, y)
        if (i+1) % 50 == 0: print(f"{i+1}/{model.iter_count} loss:{l}")

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"w: {model.w}")
```

```shell
50/500 loss:tensor(0.0004, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
100/500 loss:tensor(2.2268e-07, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
150/500 loss:tensor(1.3461e-10, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
200/500 loss:tensor(3.8654e-12, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
250/500 loss:tensor(3.8654e-12, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
300/500 loss:tensor(3.8654e-12, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
350/500 loss:tensor(3.8654e-12, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
400/500 loss:tensor(3.8654e-12, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
450/500 loss:tensor(3.8654e-12, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
500/500 loss:tensor(3.8654e-12, dtype=oneflow.float32, grad_fn=<reduce_sum_backward>)
w: tensor([[2.],
        [3.]], dtype=oneflow.float32, grad_fn=<accumulate_grad>)
```

可以看到，我们不需要再手写损失函数以及其导数。oneflow.nn 中含有大量的损失函数供用户使用。其次，在实现反向传播时，用户只需在训练循环最后加入：

```python
l.backward()
optimizer.step() #更新参数
optimizer.zero_grad() #清除导数
```

即可。

