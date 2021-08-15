# 反向传播与 optimizer

到目前为止，我们已经掌握如何使用 OneFlow [搭建模型](./04_build_network.md)、[加载数据](todo_dataset_dataloader.md)、[自动计算模型参数的梯度](./05_autograd.md)，将它们组合在一起，我们就可以利用反向传播算法训练模型。

在 [oneflow.optim](https://oneflow.readthedocs.io/en/master/optim.html) 中，有各类 `optimizer`，它们可以简化实现反向传播的代码。

本文将先介绍反向传播的基本概念，再介绍如何使用 `oneflow.optimz` 类。

## numpy 手工实现反向传播

为了读者更方便理解反向传播与自动求导的关系，在这里提供了一份仅用 numpy 实现的简单模型的训练过程：

```python
import numpy as np

ITER_COUNT = 500
LR = 0.01

# 前向传播
def forward(x, w):
    return np.matmul(x, w)

# 损失函数
def loss(y_pred, y):
    return (0.5*(y_pred-y)**2).sum()

# 计算梯度
def gradient(x, y, y_pred):
    return np.matmul(x.T, (y_pred-y))

if __name__ == "__main__":
    # 训练目标: Y = 2*X1 + 3*X2
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

```text
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
w:[[2.00000001]
 [2.99999999]]
```

注意我们选择的 loss 函数表达式为 $\sum \frac{1}{2}(y_{p} - y)^2$，因此 `loss` 对参数 `w`求梯度的代码为：

```python
def gradient(x, y, y_pred):
    return np.matmul(x.T, (y_pred-y))
```

更新参数采用的是 [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)：

```python
grad = gradient(x, y, y_pred)
w -= LR*grad
```

总结而言，训练中的一次完整迭代包括以下步骤：

1. 模型根据输入、参数，计算得出预测值 (`y_pred`)
2. 计算 loss，即预测值与标签之间的误差
3. 求 loss 对参数的梯度
4. 更新参数

其中 1~2 为正向传播过程；3~4为反向传播过程。

## 超参 Hyperparameters

超参数是有关模型训练设置的参数，可以影响到模型训练的效率和结果。如以上代码中的 `ITER_COUNT`、`LR` 就是超参数。

## 使用 `oneflow.optim` 中的优化器类

使用 `oneflow.optim` 中的优化器类进行反向传播会更简洁方便，接下来，我们展示如何使用。

首先，先准备好数据和模型，使用 Module 的一个方便之处就是，可以把超参放置在 Module 中便于管理。

```python
import oneflow as flow

x = flow.tensor([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=flow.float32)
y = flow.tensor([[8], [13], [26], [9]], dtype=flow.float32)


class MyLrModule(flow.nn.Module):
    def __init__(self, lr, iter_count):
        super().__init__()
        self.w = flow.nn.Parameter(flow.randn(2, 1, dtype=flow.float32))
        self.lr = lr
        self.iter_count = iter_count

    def forward(self, x):
        return flow.matmul(x, self.w)


model = MyLrModule(0.01, 500)
```

### loss 函数

然后，选择好 loss 函数，OneFlow 自带了多种 loss 函数，我们在这里选择 [MSELoss](https://oneflow.readthedocs.io/en/master/nn.html?highlight=mseloss#oneflow.nn.MSELoss)：

```python
loss = flow.nn.MSELoss(reduction='sum')
```
### 构造 optimizer
上文总结的训练中一次迭代里，反向传播的逻辑，都被封装在 optimizer 中。我们在此选择 [SGD](https://oneflow.readthedocs.io/en/master/optim.html?highlight=sgd#oneflow.optim.SGD) 优化器，你可以根据需要选择其它的优化器，如 [Adam](https://oneflow.readthedocs.io/en/master/optim.html?highlight=adam#oneflow.optim.Adam)、[AdamW](https://oneflow.readthedocs.io/en/master/optim.html?highlight=adamw#oneflow.optim.AdamW) 等。

```python
optimizer = flow.optim.SGD(model.parameters(), model.lr)
```

构造时 `optimizer`，将模型参数及 learning rate 传递给 `SGD`，在之后若调用 `optimizer.step()`，在其内部就会自动完成对模型参数求梯度、并按照 SGD 算法更新模型参数。

### 训练

以上准备完成后，可以开始训练：

```python
for i in range(0, model.iter_count):
    y_pred = model(x)
    l = loss(y_pred, y)
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{model.iter_count} loss:{l}")

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

print(f"\nw: {model.w}")
```

输出：
```text
50/500 loss:0.0015626397216692567
100/500 loss:8.896231520338915e-07
150/500 loss:5.038600647822022e-10
200/500 loss:9.094947017729282e-13
250/500 loss:9.094947017729282e-13
300/500 loss:9.094947017729282e-13
350/500 loss:9.094947017729282e-13
400/500 loss:9.094947017729282e-13
450/500 loss:9.094947017729282e-13
500/500 loss:9.094947017729282e-13

w: tensor([[2.0000],
        [3.0000]], dtype=oneflow.float32, grad_fn=<accumulate_grad>)
```
