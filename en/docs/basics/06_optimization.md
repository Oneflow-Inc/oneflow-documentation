# BACKPROPAGATION AND OPTIMIZER

So far, we have learned how to use OneFlow to [Dataset and DataLoader](./03_dataset_dataloader.md), [Build Models](./04_build_network.md),[Autograd](./05_autograd.md), and combine them so that we can train models by using backpropagation algorithms.

In [oneflow.optim](https://oneflow.readthedocs.io/en/v0.8.1/optim.html), there are various `optimizer`s that simplify the code of back propagation.

This article will first introduce the basic concepts of back propagation and then show you how to use the `oneflow.optim` class.

## Backpropagation by Numpy Code

In order to make it easier for readers to understand the relationship between backpropagation and autograd, a training process of a simple model implemented with numpy is provided here:

```python
import numpy as np

ITER_COUNT = 500
LR = 0.01

# Forward propagation
def forward(x, w):
    return np.matmul(x, w)


# Loss function
def loss(y_pred, y):
    return ((y_pred - y) ** 2).sum()


# Calculate gradient
def gradient(x, y, y_pred):
    return np.matmul(x.T, 2 * (y_pred - y))


if __name__ == "__main__":
    # Train: Y = 2*X1 + 3*X2
    x = np.array([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=np.float32)
    y = np.array([[8], [13], [26], [9]], dtype=np.float32)

    w = np.array([[2], [1]], dtype=np.float32)
    # Training cycle
    for i in range(0, ITER_COUNT):
        y_pred = forward(x, w)
        l = loss(y_pred, y)
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{500} loss:{l}")

        grad = gradient(x, y, y_pred)
        w -= LR * grad

    print(f"w:{w}")
```

output：

```text
50/500 loss:0.0034512376878410578
100/500 loss:1.965487399502308e-06
150/500 loss:1.05524122773204e-09
200/500 loss:3.865352482534945e-12
250/500 loss:3.865352482534945e-12
300/500 loss:3.865352482534945e-12
350/500 loss:3.865352482534945e-12
400/500 loss:3.865352482534945e-12
450/500 loss:3.865352482534945e-12
500/500 loss:3.865352482534945e-12
w:[[2.000001 ]
 [2.9999993]]
```

Note that the loss function expression we selected is $\sum (y_{p} - y)^2$, so the code for gradient of `loss` to parameter `w` is：

```python
def gradient(x, y, y_pred):
    return np.matmul(x.T, 2 * (y_pred - y))
```

[SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) is used to update parameters：

```python
grad = gradient(x, y, y_pred)
w -= LR*grad
```

In summary, a complete iteration in the training includes the following steps:

1. The model calculates the predicted value based on the input and parameters (`y_pred`)
2. Calculate loss, which is the error between the predicted value and the label
3. Calculate the gradient of loss to parameter
4. Update parameter(s)

1 and 2 are forward propagation process; 3 and 4 are back propagation process.

## Hyperparameters

Hyperparameters are parameters related to model training settings, which can affect the efficiency and results of model training.As in the above code `ITER_COUNT`,`LR` are hyperparameters.

## Using the optimizer class in `oneflow.optim`

Using the optimizer class in `oneflow.optim` for back propagation will be more concise.

First, prepare the data and model. The convenience of using Module is that you can place the hyperparameters in Module for management.

```python
import oneflow as flow

x = flow.tensor([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=flow.float32)
y = flow.tensor([[8], [13], [26], [9]], dtype=flow.float32)


class MyLrModule(flow.nn.Module):
    def __init__(self, lr, iter_count):
        super().__init__()
        self.w = flow.nn.Parameter(flow.tensor([[2], [1]], dtype=flow.float32))
        self.lr = lr
        self.iter_count = iter_count

    def forward(self, x):
        return flow.matmul(x, self.w)


model = MyLrModule(0.01, 500)
```

### Loss function

Then, select the loss function. OneFlow comes with a variety of loss functions. We choose [MSELoss](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.MSELoss.html) here：

```python
loss = flow.nn.MSELoss(reduction="sum")
```

### Construct Optimizer

The logic of back propagation is wrapped in optimizer. We choose [SGD](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.optim.SGD.html) here, You can choose other optimization algorithms as needed, such as [Adam](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.optim.Adam.html) and[AdamW](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.optim.AdamW.html) .

```python
optimizer = flow.optim.SGD(model.parameters(), model.lr)
```

When the `optimizer` is constructed, the model parameters and learning rate are given to `SGD`. Then the `optimizer.step()` is called, and it automatically completes the gradient of the model parameters and updates the model parameters according to the SGD algorithm.

### Train

When the above preparations are completed, we can start training:

```python
for i in range(0, model.iter_count):
    y_pred = model(x)
    l = loss(y_pred, y)
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{model.iter_count} loss:{l.numpy()}")

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

print(f"\nw: {model.w}")
```

output：

```text
50/500 loss:0.003451163647696376
100/500 loss:1.965773662959691e-06
150/500 loss:1.103217073250562e-09
200/500 loss:3.865352482534945e-12
250/500 loss:3.865352482534945e-12
300/500 loss:3.865352482534945e-12
350/500 loss:3.865352482534945e-12
400/500 loss:3.865352482534945e-12
450/500 loss:3.865352482534945e-12
500/500 loss:3.865352482534945e-12

w: tensor([[2.],
        [3.]], dtype=oneflow.float32, grad_fn=<accumulate_grad>)
```
