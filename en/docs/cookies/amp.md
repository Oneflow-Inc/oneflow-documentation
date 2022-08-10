#  Automatic Mixed Precision Training

## Introduction to AMP 

When we train deep learning models, we typically use 32-bit single-precision floating point (FP32), while **AMP (Automatic Mixed Precision)** is a technique that allows both FP32 and FP16 to be used when training models. This can make the memory usage less and the computation faster when training the model. But because the numerical range of FP16 is smaller than that of FP32, it is more prone to numerical overflow problems, and there may be some errors. But lots of practice has proved that many deep learning models can be trained with this technique without loss of accuracy.

##  Example of using AMP

First, we define a simple model, loss function and optimizer in exactly the same way as before.

```python
import oneflow as flow
import oneflow.nn as nn

DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
print("Using {} device".format(DEVICE))

model = nn.Sequential(
    nn.Linear(256, 128), 
    nn.ReLU(),
    nn.Linear(128, 10)
)
model = model.to(DEVICE)
model.train()

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)
```

If you want to enable AMP mode, just add `self.config.enable_amp(True)` to the model [nn.Graph](../basics/08_nn_graph.md). The details of this API is at: [enable_amp](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.graph.graph_config.GraphConfig.enable_amp.html).

```python
class CustomGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)
        self.config.enable_amp(True)    # 开启 AMP 模式

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return y_pred
```

Then, you can start training and other operations as usual.

```python
graph_model = CustomGraph()

for _ in range(100):
    x = flow.randn(128, 256).to(DEVICE)
    y = flow.ones(128, 1, dtype=flow.int64).to(DEVICE)

    graph_model(x, y)
```

## Gradient Scaling

**Gradient Scaling** is a method for solving the problem that FP16 is prone to numerical overflow. The basic principle is to use a scale factor to scale the loss and gradient in the process of backpropagation to change the magnitude of its value, thereby mitigate numerical overflow problems as much as possible.

OneFlow provides `GradScaler` to use Gradient Scaling in AMP mode. You only need to instantiate a `GradScaler` object in the `__init__` method of the nn.Graph model, and then specify it through the interface [set_grad_scaler](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.Graph.set_grad_scaler.html). nn.Graph will automatically manage the whole process of Gradient Scaling. Taking the `CustomGraph` above as an example, you need to add the following code to its `__init__` method:

```python
grad_scaler = flow.amp.GradScaler(
    init_scale=2**12,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=1000,
)
self.set_grad_scaler(grad_scaler)
```

The calculation process of the scale factor and the meaning of the GradScaler parameters are as follows:

The size of the scale factor is dynamically estimated in the iterative update (the initial value is specified by `init_scale`). In order to reduce the numerical underflow as much as possible, the scale factor should be larger; but if it is too large, FP16 is prone to numerical overflow , resulting in an inf or NaN. The process of dynamic estimation is to increase the scale factor as much as possible without occuring inf or NaN. At each iteration, it will check whether there is a gradient of inf or NaN:

1. If there is: this weight update will be ignored and the scale factor will be reduced (multiplied by the `backoff_factor`)

2. If not: weight will update normally. Scale factor will be increased (multiplied by `growth_factor`) when no inf or NaN occurs in successive iterations (specified by `growth_interval`)
