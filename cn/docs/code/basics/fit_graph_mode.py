import math
import numpy as np
import oneflow as flow

device = flow.device("cuda")
dtype = flow.float32

# Create Tensors to hold input and outputs.
x = flow.tensor(np.linspace(-math.pi, math.pi, 2000), device=device, dtype=dtype)
y = flow.tensor(np.sin(x), device=device, dtype=dtype)

# For this example, the output y is a linear function of (x, x^2, x^3), so
# we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
xx = flow.cat(
    [x.unsqueeze(-1).pow(1), x.unsqueeze(-1).pow(2), x.unsqueeze(-1).pow(3)], dim=1
)

# The Linear Module
model = flow.nn.Sequential(flow.nn.Linear(3, 1), flow.nn.Flatten(0, 1))
model.to(device)

# Loss Function
loss_fn = flow.nn.MSELoss(reduction="sum")
loss_fn.to(device)

# Optimizer
optimizer = flow.optim.SGD(model.parameters(), lr=1e-6)


# The Linear Train Graph
class LinearTrainGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)

    def build(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        return loss


linear_graph = LinearTrainGraph()

for t in range(2000):
    # Print loss.
    loss = linear_graph(xx, y)
    if t % 100 == 99:
        print(t, loss.numpy())


linear_layer = model[0]
print(
    f"Result: y = {linear_layer.bias.numpy()} + {linear_layer.weight[:, 0].numpy()} x + {linear_layer.weight[:, 1].numpy()} x^2 + {linear_layer.weight[:, 2].numpy()} x^3"
)
