import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp

train_x = [
    flow.tensor([[1, 2], [2, 3]], dtype=flow.float32),
    flow.tensor([[4, 6], [3, 1]], dtype=flow.float32),
]
train_y = [
    flow.tensor([[8], [13]], dtype=flow.float32),
    flow.tensor([[26], [9]], dtype=flow.float32),
]


class Model(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = 0.01
        self.iter_count = 500
        self.w = flow.nn.Parameter(flow.randn(2, 1, dtype=flow.float32))
        self.used_only_in_rank0 = flow.nn.Parameter(
            flow.randn(2, 1, dtype=flow.float32)
        )

    def forward(self, x):
        x = flow.matmul(x, self.w)
        if flow.framework.distribute.get_rank() == 0:
            x = x * self.used_only_in_rank0
        return x


m = Model().to("cuda")
m = ddp(m)
loss = flow.nn.MSELoss(reduction="sum")
optimizer = flow.optim.SGD(m.parameters(), m.lr)

for i in range(0, m.iter_count):
    rank = flow.framework.distribute.get_rank()
    x = train_x[rank].to("cuda")
    y = train_y[rank].to("cuda")

    y_pred = m(x)
    l = loss(y_pred, y)
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{m.iter_count} loss:{l}")

    optimizer.zero_grad()
    l.backward()
    optimizer.step()

print(f"\nw:{m.w}")
