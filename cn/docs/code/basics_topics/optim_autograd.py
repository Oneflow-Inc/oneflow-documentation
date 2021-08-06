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