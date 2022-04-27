import oneflow as flow
from oneflow import nn

print("定义之后要使用到的 placement、SBP 等")
P = flow.placement("cuda", ranks=[0, 1])
B = flow.sbp.broadcast
S0 = flow.sbp.split(0)
DEVICE = "cuda"

print("定义一个简单的模型，然后广播到集群上")
model = nn.Sequential(nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 10))
model = model.to(DEVICE)
model.train()
model = model.to_global(placement=P, sbp=B)

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)

print("ZeRO 是在 nn.Graph 中设置的，因此需要将动态图模型转换为 nn.Graph")


class CustomGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.add_optimizer(optimizer)

        print("在 nn.Graph 中开启 ZeRO")
        print("阶段 1：通过 config.set_zero_redundancy_optimizer_mode 接口开启")
        self.config.set_zero_redundancy_optimizer_mode("distributed_split")
        print("阶段 2：在阶段 1 的基础上，增加 flow.boxing.nccl.enable_use_compute_stream(True)")
        self.config.set_zero_redundancy_optimizer_mode("distributed_split")
        flow.boxing.nccl.enable_use_compute_stream(True)
        print(
            "阶段 3：在阶段 2 的基础上，增加 flow.boxing.nccl.disable_group_boxing_by_dst_parallel(True)")
        self.config.set_zero_redundancy_optimizer_mode("distributed_split")
        flow.boxing.nccl.enable_use_compute_stream(True)
        flow.boxing.nccl.disable_group_boxing_by_dst_parallel(True)

    def build(self, x, y):
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        loss.backward()
        return preds


print("定义训练流程")
graph_model = CustomGraph()

for _ in range(100):
    x = flow.randn(128, 256).to(DEVICE)
    y = flow.ones(128, 1, dtype=flow.int64).to(DEVICE)
    global_x = x.to_global(placement=P, sbp=S0)
    global_y = y.to_global(placement=P, sbp=S0)

    graph_model(global_x, global_y)

print("测试结束")
