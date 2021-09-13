# 流水并行训练

在 [常见的分布式并行策略](./01_introduction.md) 一文中介绍了流水并行的特点。

在 OneFlow 的 [一致性视角](./03_consistent_tensor.md) 下，通过简单的设置 Tensor 的 `placement` 属性，就可以实现流水并行。

以下代码是简单的示范，它将 [快速上手](../basics/01_quickstart.md) 中的网络，以流水并行的方式运行。前几层的 Module `nn.Flatten`、`nn.Linear(28*28, 512)`、`nn.ReLU()` 在 GPU0 上运行；剩余的网络部分在 GPU1 上运行。

??? code
    ```python
    import oneflow as flow

    BATCH_SIZE = 16
    DEVICE = "cuda"
    BROADCAST = [flow.sbp.broadcast]
    P0 = flow.placement("cuda", {0: [0]})
    P1 = flow.placement("cuda", {0: [1]})

    class Stage0Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = flow.nn.Flatten()
            self.linear0 = flow.nn.Linear(28*28, 512)
            self.relu0 = flow.nn.ReLU()

        def forward(self, x):
            out = self.flatten(x)
            out = self.linear0(out)
            out = self.relu0(out)
            return out

    class Stage1Module(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = flow.nn.Linear(512, 512)
            self.relu1 = flow.nn.ReLU()
            self.linear2 = flow.nn.Linear(512, 10)
            self.relu2 = flow.nn.ReLU()

        def forward(self, x):
            out = self.linear1(x)
            out = self.relu1(out)
            out = self.linear2(out)
            out = self.relu2(out)
            return out

    class PipelineModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.m_stage0 = Stage0Module()
            self.m_stage1 = Stage1Module()
            
            self.m_stage0.to_consistent(placement=P0, sbp=BROADCAST)
            self.m_stage1.to_consistent(placement=P1, sbp=BROADCAST)

        def forward(self, x):
            out_stage0 = self.m_stage0(x)
            in_stage1 = out_stage0.to_consistent(placement=P1, sbp=BROADCAST)
            out_stage1 = self.m_stage1(in_stage1)
            return out_stage1

    module_pipeline = PipelineModule()
    sgd = flow.optim.SGD(module_pipeline.parameters(), lr=0.001)

    class PipelineGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.module_pipeline = module_pipeline
            self.module_pipeline.m_stage0.config.stage_id = 0
            self.module_pipeline.m_stage1.config.stage_id = 1
            self.loss_fn = flow.nn.CrossEntropyLoss()
            self.config.set_gradient_accumulation_steps(2)
            self.add_optimizer(sgd)

        def build(self, x, y):
            out = self.module_pipeline(x)
            loss = self.loss_fn(out, y)
            loss.backward()
            return loss

    graph_pipeline = PipelineGraph()

    x = flow.randn(BATCH_SIZE, 1, 28, 28)
    x = x.to_consistent(P0, BROADCAST)
    y = flow.randint(0, 10, (BATCH_SIZE,))
    y = y.to_consistent(P1, BROADCAST)

    for i in range(20):
        loss = graph_pipeline(x, y)
        print(loss.to_local())
    ```

以上代码，保存为脚本（如 `pipeline.py`）后，使用 [launch 模块启动分布式训练](./04_launch.md)：

```shell
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./pipeline.py
```

## 代码解读

### Local Tensor 与 Consistent Tensor 的转换

以上代码使用了随机生成的数据作为输入。

```python
    x = flow.randn(BATCH_SIZE, 1, 28, 28)
    x = x.to_consistent(P0, BROADCAST)
```

当使用 `launch` 模块启动训练时，因为命令行参数为 `--nproc_per_node 2`，`launch` 会启动 2 个进程。两个进程均为执行脚本中的代码。其中 `x = flow.randn(BATCH_SIZE, 1, 28, 28)` 返回的是 Local Tensor（只在本进程中有效的本地数据），当运行 `x = x.to_consistent(P0, BROADCAST)` 时，OneFlow 会自动将所有紧张中的 Local Tensor 整合为 Consistent Tensor。

在实际的训练中，各个计算设备也可以加载属于给自的本地数据，然后通过 `to_consistent` 实现 Local Tensor 到 Consistent Tensor 的转化。

### 流水设置

通过设置 Module 的 `placement` 和 `sbp` 属性，就可以指定网络分配到哪个计算设备上运行，将一个网络拆分为多个流水阶段（stage）。

在此我们定义了一个 `PipelineModule` 专门设置各阶段的流水信息。

```python
    class PipelineModule(flow.nn.Module):
        def __init__(self):
            #...
            
            self.m_stage0.to_consistent(placement=P0, sbp=BROADCAST)
            self.m_stage1.to_consistent(placement=P1, sbp=BROADCAST)

        def forward(self, x):
            out_stage0 = self.m_stage0(x)
            in_stage1 = out_stage0.to_consistent(placement=P1, sbp=BROADCAST)
            out_stage1 = self.m_stage1(in_stage1)
            return out_stage1
```

### Stage ID 及 梯度累积设置

通过设置 Module 的 `config.stage_id` 属性，设置 Stage ID，Stage ID 从0开始编号，依次加1。
调用 `self.config.set_gradient_accumulation_steps` 方法，设置梯度累积的步长。
OneFlow 通过这两项配置，获取实现流水并行中的 micro batch 技术所需的信息。

```python
    self.module_pipeline.m_stage0.config.stage_id = 0
    self.module_pipeline.m_stage1.config.stage_id = 1
    self.config.set_gradient_accumulation_steps(2)
```
