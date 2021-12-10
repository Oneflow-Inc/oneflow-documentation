# 数据并行训练

在 [常见的分布式并行策略](./01_introduction.md) 一文中介绍了数据并行的特点。
在 OneFlow 中，提供了两种做数据并行的方式。

一种是使用 OneFlow 的原生的 SBP 概念，通过设置张量的 SBP 属性，进行数据并行训练，这也是用 OneFlow 做数据并行训练的 **推荐方式** 。

此外，为了方便从 PyTorch 迁移到 OneFlow 的用户，OneFlow 提供了与 `torch.nn.parallel.DistributedDataParallel` 对齐一致的接口 [oneflow.nn.parallel.DistributedDataParallel](https://oneflow.readthedocs.io/en/master/nn.html#oneflow.nn.parallel.DistributedDataParallel)，它也能让用户方便地从单机训练脚本，扩展为数据并行训练。

## 通过设置 SBP 做数据并行训练



## 使用 DistributedDataParallel 做数据并行训练

可以用以下命令快速体验 `oneflow.nn.parallel.DistributedDataParallel` 做数据并行：

```shell
wget https://docs.oneflow.org/master/code/parallelism/ddp_train.py #下载脚本
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./ddp_train.py #数据并行训练
```

输出：

```text
50/500 loss:0.004111831542104483
50/500 loss:0.00025336415274068713
...
500/500 loss:6.184563972055912e-11
500/500 loss:4.547473508864641e-12

w:tensor([[2.0000],
        [3.0000]], device='cuda:1', dtype=oneflow.float32,
       grad_fn=<accumulate_grad>)

w:tensor([[2.0000],
        [3.0000]], device='cuda:0', dtype=oneflow.float32,
       grad_fn=<accumulate_grad>)
```

### 代码

点击以下 “Code” 可以展开以上运行脚本的代码。

??? code
    ```python
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
            self.w = flow.nn.Parameter(flow.tensor([[0], [0]], dtype=flow.float32))

        def forward(self, x):
            x = flow.matmul(x, self.w)
            return x


    m = Model().to("cuda")
    m = ddp(m)
    loss = flow.nn.MSELoss(reduction="sum")
    optimizer = flow.optim.SGD(m.parameters(), m.lr)

    for i in range(0, m.iter_count):
        rank = flow.env.get_rank()
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
    ```

可以发现，它与单机单卡脚本的不同只有2个：

- 使用 `DistributedDataParallel` 处理一下 module 对象（`m = ddp(m)`)
- 使用 [get_rank](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.env.get_rank) 获取当前设备编号，并针对设备分发数据

然后使用 `launcher` 启动脚本，把剩下的一切都交给 OneFlow，让分布式训练，像单机单卡训练一样简单：

```pytohn
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./ddp_train.py
```

### DistributedSampler

本文为了简化问题，突出 `DistributedDataParallel`，因此使用的数据是手工分发的。在实际应用中，可以直接使用 [DistributedSampler](https://oneflow.readthedocs.io/en/master/utils.html#oneflow.utils.data.distributed.DistributedSampler) 配合数据并行使用。

`DistributedSampler` 会在每个进程中实例化 Dataloader，每个 Dataloader 实例会加载完整数据的一部分，自动完成数据的分发。
