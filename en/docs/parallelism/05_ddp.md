# DATA PARALLEL TRAINING

In [Common Distributed Parallel Strategies](./01_introduction.md), we introduced the characteristics of data parallel.

OneFlow provides [oneflow.nn.parallel.DistributedDataParallel](https://oneflow.readthedocs.io/en/master/nn.html#oneflow.nn.parallel.DistributedDataParallel) module and [launcher](https://oneflow.readthedocs.io/en/master/distributed.html#oneflow-distributed), which allows users to run data parallel training almost without modifying the script of a single node.
A quick start of OneFlow's data parallel:

```shell
wget https://docs.oneflow.org/master/code/parallelism/ddp_train.py #Download
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./ddp_train.py #data parallel training
```

Out：

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

## Codes

Click "Code" below to expand the code of the above running script.

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
    ```

There are only two differences between the data parallel training code and the stand-alone single-card script:

- Use DistributedDataParallel to wrap the module object (`m = ddp(m)`)
- Use [get_rank](https://oneflow.readthedocs.io/en/master/oneflow.html#oneflow.env.get_rank) to get the current device number and distribute the data to the device.

Then use `launcher` to run the script, leave everything else to OneFlow, which makes distributed training as simple as stand-alone single-card training:

```pytohn
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./ddp_train.py
```

## DistributedSampler

The data used is manually distributed in this context to highlight `DistributedDataParallel`. However, in practical applications, you can directly use [DistributedSampler](https://oneflow.readthedocs.io/en/master/utils.html#oneflow.utils.data.distributed.DistributedSampler) with data parallel.

`DistributedSampler` will instantiate the Dataloader in each process, and each Dataloader instance will load a part of the complete data to automatically complete the data distribution.