# DATA PARALLELISM TRAINING

In [Common Distributed Parallel Strategies](./01_introduction.md), we introduced the characteristics of data parallel.

OneFlow provides two ways to accomplish data parallel, and one of them is to use the original concept of Oneflow to run data parallel training by configurating global tensor. This is also the **recommanded way** to run data parallel training on Oneflow.

Besides, to facilitate the users who are transferring from PyTorch to OneFlow, OneFlow offers the interface consistent with PyTorch `torch.nn.parallel.DistributedDataParallel`,  [oneflow.nn.parallel.DistributedDataParallel](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.nn.parallel.DistributedDataParallel.html) so that users can also conveniently extend single machine training to data parallel training. 

## Run Data Parallel Training With SBP Configuration

The following code runs data parallel training by configurating global tensor. 

??? code
    ```python
    import oneflow as flow
    import oneflow.nn as nn
    import flowvision
    import flowvision.transforms as transforms

    BATCH_SIZE=64
    EPOCH_NUM = 1

    PLACEMENT = flow.placement("cuda", [0,1])
    S0 = flow.sbp.split(0)
    B = flow.sbp.broadcast

    DEVICE = "cuda" if flow.cuda.is_available() else "cpu"
    print("Using {} device".format(DEVICE))

    training_data = flowvision.datasets.CIFAR10(
        root="data",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    train_dataloader = flow.utils.data.DataLoader(
        training_data, BATCH_SIZE, shuffle=True
    )

    model = flowvision.models.mobilenet_v2().to(DEVICE)
    model.classifer = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 10))
    model = model.to_global(placement=PLACEMENT, sbp=B)

    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCH_NUM):
        print(f"Epoch {t+1}\n-------------------------------")
        size = len(train_dataloader.dataset)
        for batch, (x, y) in enumerate(train_dataloader):
            x = x.to_global(placement=PLACEMENT, sbp=S0)
            y = y.to_global(placement=PLACEMENT, sbp=S0)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current = batch * BATCH_SIZE
            if batch % 5 == 0:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    ```

We can see that this script is almost identical to the training script for the single-machine and single-device configuration, with a few exceptions on some lines related to consitent tensor set-ups. These codes are:

- Set placement to place the training one GPU 0 and 1: 

```python
    PLACEMENT = flow.placement("cuda", [0,1])
```

- Broadcast the model on clusters:

```python
    model = model.to_global(placement=PLACEMENT, sbp=B)
```

- Split the data on cluster with `split(0)`: 

```python
    x = x.to_global(placement=PLACEMENT, sbp=S0)
    y = y.to_global(placement=PLACEMENT, sbp=S0)
```

This allows us to follow the instructions in [Common Distributed Parallel Strategies](./01_introduction.md)

## Run Data Parallel Training With DistributedDataParallel

The following codes provides a quick start for training data parallel with `oneflow.nn.parallel.DistributedDataParallel` :

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

There are only two differences between the data parallelism training code and the stand-alone single-card script:

- Use `DistributedDataParallel` to wrap the module object (`m = ddp(m)`)
- Use [get_rank](https://oneflow.readthedocs.io/en/v0.8.1/generated/oneflow.env.get_rank.html) to get the current device number and distribute the data to the device.

Then use `launcher` to run the script, leave everything else to OneFlow, which makes distributed training as simple as stand-alone single-card training:

```python
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./ddp_train.py
```

### DistributedSampler

The data used is manually distributed in this context to highlight `DistributedDataParallel`. However, in practical applications, you can directly use [DistributedSampler](https://oneflow.readthedocs.io/en/v0.8.1/utils.data.html?highlight=DistributedSampler#oneflow.utils.data.distributed.DistributedSampler) with data parallel.

`DistributedSampler` will instantiate the Dataloader in each process, and each Dataloader instance will load a part of the complete data to automatically complete the data distribution.
