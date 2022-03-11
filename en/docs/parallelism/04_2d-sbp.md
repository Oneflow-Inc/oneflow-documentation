# 2D SBP

After reading the [Global View](./02_sbp.md) and [Global Tensor](./03_consistent_tensor.md), you may have learned about the basic concepts of SBP and SBP Signature, and can get started with related tasks. In fact, both these two documents refers to **1D SBP**.

Since you have known about 1D SBP, this document introduces 2D SBP, which can more flexibly deal with more complex distributed training scenarios.

## 2D Devices Array

We are already familiar with the placement configuration of 1D SBP. In the scenario of 1D SBP, configure the cluster through the [oneflow.placement](https://oneflow.readthedocs.io/en/master/placement.html#oneflow.placement) interface. For example, use the 0~3 GPU graphics in the cluster:

```python
>>> placement1 = flow.placement("cuda", ranks=[0, 1, 2, 3])
```

The above `"cuda"` specifies the device type, and `ranks=[0, 1, 2, 3]` specifies the computing devices in the cluster. In fact, `ranks` can be not only a one-dimensional int list, but also a multi-dimensional int array:

```python
>>> placement2 = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
```

When `ranks` is in the form of a one-dimensional list like `ranks=[0, 1, 2, 3]`, all devices in the cluster form a 1D device vector, which is where the 1D SBP name comes from.

When `ranks` is in the form of a multi-dimensional array, the devices in the cluster are grouped into a multi-dimensional array of devices. `ranks=[[0, 1], [2, 3]]` means that the four computing devices in the cluster are divided into a $2 \times 2$ device array.

## 2D SBP

When constructing a Global Tensor, we need to specify both `placement` and `SBP`. When the cluster in `placement` is a 2-dimensional device array, SBP must also correspond to it, being a `tuple` with a length of 2. The 0th and 1st elements in this `tuple` respectively describes the distribution of Global Tensor in the 0th and 1st dimensions of the device array.

For example, The following code configures a $2 \times 2$ device array, and sets the 2D SBP to `(broadcast, split(0))`.

```python
>>> a = flow.Tensor([[1,2],[3,4]])
>>> placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
>>> sbp = (flow.sbp.broadcast, flow.sbp.split(0))
>>> a_to_global = a.to_global(placement=placement, sbp=sbp)
```

It means that logically the data, over the entire device array, is `broadcast` in the 0th dimension ("viewed vertically"); `split(0)` in the 1st dimension ("viewed across").

See the following figure:

![](./imgs/2d-sbp.png)

In the above figure, the left side is the global data, and the right side is the data of each device on the device array. As you can see, from the perspective of the 0th dimension, they are all in `broadcast` relations:

- The data in (group0, device0) and (group1, device0) are consistent, and they are in `broadcast` relations to each other
- The data in (group0, device1) and (group1, device1) are consistent, and they are in `broadcast` relations to each other

From the perspective of the 1st dimension, they are all in `split(0)` relations:

- (group0, device0) and (group0, device1) are in `split(0)` relations to each other
- (group1, device0) and (group1, device1) are in `split(0)` relations to each other

It may be difficult to directly understand the correspondence between logical data and physical data in the final device array. When thinking about 2D SBP, you can imagine an intermediate state (gray part in the above figure) there. Take `(broadcast, split(0))` as an example:

- First, the original logical tensor is broadcast to 2 groups through `broadcast`, and the intermediate state is obtained
- On the basis of the intermediate state, `split(0)` is continued to be done on the groups to get the status of each physical tensor in the final device array

## 2D SBP Signature

1D SBP has the concept of SBP signature, similarly, the operator also has 2D SBP signature. Based on mastering the concept of 1D SBP and its signature concept, 2D SBP signature is very simple and you only need to follow one principle:

- Independently derive in the respective dimensions

Let's take matrix multiplication as an example. First, let's review the case of 1D SBP. Suppose that $x \times w = y$ can have the following SBP Signature:

$$ broadcast \times split(1) = split(1) $$

and

$$ split(0) \times broadcast = split(0) $$

Now, suppose we set the 2D SBP for $x$ to $(broadcast, split(0))$ and set the 2D SBP for $w$ to $(split(1), broadcast)$, then in the context of the 2D SBP, operate $x \times w = y$ to obtain the SBP attribute for $y$ is $(split(1), split(0))$.

That is to say, the following 2D SBPs constitute the 2D SBP Signature of matrix multiplication:

$$ (broadcast, split(0)) \times (split(1), broadcast) =  (split(1), split(0)) $$


## An Example of Using 2D SBP

In this section, we are going to train MobileNetv2 model on CIFAR10 dataset, in order to demonstrate how to conduct distributed training using 2D SBP. Same as the example above, assume that there is a $2 \times 2$ device array. Given that readers may not have multiple GPU devices at present, we will use **CPU** to simulate the case of $2 \times 2$ device array. And adopt the "Data Parallelism" strategy introduced in [COMMON DISTRIBUTED PARALLEL STRATEGY](./01_introduction.md).

??? code
    ```python
    import oneflow as flow
    import oneflow.nn as nn
    import flowvision
    import flowvision.transforms as transforms

    BATCH_SIZE=64
    EPOCH_NUM = 1
    DEVICE = "cpu"
    print("Using {} device".format(DEVICE))

    PLACEMENT = flow.placement(DEVICE, ranks=[[0, 1], [2, 3]])
    BROADCAST = (flow.sbp.broadcast, flow.sbp.broadcast)
    BS0 = (flow.sbp.broadcast, flow.sbp.split(0))

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
    model = model.to_global(placement=PLACEMENT, sbp=BROADCAST)

    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCH_NUM):
        print(f"Epoch {t+1}\n-------------------------------")
        size = len(train_dataloader.dataset)
        for batch, (x, y) in enumerate(train_dataloader):
            x = x.to_global(placement=PLACEMENT, sbp=BS0)
            y = y.to_global(placement=PLACEMENT, sbp=BS0)

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

It can be found that the above process is almost the same as the ordinary single machine with single GPU training process. The main differences are as follows:

- Define the placement and SBP that will be used:

```python
    PLACEMENT = flow.placement(DEVICE, ranks=[[0, 1], [2, 3]])
    BROADCAST = (flow.sbp.broadcast, flow.sbp.broadcast)
    BS0 = (flow.sbp.broadcast, flow.sbp.split(0))
```
The parameter `ranks` of `PLACEMENT` is a two-dimensional list, which represents that the devices in the cluster are divided into a device arrays of $2 \times 2$.  As mentioned earlier, the SBP needs to correspond to it and be specified as a tuple with a length of 2.

- Broadcast the model on the cluster:

```python
    model = model.to_global(placement=PLACEMENT, sbp=BROADCAST)
```

- Broadcast and split data on the cluster:

```python
    x = x.to_global(placement=PLACEMENT, sbp=BS0)
    y = y.to_global(placement=PLACEMENT, sbp=BS0)
```
The operation on the data here is the same as the example described above, except that the operation is not a $2 \times 2$ tensor, but the training data obtained from the DataLoader.

We can easily start distributed training via `oneflow.distributed.launch` module. Execute the following command in the terminal (It is assumed that the above code has been saved to a file named "2d_sbp.py" in the current directory)

```bash
python3 -m oneflow.distributed.launch --nproc_per_node=4 2d_sbp.py
```
Here, the parameter `nproc_ per_ node` is assigned as 4 to create four processes, simulating a total of 4 GPUs. For detailed usage of this module, please read: [DISTRIBUTED TRAINING LAUNCHER](./04_launch.md).
