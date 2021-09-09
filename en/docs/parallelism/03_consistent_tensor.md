# Consistent Tensor

## The Mapping Between Consistent View And Physical View

## Create Consistent Tensor

To interactively experience consistent tensor on a two-GPU machine, you may launch python separately in two consoles in the following way.

!!! Note
    **Click** the Terminal 0 or Terminal 1 label to check the commands/code

=== "Terminal 0"
    ```shell
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0
    python3
    ```

=== "Terminal 1"
    ```shell
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=2 RANK=1 LOCAL_RANK=1
    python3
    ```

The above setting of environment variables is configuration for distributed computing. Please refer to the [Extended Reading](#extented-reading) section at the end of this article for detailed explanation and launching distributed computing using some tools.

### Create Consistent Tensor Directly

In the two consoles, separately import `oneflow` and create `x`.

`flow.palcement("cuda", {0:[0,1]})` appoints the range of consistent tensor in the cluster.  
- `"cuda"` means "on GPU".  The second parameter of 
- `placement` is a dictionary. Its `key` is the index of machine, `value` is the index of the graphic cards. Therefore, `{0:[0,1]}` means that consistent tensor is on the 0th, 1st graphics card of the 0th machine.

=== "Terminal 0"
    ```python
    import oneflow as flow

    placement = flow.placement("cuda",{0:[0,1]})
    sbp = flow.sbp.split(0)
    x = flow.randn(4,5,placement=placement, sbp=sbp)
    x.shape
    ```

=== "Terminal 1"
    ```python
    import oneflow as flow

    placement = flow.placement("cuda",{0:[0,1]})
    sbp = flow.sbp.split(0)
    x = flow.randn(4,5,placement=placement, sbp=sbp)
    x.shape
    ```

Output:

=== "Terminal 0"
    ```text
    flow.Size([4, 5])
    ```

=== "Terminal 1"
    ```text
    flow.Size([4, 5])
    ```

### Get Local Tensor From Consistent Tensor

Call [to_local](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_local) to check the local tensor on a device

=== "Terminal 0"
    ```python
    x.to_local()
    tensor([[ 2.9186e-01, -3.9442e-01,  4.7072e-04, -3.2216e-01,  1.7788e-01],
            [-4.5284e-01,  1.2361e-01, -3.5962e-01,  2.6651e-01,  1.2951e+00]],
        device='cuda:0', dtype=oneflow.float32)
    ```

=== "Terminal 1"
    ```python
    x.to_local()
    tensor([[-0.4363,  0.9985, -2.5387,  0.3003,  0.3803],
            [ 0.0556, -0.8077,  1.1191, -2.1278,  0.1468]], device='cuda:1',
        dtype=oneflow.float32)
    ```

###　Convert Local Tensor To Consistent Tensor

User can create local tensor first, then use [Tensor.to_consistent](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_consistent) to convert local tensor to consistent tensor.

In the following example, two local tensor of `shape=(2, 5)` are created on the two machines. Note that after calling `to_consistent`, the result consistent tensor has `shape` `(4, 5)`

This is because of the chosen `sbp=flow.sbp.split(0)`. Two local tensor of shape `(2, 5)` needs to be concatenated on the 0th dimension and result in a `(4, 5)` consistent tensor.

=== "Terminal 0"
    ```python
    import oneflow as flow

    x = flow.randn(2,5)
    placement = flow.placement("cuda",{0:[0,1]})
    sbp = flow.sbp.split(0)
    x_consistent = x.to_consistent(placement=placement, sbp=sbp)
    x_consistent.shape
    ```

=== "Terminal 1"
    ```python
    import oneflow as flow

    x = flow.randn(2,5)
    placement = flow.placement("cuda",{0:[0,1]})
    sbp = flow.sbp.split(0)
    x_consistent = x.to_consistent(placement=placement, sbp=sbp)
    x_consistent.shape
    ```

## Practice with SBP Signature

### Data-parallelism

The following code is an example of data-parallelism of [common distributed parallelism strategy](./01_introduction.md#_4)

![data parallelism](./imgs/matmul_data_paralelism.png)

=== "Terminal 0"
    ```python
    import oneflow as flow

    placement = flow.placement("cuda",{0:[0,1]})
    x = flow.randn(4,5,placement=placement, sbp=flow.sbp.split(0))
    w = flow.randn(5,8,placement=placement, sbp=flow.sbp.broadcast)
    y = flow.matmul(x,w)
    y.sbp
    y.shape
    ```

=== "Terminal 1"
    ```python
    import oneflow as flow

    placement = flow.placement("cuda",{0:[0,1]})
    x = flow.randn(4,5,placement=placement, sbp=flow.sbp.split(0))
    w = flow.randn(5,8,placement=placement, sbp=flow.sbp.broadcast)
    y = flow.matmul(x,w)
    y.sbp
    y.shape
    ```

Observe that `flow.matmul` checks its input `x` and `w` whose SBP are `split(0)` and `broadcast` respectively. Oneflow then derives and ouputs the SBP of `y` which is `split(0)`. In the end, computation is done with a matrix of `shape=(4,8)`. Output:

=== "Terminal 0"
    ```text
    (oneflow.sbp.split(axis=0),)
    flow.Size([4, 8])
    ```

=== "Terminal 1"
    ```text
    (oneflow.sbp.split(axis=0),)
    flow.Size([4, 8])
    ```

### Model-parallelism

The following code is an example of model-parallelism of [common distributed parallelism strategy](./01_introduction.md#_5).

![data parallelism](./imgs/matmul_model_paralelism.png)

=== "Terminal 0"
    ```python
    import oneflow as flow

    placement = flow.placement("cuda",{0:[0,1]})
    x = flow.randn(4,5,placement=placement, sbp=flow.sbp.broadcast)
    w = flow.randn(5,8,placement=placement, sbp=flow.sbp.split(1))
    y = flow.matmul(x,w)
    y.sbp
    y.shape
    ```

=== "Terminal 1"
    ```python
    import oneflow as flow

    placement = flow.placement("cuda",{0:[0,1]})
    x = flow.randn(4,5,placement=placement, sbp=flow.sbp.broadcast)
    w = flow.randn(5,8,placement=placement, sbp=flow.sbp.split(1))
    y = flow.matmul(x,w)
    y.sbp
    y.shape
    ```

Observe that `flow.matmul` checks its input `x` and `w` whose SBP are `broadcast` and `split(0)` respectively. Oneflow then derives and ouputs the SBP of `y` which is `split(1)`. In the end, computation is done with a matrix of `shape=(4,8)`. Output:

=== "Terminal 0"
    ```text
    (oneflow.sbp.split(axis=1),)
    flow.Size([4, 8])
    ```

=== "Terminal 1"
    ```text
    (oneflow.sbp.split(axis=1),)
    flow.Size([4, 8])
    ```

## Extented Reading

### Environment Variables in Multi-machine Training

The example in this article sets environment variables to configure distributed training. Doing so allows programmers to see the effects and output in a interactive python environment. If the training is needed in production instead of in learning or experiments, one may launch distributed training using [oneflow.distributed.launch](./04_launch.md). This module automatically sets necessary environment variables based on command-line arguments.

- `MASTER_ADDR`：The IP of the 0th machine in a multi-machine case
- `MASTER_PORT`：The listening port of the 0th machine in a multi-machine case. Note that this port should not be in use
- `WORLD_SIZE`：The number of computing devices in the whole cluster. Because currently oneflow only supports having the  same number of GPUs on the machines, `WORLD_SIZE` is actually $number\:of\:machines \times number\:of\:GPUs\:on\:one\:machine$. In our example, we have one machine and two GPUs on it, so `WORLD_SIZE=2`

`RANK` and `LOCAL_RANK` are indexs for machines. The difference is that `RANK` is a "global perspective" index, while `LOCAL_RANK` is a "local perspective" index. In the case that only one machine is involved, the `RANK` and `LOCAL_RANK` are the same. In our example, there are two GPUs, indexed 0 and 1.

When there are multiple machines, the upper bound of `LOCAL_RANK` on a machine is the number of computing devices on the machine. The upper bound of `RANK` is the sum  of all computing devices on all machines. The indexing of these computing devices starts from 0.

Assume that there are two machines and there are two graphics cards on each machine. The list below illustrates the correspondence between `LOCAL_RANK` and `RANK`

|                      | RANK | LOCAL_RANK |
| -------------------- | ---- | ---------- |
| GPU #0 on Machine #0 | 0    | 0          |
| GPU #1 on Machine #0 | 1    | 1          |
| GPU #0 on Machine #1 | 2    | 0          |
| GPU #1 on Machine #1 | 3    | 1          |

### Boxing（Automatic Conversion Of SBP）

From the coding example, we learned that an operator can derive and set the SBP of the output tensor, given the SBP of the input tensor and the built-in SBP Signature of the operator.

But what if the SBP of the output tensor does not satisfy what the next-layer operater requires?

Assume that in data-parallelism, there are two layers of matrix multiplication. Both layers uses model-parallelism.

![multi-layer-matmul](./imgs/multi-matmul.png)

The SBP (`split(1)`) of the output from the first layer is not what the second layer expects (`broadcast`). In this case, OneFlow automatically inserts Boxing in between the output of the first layer and the input of the second layer. Collective communication is used to perform necessary data conversion.。

Converting `split(1)` to `broadcast` is equivalent to an `AllGather` operation, as shown in the figure below.

![s2b](./imgs/boxing_s2b.png)

Because of Boxing, the users only need to focus on the SBP setting of the critical places (like the source operator). The rest are all handled by the OneFlow framework.