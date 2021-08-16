# 数据并行训练

在 [常见的分布式并行策略](./01_introduction.md) 一文中介绍了数据并行的特点。

OneFlow 提供了 [oneflow.nn.parallel.DistributedDataParallel](ddp_rst_todo.md) 模块及 [launcher](launcher_rst_todo.md)，可以让用户几乎不用对单机单卡脚本做修改，就能地进行数据并行训练。

可以用快速体验 OneFlow 的数据并行：

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

## 代码

以上脚本的代码如下。


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

可以发现，编写数据并行的训练代码，要点只有2个：

- 使用 DistributedDataParallel 处理一下 module 对象（`m = ddp(m)`)
- 使用 [get_rank](todo_rst_getrank.md) 获取当前设备编号，并针对设备分发数据

然后使用 `launcher` 启动脚本，把剩下的一切都交给 OneFlow，让分布式训练，像单机单卡训练一样简单：

```pytohn
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./ddp_train.py
```


## launcher 命令行选项

可以通过 `--help` 查看 `launcher` 的选项：

```shell
python3 -m oneflow.distributed.launch --help
```

```text
usage: launch.py [-h] [--nnodes NNODES] [--node_rank NODE_RANK]
                 [--nproc_per_node NPROC_PER_NODE] [--master_addr MASTER_ADDR]
                 [--master_port MASTER_PORT] [-m] [--no_python]
                 [--redirect_stdout_and_stderr] [--logdir LOGDIR]
                 training_script ...
optional arguments:
  -h, --help            show this help message and exit
  --nnodes NNODES       The number of nodes to use for distributed training
  --node_rank NODE_RANK
                        The rank of the node for multi-node distributed training
  --nproc_per_node NPROC_PER_NODE
                        The number of processes to launch on each node, for GPU training,
                        this is recommended to be set to the number of GPUs in your system
                        so that each process can be bound to a single GPU.
  --master_addr MASTER_ADDR
                        Master node (rank 0)'s address, should be either the IP address or
                        the hostname of node 0, for single node multi-proc training, the
                        --master_addr can simply be 127.0.0.1
  --master_port MASTER_PORT
                        Master node (rank 0)'s free port that needs to be used for
                        communication during distributed training
  -m, --module          Changes each process to interpret the launch script as a python
                        module, executing with the same behavior as'python -m'.
  --no_python           Do not prepend the training script with "python" - just exec it
                        directly. Useful when the script is not a Python script.
  --redirect_stdout_and_stderr
                        write the stdout and stderr to files 'stdout' and 'stderr'. Only
                        available when logdir is set
  --logdir LOGDIR       Relative path to write subprocess logs to. Passing in a relative
                        path will create a directory if needed. Note that successive runs
                        with the same path to write logs to will overwrite existing logs,
                        so be sure to save logs as needed.
```