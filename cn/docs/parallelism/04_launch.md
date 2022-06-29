# 用 launch 模块启动分布式训练

OneFlow 提供了 `oneflow.distributed.launch` 模块帮助用户更方便地启动分布式训练。

用户可以借助以下的形式，启动分布式训练：

```shell
python3 -m oneflow.distributed.launch [启动选项] 训练脚本.py
```

比如，启动单机两卡的训练：

```shell
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./script.py
```

再比如，启动两台机器，每台机器有两张显卡的训练。

在0号机器上运行：

```shell
python3 -m oneflow.distributed.launch \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.1" \
    --master_port=7788 \
    script.py
```

在1号机器上运行：

```shell
python3 -m oneflow.distributed.launch \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.1" \
    --master_port=7788 \
    script.py
```

## 常见选项说明

通过 `python3 -m oneflow.distributed.launch -h` 可以查看 `launch` 模块的选项说明，以下是部分常见选项。

- `--nnodes`：机器的数目(number of nodes)
- `--node_rank`： 机器的编号，从0开始
- `--nproc_per_node`：每台机器上要启动的进程数目(number of processes per node)，推荐与 GPU 数目一致
- `--logdir`：子进程日志的相对存储路径

## launch 模块与并行策略的关系

注意 `oneflow.distributed.launch` 的主要作用，是待用户完成分布式程序后，让用户可以更方便地启动分布式训练。它省去了配置集群中[环境变量](./03_global_tensor.md#_5) 的繁琐。

但是 `oneflow.distributed.launch` **并不决定** [并行策略](./01_introduction.md)，并行策略是由设置数据、模型的分发方式、在物理设备上的放置位置决定的。

OneFlow 提供的 [全局视角](./02_sbp.md) 和 [Global Tensor](./03_global_tensor.md) 可以灵活地配置并行策略。并且针对数据并行，OneFlow 提供了 [DistributedDataParallel](./05_ddp.md) 模块，可以在极少修改代码的前提下，将单机单卡的脚本改为数据并行的脚本。
