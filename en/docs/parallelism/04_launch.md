# DISTRIBUTED TRAINING LAUNCHER

OneFlow provides the `oneflow.distributed.launch` module to help users start distributed training more conveniently.

Users can start distributed training by the following commands:

```shell
python3 -m oneflow.distributed.launch [Boot Option] training_script.py
```

For example, to start the training on single-node double-GPUs:

```shell
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./script.py
```

For another example, start two devices,  and each devices has two graphics for training.

Run on device 0:

```shell
python3 -m oneflow.distributed.launch --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.1" \
    --master_port=7788 \
    script.py
```

Run on device 1:

```shell
python3 -m oneflow.distributed.launch --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.1" \
    --master_port=7788 \
    script.py
```

## Description of Common Options

We can view the description of the options of the `launch` module after running `python3 -m oneflow.distributed.launch -h`. The following are some common options:

- `--nnodes`: number of nodes
- `--node_rank`: the serial number of the devices, starting from 0
- `--nproc_per_node`: The number of processes per node to be started on each device, which is recommended to be global with the number of GPUs
- `--logdir`: The relative storage path of the child process log

## The Relationship between Launch Module and Parallel Strategy

The main function of `oneflow.distributed.launch` is to allow users to start distributed training more conveniently after them complete the distributed program. It saves the trouble of configuring [environment variables](./03_consistent_tensor.md#_5) in the cluster.

But `oneflow.distributed.launch` **does not determine** [Parallel Strategy](./01_introduction.md). The Parallel Strategy is determined by the setup of the distribution method of data and the model, and the placement of those on the physical devices.

OneFlow provides [Global View](./02_sbp.md) and [Global Tensor](./03_consistent_tensor.md) to flexibly configure parallel strategies. And for data parallelism, OneFlow provides the [DistributedDataParallel](./05_ddp.md) module, which can change the single-node single-GPUs script to the script of data parallel with  minimal code modification.
