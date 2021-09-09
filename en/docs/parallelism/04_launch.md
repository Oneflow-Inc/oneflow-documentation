#  START DISTRIBUTED TRAINING with the launch MODULE

OneFlow provides the `oneflow.distributed.launch` module to help users start distributed training more conveniently.

Users can start distributed training by the following forms:

```shell
python3 -m oneflow.distributed.launch [Boot Option] training_script.py
```

For example, to start the training of a single machine with two graphics cards:

```shell
python3 -m oneflow.distributed.launch --nproc_per_node 2 ./script.py
```

For another example, start two machines, and each machine has two graphics cards for training.

Run on machine 0:

```shell
python3 -m oneflow.distributed.launch --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.1" \
    --master_port=7788 \
    script.py
```

Run on machine 1:

```shell
python3 -m oneflow.distributed.launch --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=2 \
    --master_addr="192.168.1.1" \
    --master_port=7788 \
    script.py
```
### Description of Common Options

We can view the description of the options of the `launch` module after running `python3 -m oneflow.distributed.launch -h`. The following are some common options:

-`--nnodes`: number of nodes
-`--node_rank`: the serial number of the machines, starting from 0
-`--nproc_per_node`: The number of processes per node to be started on each machine, which is recommended to be consistent with the number of GPUs
-`--logdir`: The relative storage path of the child process log

### The Relationship between Launch Module and Parallel Strategy

The main function of `oneflow.distributed.launch` is to allow users to start distributed training more conveniently after the user completes the distributed program. It saves the trouble of configuring [environment variables](./03_consistent_tensor.md#_5) in the configure cluster.

But `oneflow.distributed.launch` **does not determine** [Parallel Strategy](./01_introduction.md). the Parallel Strategy is determined by the setting data, the distribution method of the model, and the path on the physical device.

OneFlow provides [Consistent Perspective](./02_sbp.md) and [Consistent Tensor](./03_consistent_tensor.md) to flexibly configure parallel strategies. And for data parallelism, OneFlow provides the [DistributedDataParallel](./05_ddp.md) module, which can change the stand-alone single-card script to the script of data parallel under the premise of minimal code modification.
