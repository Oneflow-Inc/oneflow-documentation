# Using Global Tensor to Program on Multi-Device Multi-GPU: Basic Operations

By [YaoChi](https://github.com/doombeaker), [Xu Xiaoyu](https://github.com/strint), [Zuo Yihao](https://github.com/Alive1024), [Guoliang Cheng](https://github.com/lmyybh), [Shen Jiali](https://github.com/Carly-Shen)


Global tensor can be executed on multi-device multi-GPU, and it’s an interface to implement the Global View programming.

Today, most parallel programs adopt the SPMD (Single program, multiple data) programming method, which means the devices will execute the same program but process different parts of the data to realize data parallelism. Take PyTorch’s DDP (Distributed Data Parallel) for example, each process executes the same neural network computing logic, but the difference is that they load different slices of one dataset.

But, the defect of SPMD programming is that multiple data makes communications more complicated. In a deep learning scenario, SPMD programming needs to insert communication operations into original computing codes, such as AllReduce for data parallelism and AllGather/ReduceScatter for model parallelism. If the parallel mode is much more complicated or a new mode needs to be experimented with, it will be troublesome to develop and maintain after inserting the communication operations.

Global View programming permits users to program from the SPSD view. Different from SPMD programming, SPSD programming is a method that data is also single from the programming interface layer.

When we extend a single-process program to a parallelly executed one, the single-process data will also be extended to the multi-process data, so it's natural that the data on different processes corresponds to the same logic data on the originally single-process program. And the logic data is called Global Tensor in OneFlow.

Global Tensor supports users to utilize the SPSD interface to program, which means users can program on a single device and OneFlow framework will automatically convert to physical SPMD/MPMD mode and execute the program in a parallel/distributed way.

With Global Tensor, a more naturally Global View programming method is available, and users can regard the multi-devices as a single device to implement SPSD programming.


## Global Tensor

In programming languages, "Global" usually refers to in-process global visibility, such as [Global Variable](https://en.wikipedia.org/wiki/Global_variable).

Instead, the "Global" of the "Global Tensor" means inter-process global visibility. So, it’s more accurate to regard the Global Tensor as a tensor that can be seen on all processes.

Global Tensor exists on all processes. When the tensor is executed by an operator on all processes, it will be automatically executed on multi-device multi-GPU.

At present, the commonly-used tensor is only visible on one process and also exists on a single device. OneFlow calls it the Local Tensor, which means it’s a tensor that can be seen on only one process. Local is relative to Global, so Local Tensor can be considered as Local (on one process) Tensor.

Most of OneFlow’s operators are compatible with the execution of Local Tensors and Global Tensors. It’s convenient to convert the Local Tensor to the Global Tensor, so the code originally executed on single-device single-GPU can be smoothly converted to ones that can be executed on multi-device multi-GPU.

Global Tensor allows users to easily develop models on multi-device multi-GPU. Compared to utilizing the original communication operators, the efficiency of developing parallelly executed models will be doubled.

## Creating Global Tensor

Let’s try to create a Global Tensor on a machine with two GPUs. Take `randn` operator for example, a Python file named `test_randn_global.py` needs to be created and add the following content to it:

```python
import oneflow as flow
# Place a global tensor on cuda device of rank(process) 0 and 1
placement = flow.placement(type="cuda", ranks=[0, 1])
# Each rank's local data is a part data as a result of spliting global data on dim 0
sbp = flow.sbp.split(dim=0)
# Create a global tensor by randn
x = flow.randn(4, 5, placement=placement, sbp=sbp)
# Print local data
print("Local data of global tensor:\n ", x.to_local().numpy())
# Print global data
print("Global data of global tensor:\n ", x.numpy())
```

Here are some explanations for some new concepts in the code above:

- `placement` refers to the physical device where the Global Tensor locates. The parameter `type` specifies the type of the physical device, and here we use ` "cuda"` to represent the GPU device. The parameter `ranks` specifies the device ID. For readers who don’t have 2 GPUs, the parameter `type` can be specified as `"cpu"` to use the CPU to simulate multiple devices, and the following code still works.
- `sbp` refers to the distributed way of the Global Tensor. Here, `sbp = flow.sbp.split(dim=0)` means that the Global Tensor is evenly split along dimension 0.
- The `to_local()` method is to acquire the Local Tensor in the present rank from the Global Tensor because the Global Tensor has one Local Tensor in each rank as its practically existing local component.


Next, configure the environment variables required by multi-process launching. Here, the machine owns 2 GPUs, which correspond to 2 process launchings. So, we should turn on 2 terminals and respectively configure the following environment variables:

!!! Note

    **Clicking** the label "Terminal 0" or "Terminal 1" separately to check its corresponding console’s command/code.
    

=== "Terminal 0"

    ```shell
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0
    ```

=== "Terminal 1"

    ```shell
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=2 RANK=1 LOCAL_RANK=1
    ```

More about detailed explanation of the environment variables above and how to conduct a distributed launching with the help of tools, please refer to [Further reading](#_2).

Finally, launch `test_randn_global.py` in two terminals respectively and observe the results of creating the Global Tensor:

```
python3 test_randn_global.py
```

In Terminal 0 (rank 0), we can see:

```
Local data of global tensor:
  [[-0.07157125 -0.92717147  1.5102768   1.4611115   1.014263  ]
 [-0.1511031   1.570759    0.9416077   0.6184639   2.4420679 ]]
Global data of global tensor:
  [[-0.07157125 -0.92717147  1.5102768   1.4611115   1.014263  ]
 [-0.1511031   1.570759    0.9416077   0.6184639   2.4420679 ]
 [-0.38203463  0.453836    0.9136015   2.35773    -0.3279942 ]
 [-0.8570119  -0.91476554 -0.06646168  0.50022084 -0.4387695 ]]
```

In Terminal 1 (rank 1), we can see:

```
Local data of global tensor:
  [[-0.38203463  0.453836    0.9136015   2.35773    -0.3279942 ]
 [-0.8570119  -0.91476554 -0.06646168  0.50022084 -0.4387695 ]]
Global data of global tensor:
  [[-0.07157125 -0.92717147  1.5102768   1.4611115   1.014263  ]
 [-0.1511031   1.570759    0.9416077   0.6184639   2.4420679 ]
 [-0.38203463  0.453836    0.9136015   2.35773    -0.3279942 ]
 [-0.8570119  -0.91476554 -0.06646168  0.50022084 -0.4387695 ]]
```
It’s clear that if we concatenate the Local Tensors in rank 1 and rank 2 on dimension 0, we can get the complete value of the Global Tensor.



## Converting Local Tensor to Global Tensor

We can firstly create a Local Tensor and then utilize the [Tensor.to_global](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_global) method to convert the Local Tensor to a Global Tensor.


Create the following program and launch it in the similar way mentioned above:

```python
import oneflow as flow
x = flow.randn(2, 5).cuda()
print(x.is_local) # True
print(x.is_global) # False
placement = flow.placement(type="cuda", ranks=[0, 1])
sbp = flow.sbp.split(0)
x_global = x.to_global(placement=placement, sbp=sbp)
print(x_global.shape) # (4, 5)
print(x.is_local) # True
print(x_global.is_global) # True
```


This program separately creates a Local Tensor with the shape of (2,5) on 2 GPUs, and the newly-created tensors are called x.

Then, we specify cuda devices in rank 0 and rank 1 as the placement and `split(dim=0)` as its SBP. After the `to_global` method, the original Local Tensor is converted to the Global Tensor named `x_global`.

We can see that the shape of `x_global` has been changed into `(4, 5)`, which is the same as the (global) shape of the Global Tensor.

The relationship between the Global Tensor and the Local Tensor is the total and the component, and the Local Tensor is the component of the total in a certain rank. The specific relationship between the Global Tensor and the Local Tensor is decided by the placement and SBP. For example, in the above case, the relationship is between tensors on GPU 0 and GPU 1, and we split `x_global` along dimension 0 to get `x`.

Based on the above relationship, the `to_global` method can infer `x_global.shape` according to `x.shape`: it concatenates the Local Tensor `x` on 2 GPUs along dimension 0 to obtain `x_global`.

Except for shape, the Global Tensor also contains some data. The Global Tensor has a Local Tensor in each rank to symbolize its local component, which is its physical data in every rank. By the way, each rank only stores different parts of the data.



## Converting Global Tensor to Local Tensor



You can utilize the [to_local](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_local) method to obtain the local component of the Global Tensor, just like the following:

```python
import oneflow as flow
placement = flow.placement(type="cuda", ranks=[0, 1])
sbp = flow.sbp.split(0)
x = flow.randn(4, 5, placement=placement, sbp=sbp)
print(x.to_local())
```



When the `x.to_local()` method is executed, two different ranks will separately obtain a Local Tensor with the shape of `(2, 5)`.



In Terminal 0 (rank 0), we can see:

```
tensor([[-0.2730,  1.8042,  0.0721, -0.5024, -1.2583],
    	[-0.3379,  0.9371,  0.7981, -0.5447, -0.5629]],
   	   dtype=oneflow.float32)
```



In Terminal 1 (rank 1), we can see:

```
tensor([[ 0.6829,  0.4849,  2.1611,  1.4059,  0.0934], 
        [-0.0301, -0.6942, -0.8094, -1.3050, -0.1778]], 
       dtype=oneflow.float32)
```


The `to_local()` has no parameters, because the Global Tensor has already confirmed its local component according to the placement and SBP, and it’s fine to directly acquire the Local Tensor that its local component corresponds to.


## Converting One Global Tensor to Another Global Tensor 


Usually, distributed computing requires inserting communication operations into normal computational logic, but OneFlow only needs users to convert the data distribution type of the Global Tensor.

In terms of type, the biggest difference between the Global Tensor and the general Local Tensor is that the Global Tensor has global data distribution type, which specifies how the Global Tensor is distributed in each rank, including its placement and SBP.


The function of placement in global data distribution type is to specify the device group where data is distributed: 

- The parameter `type` specifies the physical device type. `cuda represents the GPU device memory, and `cpu` refers to the CPU device memory.
- The parameter `ranks` specifies the process ID set. Because each rank corresponds to one physical device, `ranks` can also be seen as the device ID set. Actually, `ranks` is an nd-array composed of rank ID, which supports high-dimensional device arrangement.

For more details, please refer to [oneflow.placement](https://oneflow.readthedocs.io/en/master/tensor_attributes.html?highlight=placement#oneflow.placement).



The function of SBP in the global data distribution type is to specify the relationship between global data and local data:

- S, i.e., split(dim), notes that the relationship between global data and local data is split, indicating the global data is evenly split according to the dimension dim and distributed in each rank.

- B, i.e., broadcast, notes that the relationship between global data and local data is broadcast, indicating the global data is replicated in each rank.

- P, i.e., partial_sum, notes that the relationship between global data and local data is partial, indicating the value of the global data is the element-wise sum of the local data distributed in each rank.

For more details, please refer to [oneflow.sbp.sbp](https://oneflow.readthedocs.io/en/master/tensor_attributes.html?highlight=placement#oneflow.sbp.sbp).


Data re-distribution is commonly seen in parallel computing, i.e., changing the distributed way of data, such as gathering all data slices. In the MPI programming paradigm (SPMD), data re-distribution requires writing explicit communication operations like AllReduce, AllGather, and ReduceScatter. But in OneFlow’s Global View programming paradigm (SPSD), data re-distribution can be achieved by utilizing Global Tensor’s global data distribution type conversion.


The conversion of the global data distribution type is similar to (explicit) type conversion in general programming languages. Users only need to specify the targeted type when they convert types, and some implicit operations can be executed automatically. For example, when converting the type from double to int, the system will remove the decimal point automatically.

Similarly, it’s only required to specify the new global data distribution type that the Global Tensor will be converted into, and OneFlow will complete implicit communication operations automatically. And the interface to convert the global data distribution type is [Tensor.to_global](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_global). The `to_global` method contains two parameters- `placement` and `sbp`, which decide the newly-converted global data distribution type.

The main implicit operations in converting the global data distribution type are to infer and execute the communications, and these operations are implemented by OneFlow’s [Boxing](https://docs.oneflow.org/en/master/parallelism/03_consistent_tensor.html#boxingautomatic-conversion-of-sbp), which is a mechanism to re-distribute data automatically.


The following is a case to convert a split-distributed Global Tensor to a broadcast-distributed one:

```python
import oneflow as flow
x = flow.randn(2, 5).cuda()
placement = flow.placement(type="cuda", ranks=[0, 1])
sbp = flow.sbp.split(0)
x_global = x.to_global(placement=placement, sbp=sbp)
print(x_global.shape) # (4, 5)
print(x_global.to_local())
sbp_b = flow.sbp.broadcast
x_global_b = x_global.to_global(placement=placement, sbp=sbp_b)
print(x_global_b.shape) # (4, 5)
print(x_global_b.to_local())
```


When the global data distribution type is converted from `x_global` to `x_global_b`, the parameter `sbp` has changed from `flow.sbp.split(0)` to `flow.sbp.broadcast`. Their global shapes have remained `(4, 5)`, but the local component has turned from a data slice into complete data, and this change can be seen from the printed result of the `to_local()`.

Here, the `to_global` conversion has merged the Local Tensors. Generally speaking, SPMD programming mode requires users to write an `all-gather` collective communication to merge the Local Tensors, but in OneFlow Global View programming, the type conversion is enough to complete the merging process.  

Global Tensor’s type conversion can infer and execute the communication operations automatically. So, algorithm developers can concentrate on **thinking in data distribution** rather than **thinking in data communication operation**, and what they imagine is what they obtain, which helps them to develop distributed programs more efficiently.

Let’s add by introducing how to apply `numpy()` to the Global Tensor. For random Global Tensor, such as `x_global`, `x_global.numpy()` is equivalent to `x_global.to_global(spb=flow.sbp.broadcast).to_local().numpy()`, which means `x_global.numpy()` will firstly convert the original Global Tensor to one, which SBP is flow.sbp.broadcast(), then conduct a `to_local ` operation and finally invoke `numpy()` for the Local Tensor. Therefore, the `x_global.numpy()` method can obtain complete data.

## Global Tensor Participating in Computation

This section introduces how the Global Tensor participates in practical computation. Take the Global Tensor participating in matrix multiplication computation for example, please firstly create the following program:

```python
import oneflow as flow
placement = flow.placement(type="cuda", ranks=[0, 1])
x = flow.randn(4, 5, placement=placement, sbp=flow.sbp.split(dim=0))
w = flow.randn(5, 8, placement=placement, sbp=flow.sbp.broadcast)
y = flow.matmul(x, w)
print(y.is_global)  # True
print(y.shape)  # (4, 8)
print(y.sbp)  # (flow.sbp.split(dim=0))
print(y.to_local().numpy())
```


In the program above, we have created 2 Global Tensors-`x` and `w`, and they participate in `oneflow.matmul` computation and generate `y`. 

Most of OneFlow’s operators support computing the Global Tensor. When `flow.matmul` executes the Global Tensor, there is nothing special about its interface. Arguably, most of OneFlow’s operators are polymorphic, so they can decide how to compute according to the input:

- If the input of the operator is a Local Tensor, the operator will compute the tensor in normal single-device single-GPU execution mode.
- If the input of the operator is a Global Tensor, the operator will compute the tensor in global view (multi-device multi-GPU) mode.



The operators supporting polymorphic execution are very convenient for users to change the single-GPU code into distributed code: they only need to convert the (Local) Tensor they accept to a Global Tensor.

Just like single-device execution requires the data to be input into the same device, in the program above, the premise of the operator being executed successfully is that `x` and `w` have the same placement.

The result of matrix multiplication-`y` is also a Global Tensor. When `flow.matmul` computes `x` and `w`, it will automatically infer the placement and SBP of the output data. The following are the principles: 

- Placement: The input data and the output data have the same placement;
- SBP: The inference principle of the output data's SBP is decided by the operator type, and this principle is built into OneFlow. For more details, please refer to [SBP Signature](../parallelism/02_sbp.md#sbp-signature).


Here, the multiplied result of `flow.sbp.split(0)` and `flow.sbp.broadcast` will be inferred as `flow.sbp.split(0)`. `x` is a data slice in each rank, `w` complete data, and `y` a data slice. Anyone familiar with common parallel execution approaches will find that a forward computation with data parallelism is conducted here. `x` is a data slice, and `w` the complete parameters.

## Conclusion

This article has discussed:

- Global View offers the SPSD programming view;
- Global Tensor is visible on all processes when being executed;
- Global Tensor and Local Tensor are mutually convertible;
- Global Tensor supports converting the global data distribution type to implement distributed communication;
- OneFlow operators are polymorphic enough to enable the execution of the Global Tensor;

So, this article will come to a close, and it fisrtly introduces how to create a Global Tensor and finally explains the detailed steps for data parallelism computation that is based on a Global Tensor.

More about parallelism ways and SBP's inference logic will be discussed in our later articles. 

## 扩展阅读

## Further Reading

### OneFlow 多机多卡启动 和 依赖的环境变量

### OneFlow’s multi-machine multi-GPU launching and its required environment variables

OneFlow’s Global Tensors are executed under ** Multi-Client mode**, which means each device corresponds to one process. For example, `n Machine m GPU` has `n * m` processes. Besides, each process has its own rank ID, which corresponds to the ranks of the Global Tensor's `placement` parameter.

Take `2 Machines 2 GPUs` for example, Machine 0 corresponds to GPU 0 and GPU 1, and Machine 1 corresponds to GPU 2 and GPU 3. So, `flow.placement(type="cuda", ranks=[2])` can only identify the GPU 0 on Machine 1.


Generally, in the `n Machine m GPU` environment, `flow.placement(type="cuda", ranks=[k])` only identifies the GPU `k % m` on Machine `k / n`.

Because the Multi-Client mode is adopted , we need to launch different processes corresponding to each device. In OneFlow, all processes need to launch the same scripts, and different processes distinguish process ID and establish communications according to different environment variables.


Notes of environment variables:

- `MASTER_ADDR`：the IP of Machine 0 under multi-machine training;
- `MASTER_PORT`：the listening port of Machine 0 under multi-machine training, and this port shouldn’t conflict with the occupied ports;
- `WORLD_SIZE`: the number of computing devices in the whole cluster. Because it’s still not feasible to configure different number of GPUs on each device, the `WORLD_SIZE` equals the machine numbers multiplies the GPU numbers on each machine. In the previous case, we [create the Global Tensor](#global-tensor_2) in single machine 2 GPUs environment, so the `WORLD_SIZE=2`;
- `RANK`：the process ID of all devices in the whole cluster;
- `LOCAL_RANK`：the process ID of single device;


Differences between `RANK` and `LOCAL_RANK`: 

- For single machine training, including single-machine single-GPU and single-machine multi-GPU, `RANK` equals to `LOCAL_RANK`;
- For multi-machine training, the upper limit to `LOCAL_RANK` for each device is the number of computing devices on each machine; the upper limit to `RANK` is the sum of computing devices on all machines, and all devices are numbered from 0. (Because these computing devices are numbered from 0, the upper limit doesn’t exist.)


Take `2 Machines 2 GPUs` for example, the corresponding relationship between `LOCAL_RANK` and `RANK` for each GPU is listed as follows:


|               | RANK | LOCAL_RANK |
| ------------- | ---- | ---------- |
| GPU 0 on Machine 0 | 0    | 0          |
| GPU 0 on Machine 1 | 1    | 1          |
| GPU 1 on Machine 0 | 2    | 0          |
| GPU 1 on Machine 1 | 3    | 1          |


Although it is complicated to utilize environment variables launching, this approach is widely applicable because users can adopt random ways to launch the processes.

Besides, OneFlow also offers a convenient tool, [oneflow.distributed.launch](../parallelism/04_launch.md), to help users launch multiple processes in a distributed way and construct environment variables automatically.
