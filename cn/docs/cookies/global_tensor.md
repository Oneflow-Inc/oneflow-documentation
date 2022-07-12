# 使用 Global Tensor 进行多机多设备编程：基础操作

By [YaoChi](https://github.com/doombeaker), [Xu Xiaoyu](https://github.com/strint), [Zuo Yihao](https://github.com/Alive1024), [Guoliang Cheng](https://github.com/lmyybh)

Global Tensor 是多机多设备执行的 Tensor，是实现全局视角（Global View）编程的接口。

当前的并行程序，大都采用单程序多数据（SPMD）的方式来编程。并行执行同样的程序，但是处理不同数据，以此实现数据的并行处理。以 PyTorch DistributedDataParallel（DDP） 为例，每个进程执行同样的神经网络计算逻辑，但是每个进程加载数据集的不同分片。

单程序多数据（SPMD）编程的缺陷是多数据的通信麻烦。在深度学习的场景下，SPMD编程需要在原计算代码中插入通信操作，比如数据并行时对梯度汇总（AllReduce 操作），模型并行时需要 AllGather/ReduceScatter 操作。如果并行模式复杂，或者需要试验新并行模式，插入通信操作就变得难以开发和维护。

全局视角（Global View）编程提供了单程序单数据（SPSD）的编程视角。和 SPMD 编程不同的是，数据从编程接口层面看也是单一的了。

数据是同一个逻辑数据，其实很自然。当我们把一个单进程程序扩展到并行执行时，一个单进程数据被扩展成多进程数据，多个进程上的这些数据都对应原单进程程序中的同一个逻辑数据。这个逻辑数据在 OneFlow 中叫 Global Tensor。

编程时，Global Tensor 让用户可以用 SPSD 的接口来编程，即按照单机单设备的逻辑视角来写程序。然后 OneFlow 框架内部会自动的转换成物理的 SPMD/MPMD 方式来做并行/分布式执行。

使用 Global Tensor，就可以采用比较自然的 Global View 视角，把多机多设备看做一机一设备来编程，实现 SPSD 编程。


## Global Tensor

在编程语言中，Global 的含义通常是进程内的全局可见，比如[全局变量（Global Variable）](https://en.wikipedia.org/wiki/Global_variable)。

但是 Global Tensor 中的 “Global” 的含义是进程间全局可见，所以 Global Tensor 更为准确的的说法是 Global (on all processes) Tensor，即所有进程可见的 Tensor。

Global Tensor 在每个进程上都存在，在所有进程上被某算子执行时，就自动完成了对该 Tensor 的多机多设备执行。

当前常用的 Tensor，只在单个进程内可见，存在于一个设备上，OneFlow 中把这种 Tensor 叫做 Local Tensor。Local 是相对 Global 而言的，所以 Local Tensor 可以认为是 Local (on one process) Tensor。

OneFlow 的算子大部分兼容 Local Tensor 和 Global Tensor 的执行。Local Tensor 可以便捷地转化为 Global Tensor。如此，单机单卡执行的代码可以平滑地转换成多机多卡执行的代码。

使用 Global Tensor，可以非常便捷地进行多机多卡的模型开发，相比使用原始通信算子，可以成倍提高并行执行模型的开发效率。

## 创建 Global Tensor

现在尝试在有 2 张 GPU 的主机上创建一个 Global Tensor。以 `randn` 算子为例，创建一个 Python 文件 `test_randn_global.py`，加入以下内容：

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

在上述代码中有一些新出现的概念：

- `placement` 表示 Global Tensor 分布的物理设备，参数 `type` 指定了物理设备的类型，这里使用` “cuda”` 表示 GPU 设备，参数 `ranks` 指定了设备 ID。对于没有 2 张 GPU 的读者，在这里可以将 `type` 指定为 `"cpu"`，这样可以使用 CPU 模拟多个设备，下文的代码同样适用；
- `sbp` 表示 Global Tensor 分布的方式，代码中的 `sbp = flow.sbp.split(dim=0)` 表示把 Global Tensor 在维度 0 均匀切分；
- `to_local()` 可以从 Global Tensor 中获取其在当前 rank 的 Local Tensor，因为 Global Tensor 在每个 rank 都内含了一个 Local Tensor 作为实际存在的本地分量。

然后配置下多进程启动依赖的环境变量。这里是两卡执行，对应两个进程启动，所以需要打开两个 Terminal，分别配置如下环境变量：
!!! Note
    分别 **点击** 以下 Terminal 0 或 Terminal 1 标签，查看 2 个控制台的命令/代码

=== "Terminal 0"

    ```shell
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0
    ```

=== "Terminal 1"

    ```shell
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=2 RANK=1 LOCAL_RANK=1
    ```

以上详细解释及借助工具启动分布式，请参考文末的 [扩展阅读](#_5)。

最后，在两个 Terminal 下分别启动一下`test_randn_global.py`，观察 Global Tensor 的创建结果：
```
python3 test_randn_global.py
```
这样，在 Terminal 0 即 rank 0 可以看到：
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
在 Terminal 1 即 rank 1 可以看到：
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
可以发现两个 rank 的 Local Tensor 在维度 0 拼接后，就是完整的 Global Tensor 的值。

## 由 Local Tensor 得到 Global Tensor

可以先创建 Local Tensor，再利用 [Tensor.to_global](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_global) 方法，将 Local Tensor 转为 Global Tensor。

创建如下程序，采用上文同样的方式启动：

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

该程序在 2 个 GPU 设备上分别创建了 `shape=(2,5)` 的  Local Tensor，即 x。

然后定义 placement 为 rank 0 和 1 上的 cuda 设备，SBP 为 tensor 第 0 维的切分，原本 Local Tensor 经过 `to_global` 变换后，就得到一个名为 `x_global` 的 Global Tensor。

可以观察到 `x_global` 的 shape 变为了 `(4, 5)`，这是 Global Tensor 的 shape（global shape）。

Global Tensor 与 Local Tensor 之间为总量与分量的关系。Local Tensor 是总量在本 rank 的分量，分量和总量的具体关系由 Placement 和 SBP 确定。比如这里 `x` 和 `x_global` 的关系是在 0 和 1 号 GPU 上，`x_global` 在第 0 维 split 而得到 `x`。

`to_global` 方法根据如上关系可以从 x shape 推理出 x_global shape ：把两个 GPU 上的 Local Tensor x 在第 0 维拼接后得到 x_global。

Global Tensor 除了 shape，还有数据部分。一个 Global Tensor 的内部，在每个 rank 上都内含了一个 Local Tensor 作为其本地分量。 这个 Local Tensor 就是 Global Tensor 在每个 rank 的物理数据。这也是我们期待的，物理上每个 rank 只需要保存一部分数据。

## 由 Global Tensor 得到 Local Tensor

如果想得到 Global Tensor 的本地分量，可以通过 [to_local](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_local) 方法得到。

接上面的程序，增加 `print(x_global.to_local())`，在不同的 rank 分别得到一个 shape 为 `(2, 5)` 的本地分量 tensor。

=== "Terminal 0"

    ```python
    tensor([[-0.2730,  1.8042,  0.0721, -0.5024, -1.2583],
        		[-0.3379,  0.9371,  0.7981, -0.5447, -0.5629]],
       		 dtype=oneflow.float32)
    ```

=== "Terminal 1"

    ```python
    tensor([[ 0.6829,  0.4849,  2.1611,  1.4059,  0.0934],
    		    [-0.0301, -0.6942, -0.8094, -1.3050, -0.1778]],
       		 dtype=oneflow.float32)
    ```

`to_local()` 没有任何参数，是因为 Global Tensor 已经通过 placement 和 SBP 指定好了它的本地分量的信息。

## 由 Global Tensor 转成另一个 Global Tensor

进行分布式计算通常都需要在正常的计算逻辑之间插入通信操作，而使用 OneFlow 时只需要做 Global Tensor 的数据分布类型转换。

Global Tensor 相比普通的 Local Tensor，从类型上讲，最大的区别是带有全局数据分布类型（Global Data Distribution Type）。全局数据分布类型指定了 Global Tensor 在每个进程（Rank）的数据分布情况。由 Placement 和 SBP 组成。

全局数据分布类型中的 Placement 指定了数据分布的设备集合:

- 参数 `type` 指定了物理设备的类型，`cuda` 表示 GPU 设备内存, `cpu` 表示 CPU 设备内存；
- 参数 `ranks` 指定了进程 ID 集合，因为隐含了一个 Rank 对应一个物理设备，所以 `ranks` 就是设备 ID 集合; 实际上 `ranks` 是一个由 rank id 组成 nd-array，支持高维设备排布。 

详情参考 [oneflow.placement](https://oneflow.readthedocs.io/en/master/tensor_attributes.html?highlight=placement#oneflow.placement).


全局数据分布类型中的 SBP 指定了全局数据和局部数据的关系:

- S，即 split(dim)，局部和全局是切分关系， 表示在 dim 维度做了切分的数据分布关系；

- B，即 broadcast，局部和全局是广播关系，表示做了广播的数据分布关系；

- P，即 partial_sum，局部和全局是部分关系，表示做了 element-wise 累加的数据分布关系；

详情参考 [oneflow.sbp.sbp](https://oneflow.readthedocs.io/en/master/tensor_attributes.html?highlight=placement#oneflow.sbp.sbp).

数据重分布（Re-distribution)是并行计算中经常要处理的，即变换数据分布，比如把分片数据聚合到一起。在 MPI 编程范式（SPMD）下, 数据重分布需要写显式的通信操作，如 AllReduce、AllGather、ReduceScatter。在 OneFlow 的 Global View 编程范式（SPSD) 下，数据重分布可以通过 Global Tensor 的全局数据分布类型转换完成。

全局数据分布类型转换类似常规编程语言中的（显式）类型转换。类型转换时，只需指定要变换到的类型，里面隐含的操作会被系统自动完成。比如 double 类型到 int 类型的转换，去掉小数点部分的操作就是系统自动完成的。

同样，只需指定 Global Tensor 要转换的新全局数据分布类型，里面隐含的通信操作会被 OneFlow 自动完成。全局数据分布类型转换的接口是 [Tensor.to_global](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_global)，`to_global` 有 `placement` 和 `sbp` 两个参数，这两个参数即期望转换成的新全局数据分布类型。 

全局数据分布类型转换中隐含的主要操作是自动推理并执行通信，背后的实现机制是 OneFlow 的 [Boxing](https://docs.oneflow.org/master/parallelism/03_consistent_tensor#boxing-sbp)，一种自动做数据 Re-distribution 的机制。

下面看一个例子，该例子可以把一个按 split 分布的 Global Tensor 转换为一个按 broadcast 分布的 Global Tensor：

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

可以看到，`x_global` 到 `x_global_b` 的全局数据分布类型变化就是 sbp 从 `flow.sbp.split(0)` 变成了 `flow.sbp.broadcast`。他们的 global shape 都是 `(4, 5)`，但是本地分量从一个分片变成了一个完整的数据，这个变化可以从对 `to_local()` 的打印结果观察到。

这里的 `to_global` 变换完成了对 local tensor 的归并。通常来讲，SPMD 编程模式要求用户手写一个 `all-gather` 集合通信来完成。而在 OneFlow Global View 中，只需做一下全局数据分布类型变换。

通过 Global Tensor 的类型变换，就自动完成通信操作的推理和执行。让算法开发者可以 `思考数据的分布`(`Thinking in data distribution`)，而不是 `思考如何通信`(`Thinking in data communication operation`)，实现了所想即所得，从而提高分布式程序的开发效率。

这里补充介绍一下 Global Tensor 的 `numpy()` 方法。对于任意的 Global Tensor 如 `x_global`，`x_global.numpy()` 等价于 `x_global.to_global(spb=flow.sbp.broadcast).to_local().numpy()`，即内部隐含了一次将原 Global Tensor 转成 SBP 为 flow.sbp.broadcast() 的 Global Tensor，然后进行一次 to_local 操作，最后对这个 Local Tensor 调用 `numpy()` 方法。所以 `x_global.numpy()` 得到的是一个完整的数据。

## Global Tensor 参与计算

上文了解了 Global Tensor 的基本概念、如何创建 Global Tensor、Global 与 Local Tensor 的转换以及 Global Tensor 之间的转换。

这一节将介绍 Global Tensor 如何参与实际计算。这里以 Global Tensor 参与矩阵乘法计算为例，构造如下程序：

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

以上程序创建了两个 Global Tensor，分别是 `x` 和 `w`，它们参与 `flow.matmul` 计算得到 `y`。

OneFlow 中的大部分算子都支持计算 Global Tensor，所以 `flow.matmul` 在接口上并无特殊之处。可以认为 OneFlow 中的算子都是多态的。即，会自动根据输入，决定自己的行为：

- 如果算子的输入是 Local Tensor，那么算子会按照普通的单卡执行模式进行计算；
- 如果算子的输入是 Global Tensor，那么算子会采用 Global View（多机多设备分布式）模式进行计算；

当用户需要将单卡代码改为分布式代码时，算子支持多态执行这个特性为用户提供了极大的便利：只需要把输入的 (Local) Tensor 转换成 Global Tensor 。

类似于单卡执行时要求输入数据所在设备相同，以上程序中， `flow.matmul` 这一算子可以顺利执行的前置条件是：输入的 `x` 和 `w` 的 placement 相同。

程序中矩阵相乘的结果 `y` 同样是一个 Global Tensor 。`flow.matmul` 对输入 `x` 和 `w` 做中间计算时，会自动进行输出的 placement 和 SBP 的推理，规则如下：

- Placement: 输出和输入的 placement 相同；
- SBP: 输出的 SBP 的推理规则，因算子类型而异，这个推理规则是 OneFlow 内置的，详情可见: [SBP Signature](../parallelism/02_sbp.md#sbp-signature)

此处，`flow.sbp.split(0)` 和 `flow.sbp.broadcast` 相乘的输出数据会被推理成 `flow.sbp.split(0)`。`x` 在每个 rank 上是一个分片数据，`w` 是一个完整的数据，二者矩阵乘法得到的 `y` 是一个分片的数据。看到这里，了解常见并行方式的朋友可以发现：这里实现了一个数据并行的前向计算，`x` 是切片的数据，`w` 是完整的参数。

## 结语
上文介绍了：

- Global View 提供的 SPSD 编程视角；
- Global Tensor 的跨进程可见的执行特点；
- Global Tensor 和 Local Tensor 的互转；
- 通过 Global Tensor 的全局数据分布类型转换来实现分布式通信；
- OneFlow 算子的多态特性支持 Global Tensor 的执行；

至此，本文从 Global Tensor 的创建开始，最终完成了一个基于 Global Tensor 的数据并行计算流程。

更多并行方式和 SBP 的推理逻辑，将在后续内容介绍。

## 扩展阅读

### OneFlow 多机多卡启动 和 依赖的环境变量

OneFlow 的 Global Tensor 执行采用的是 **多客户端模式 (Multi-Client)**，每个设备对应一个进程。`n 机 m 卡` 的环境，就对应 `n * m` 个进程。每个进程都有一个进程 rank 编号，Global Tensor 中的 placement 参数中的 ranks 对应的就是这个 rank 编号。

以 `2 机 2 卡` 为例， 0 号机器中两张卡分别对应编号 0 和 1，第 1 号机器中两张卡分别对应编号 2 和 3。此时 `flow.placement(type="cuda", ranks=[2])` 可以唯一标识第 1 号机器中的第 0 卡。

一般地，对于 `n 机 m 卡` 的环境，`flow.placement(type="cuda", ranks=[k])` 唯一标识第 `k / n` 号机器的第 `k % m` 张卡。

因为采用多客户端模式，所以需要对应每个设备都启动一个进程。在 OneFlow 中，所有进程都只需要启动相同的脚本程序，不同进程之间通过不同的环境变量配置区分进程编号和建立通信连接。

环境变量说明：

- `MASTER_ADDR`：多机训练的第 0 号机器的 IP
- `MASTER_PORT`：多机训练的第 0 号机器的监听端口，不与已经占用的端口冲突即可
- `WORLD_SIZE`：整个集群中计算设备的数目，因为目前还不支持各个机器上显卡数目不一致，因此 `WORLD_SIZE` 的数目实际上是 $机器数目 \times 每台机器上的显卡数目$。[创建 Global Tensor](#创建-global-tensor) 中的示例是单机 2 卡的情况，因此 `WORLD_SIZE=2`
- `RANK`：集群内所有机器下的进程编号
- `LOCAL_RANK`：单个机器内的进程编号

`RANK` 和 `LOCAL_RANK` 的区别：

- 当是单机训练（单机单卡或单机多卡）时，两者相等；
- 当是多机训练时，每台机器上的 `LOCAL_RANK` 的上限，就是每台机器上的计算设备的数目；`RANK` 的上限，就是所有机器上所有计算设备的总和，它们的编号均从0开始。（因为编号从0开始，所以不包含上限）。

以 `2 机 2 卡` 为例，每张显卡的 `LOCAL_RANK` 与 `RANK` 对应情况如下：

|                      | RANK | LOCAL_RANK |
| -------------------- | ---- | ---------- |
| 机器 0 的第 0 张显卡 | 0    | 0          |
| 机器 0 的第 1 张显卡 | 1    | 1          |
| 机器 1 的第 0 张显卡 | 2    | 0          |
| 机器 1 的第 1 张显卡 | 3    | 1          |

使用环境变量启动参数虽然参数繁琐，但是适用性广，可以采用任意的方式来启动进程，只要提供好 OneFlow 分布式执行提供的环境变量就好。另外为了方便使用，OneFlow 也提供了一个分布式启动多进程且自动构建环境变量的工具 [oneflow.distributed.launch](../parallelism/04_launch.md)。
