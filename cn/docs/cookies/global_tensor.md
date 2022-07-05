# 使用 Global Tensor 进行多机多设备编程
By [YaoChi](https://github.com/doombeaker), [Xu Xiaoyu](https://github.com/strint)

Global Tensor 是为了方便多机多设备分布式执行的 Tensor，是实现全局视角（Global View）编程的接口。

在大部分编程语言中，Global 的含义是进程内的全局可见，比如[全局变量（Global Variable）](https://en.wikipedia.org/wiki/Global_variable)。但是 Global Tensor 中的 Global 的含义是进程间全局可见，所以 Global Tensor 更为准确的的说法是 Global (on all processes) Tensor，即所有进程可见的 Tensor. Global Tensor 使用中会在每个进程（也叫 Rank）有一个对应设备（如 GPU），在所有进程被算子执行时，算子就会自动完成对该 Tensor 的多机多设备分布式执行。

当前常见使用的 Tensor，是进程内可见的，存在一个设备设备上。为了区分，会把这种 Tensor 叫做 Local Tensor。Local 是 相对 Global 而言的，所以 Local Tensor 可以认为是 Local (on one process) Tensor。

在 OneFlow 中，同一个算子，大部分都同时支持输入 Local Tensor 和 Global Tensor。输入 Local Tensor 时，进行的是单进程单设备执行；但是输入 Global Tensor时，就进行的是多进程多设备执行。Local Tensor 可以便捷的转化为 Global Tensor。如此，单机单卡执行的代码可以平滑的转换成多机多卡执行的代码。

使用 Global Tensor，可以非常便捷的进行多机多卡的模型开发，相比使用原始通信算子，可以成倍的提高超大模型的开发效率。

## 创建 Global Tensor

现在尝试在2张 GPU 显卡的主机上创建一个 global tensor。以 `randn` 算子为例，创建一个 Python 文件`test_randn_global.py`，加入以下内容：
```python
import oneflow as flow

# Place a global tensor on cuda device of rank(process) 0 and 1
placement = flow.placement(type="cuda", ranks=[0,1]) 
# Each rank's local data is a part data as a result of spliting global data on dim 0
sbp = flow.sbp.split(dim=0)
# Create a global tensor by randn
x = flow.randn(4,5,placement=placement, sbp=sbp)
# Print local data
print("Local data of global tensor:\n ",x.to_local().numpy())
# Print global data
print("Global data of global tensor:\n ",x.numpy())
```
上述代码中的有一些新增的概念：
- `placement` 表示 global tensor 分布的物理设备；
- `sbp` 表示 global tensor 分布的方式，代码中 `sbp = flow.sbp.split(dim=0) 表示把 globa tensor 在维度 0 均匀切分；
- `to_local()` 可以从 global tensor 中获取其在当前 rank 的 local tensor，因为 global tensor 在每个 rank 都内含了一个 local tensor 作为实际存在的本地分量；

然后配置下多进程启动依赖的环境变量。这里是两卡执行，对应两个进程启动，所以需要打开两个 Terminal，分别配置如下环境变量：
!!! Note
    分别 **点击** 以下 Terminal 0 或 Terminal 1 标签，查看2个控制台的命令/代码

=== "Terminal 0"
    ```shell
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=2 RANK=0 LOCAL_RANK=0
    ```

=== "Terminal 1"
    ```shell
    export MASTER_ADDR=127.0.0.1 MASTER_PORT=17789 WORLD_SIZE=2 RANK=1 LOCAL_RANK=1
    ```

以上详细解释及借助工具启动分布式，请参考文末的 [扩展阅读](#_5)

最后，在两个 Terminal 下分别启动一下`test_randn_global.py`，观察 global tensor 的创建结果：
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
可以发现两个 rank 的 local tensor 在维度 0 拼接后，就是完整的 Global Tesnor的值。
### 由 local tensor 转换得到 global tensor

可以先创建 local tensor，再利用 [Tensor.to_global](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_global) 方法，将 local tensor 转为 global tensor。

创建如下程序，采用上文同样的方式启动：

```python
import oneflow as flow

x = flow.randn(2,5).cuda()
print(x.is_local) # True
print(x.is_global) # False
placement = flow.placement(type="cuda", ranks=[0,1])
sbp = flow.sbp.split(0)
x_global = x.to_global(placement=placement, sbp=sbp)
print(x_global.shape) # (4, 5)
print(x.is_local) # False
print(x_global.is_global) # True
```

该程序在2台设备上分别创建了 `shape=(2,5)` 的2个 gup 内存上的 local tensor, 即 x。

然后定义 placement 为 rank 0 和 1 上的 cuda 设备，sbp 为 tensor 第 0 维的切分，原本 local tensor 经过 `to_global` 变换后，就变成一个 global tensor 名为 x_global.

可以观察到 x_global 的 shape 变为了 `(4, 5)`，这里显示的 shape 是 global tensor 的 shape。local tensor 的 `to_global` 方法提供了 tensor 类型的转换，含义是原本的 local tensor 是要转换成的总量（global tensor） 在本 rank 的分量(local tensor)。分量和总量的关系是在 placement 上 按 sbp 转换而来的，比如这里原 x 和 x_global 的关系是在 0 和 1 gpu 上，按 x_global tensor 的第 0 维 split 而来的 x。因此 `to_global` 可以从 x 的 shape 推理出 x_global 的 shape：把原 local tensor 的 shape 在第 0 维拼接。这里说的 global tensor 的 shape，准确的讲是 global shape。

global tensor 除了shape，还有数据部分。一个 global tensor 的内部，在每个 rank 上都内含了一个 local tensor 作为其本地分量，这个 local tensor 就是 global tensor 在每个 rank 的物理数据。这也是我们期待的，物理上每个 rank 只需要保存一个分量的数据。
### 由 global tensor 得到 local tensor
如果想得到 global tensor 的本地分量，可以通过 [to_local](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_local) 方法得到这个对应的 local tensor。接上面的程序，增加 `print(x.to_local())`，在不同的 rank 分别得到一个 shape 为 `(2, 5)` 的本地分量 tensor。

=== "Terminal 0"
    ```python
    prrint(x.to_local())
    tensor([[ 2.9186e-01, -3.9442e-01,  4.7072e-04, -3.2216e-01,  1.7788e-01],
            [-4.5284e-01,  1.2361e-01, -3.5962e-01,  2.6651e-01,  1.2951e+00]],
        device='cuda:0', dtype=oneflow.float32)
    ```

=== "Terminal 1"
    ```python
    print(x.to_local())
    tensor([[-0.4363,  0.9985, -2.5387,  0.3003,  0.3803],
            [ 0.0556, -0.8077,  1.1191, -2.1278,  0.1468]], device='cuda:1',
        dtype=oneflow.float32)
    ```
`to_local()` 没有任何参数，是因为 global tensor 已经通过 placement 和 sbp 指定好了它的本地分量的信息。
### 由 global tensor 转成 另外一个 global tensor
通常做分布式计算都需要在正常的计算逻辑之间插入通信操作，在 OneFlow 只需要做 global tensor 的转换。因为 global tensor 中的 sbp 参数指定了数据的分布情况：
- s，即 split(dim)， 表示在 dim 维度切分的分布关系；
- b，即 broadcast，表示广播的数据分布关系；
- p，即 partial_sum，表示 element-wise 的部分累加分布关系；
详情参考[SBP](https://docs.oneflow.org/master/parallelism/02_sbp.html#sbp).

因为 global tensor 中含有数据分布的信息，如果需要变成另外一种数据分布，只需要创建另外一个 global tensor就好了。创建另外一个 global tensor 的过程，其中需要的通信会被自动推理和执行，从而避免了手写通信操作。自动推理并执行通信背后依赖的是 OneFlow 的 [Boxing](https://docs.oneflow.org/master/parallelism/03_consistent_tensor#boxing-sbp)，一种自动做数据 re-distribution 的机制。

下面看一个例子，该例子可以把一个按 split 分布的 global tensor 转换为一个按 broadcast 分布的 global tensor：
```python
import oneflow as flow

x = flow.randn(2,5).cuda()
placement = flow.placement(type="cuda", ranks=[0,1])
sbp = flow.sbp.split(0)
x_global = x.to_global(placement=placement, sbp=sbp)
print(x_global.shape) # (4, 5)
print(x_global.to_local())
sbp_b = flow.sbp.broadcast
x_global_b = x_global.to_global(placement=placement, sbp=sbp_b)
print(x_global_b.shape) # (4, 5)
print(x_global_b.to_local())
```
可以看到，`x_global` 到 `x_global_b` 的变化就是 sbp 从 `flow.sbp.split(0)` 变成了 `flow.sbp.broadcast`。他们的 global shape 都是 `(4, 5)`，但是本地分量从一个分片变成了一个完整的数据，这个变化可以从对 `to_local()` 的打印结果观察到：这里的 `to_global` 变换完成了物理数据的归并。通常来讲，需要用户手写一个 `all-gather` 集合通信来完成同样的操作，而在 OneFlow Global Tensor 中，这个通信操作的推理和执行被自动完成了，用户只需要指定期望的 global tensor 的数据分布就好。

通过指定期望的数据分布，就自动完成通信操作的推理和执行。让算法开发者可以 `thinking in data distribution` 而不是 `thinking in data communication operation`，从而极大提高分布式程序的开发效率。


这里补充介绍一下 global tensor 的 `numpy()` 方法，对于任意的 global tensor 如 `x_global，x_global.numpy` 等价于 `x_global.to_global(spb=flow.sbp.broadcast).to_local().numpy()`，即内部隐含了一次原 global tensor 到 sbp 为 flow.sbp.broadcast() 的操作，加一次 to_local 操作，最后对这个 local tensor 调用 numpy() 方法。所以 x_global.numpy() 得到的是一个完整的数据。
```python

print(x_global.numpy())
```

### Global Tensor 参与计算

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

- 如果算子的输入是 Local Tensor，那么算子会按照普通的单卡执行模式进行计算
- 如果算子的输入是 Global Tensor，那么算子会采用 Global View（多机多设备分布式）模式进行计算

当用户需要将单卡代码改为分布式代码时，OneFlow 的这个特性为用户提供了极大的便利：只需要把输入的 Tensor 转换成 Global Tensor 。

类似于单卡执行时要求输入数据所在设备相同，以上程序中， `flow.matmul` 这一算子可以顺利执行的前置条件是：输入的 `x` 和 `w` 的 placement 相同。

程序中矩阵相乘的结果 `y` 同样是一个 Global Tensor 。`flow.matmul` 对输入 `x` 和 `w` 做中间计算时，会自动进行输出 `y` 的 placement 和 SBP 的推理，规则如下：

- placement: 输出和输入的 placement 相同
- SBP: 输出的 SBP 的推理规则，因算子类型而异，这个推理规则是 OneFlow 内置的，详情可见: [SBP Signature](../parallelism/02_sbp.md#sbp-signature)

此处，`flow.sbp.split(0)` 和 `flow.sbp.broadcast` 相乘的输出数据会被推理成 `flow.sbp.split(0)`。`x` 在每个 rank 上是一个分片数据，`w` 是一个完整的数据，二者矩阵乘法得到的 `y` 是一个分片的数据。看到这里，了解常见并行方式的朋友应该可以发现：这里实现了一个数据并行的前向计算，`x` 是切片的数据，`w` 是完整的参数数据。

至此，本文从 Global Tensor 的创建开始，最终完成了一个基于 Global Tensor 的数据并行计算流程。更多并行方式和 SBP 的推理逻辑，将在后续内容继续展开。

## 扩展阅读

### OneFlow 多机多卡启动 和 依赖的环境变量

OneFlow 的 Global Tensor 执行采用的是**多客户端模式 (Multi-Client)**，每个设备对应一个进程。n 机 m 卡 的环境，就对应 n * m 个进程。每个进程都有一个进程 rank 编号，Global Tensor 中的 placement 参数中的 ranks 对应的就是这个 rank 编号。

以 `2 机 2 卡` 为例，则第 0 号机器中两张卡分别对应编号 0 和 1，第 1 号机器中两张卡分别对应编号 2 和 3。此时 `flow.placement(type="cuda", ranks=[2])` 即可唯一标识第 1 号机器中的第一张卡。

一般地，对于 `n 机 m 卡` 的环境，`flow.placement(type="cuda", ranks=[k])` 唯一标识第 `k / n` 号机器的第 `k % m` 张卡。

因为采用多客户端模式，所以需要对应每个设备都启动一个进程。在 OneFlow 中，所有进程都只需要启动相同的脚本程序，不同进程之间通过不同的环境变量配置区分进程编号和建立通信连接。

环境变量说明：

- `MASTER_ADDR`：多机训练的第 0 号机器的 IP
- `MASTER_PORT`：多机训练的第 0 号机器的监听端口，不与已经占用的端口冲突即可
- `WORLD_SIZE`：整个集群中计算设备的数目，因为目前还不支持各个机器上显卡数目不一致，因此 `WORLD_SIZE` 的数目实际上是 $机器数目 \times 每台机器上的显卡数目$。[创建 Global Tensor](#创建-global-tensor) 的示例是单机2卡的情况，因此 `WORLD_SIZE=2`
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
