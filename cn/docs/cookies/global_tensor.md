# 使用 Global Tensor 进行多机多设备编程

By [YaoChi](https://github.com/doombeaker), [Xu Xiaoyu](https://github.com/strint), [Zuo Yihao](https://github.com/Alive1024), [Guoliang Cheng](https://github.com/lmyybh)

Global Tensor 是为了方便多机多设备分布式执行的 Tensor，是实现全局视角（Global View）编程的接口。

在大部分编程语言中，Global 的含义是进程内的全局可见，比如[全局变量（Global Variable）](https://en.wikipedia.org/wiki/Global_variable)。但是 Global Tensor 中的 “Global” 的含义是进程间全局可见，所以 Global Tensor 更为准确的的说法是 Global (on all processes) Tensor，即所有进程可见的 Tensor。使用 Global Tensor 时，在每个进程（也叫 Rank）有一个对应设备（如 GPU），在所有进程被算子执行时，算子就会自动完成对该 Tensor 的多机多设备分布式执行。

当前常用的 Tensor，只在单个进程内可见，存在于一个设备设备上。为了区分，会把这种 Tensor 叫做 Local Tensor。Local 是相对 Global 而言的，所以 Local Tensor 可以认为是 Local (on one process) Tensor。

在 OneFlow 中，对于同一个算子，大部分都同时支持输入 Local Tensor 和 Global Tensor。输入 Local Tensor 时，进行的是单进程单设备执行；但是输入 Global Tensor 时，就进行的是多进程多设备执行。Local Tensor 可以便捷地转化为 Global Tensor。如此，单机单卡执行的代码可以平滑地转换成多机多卡执行的代码。

使用 Global Tensor，可以非常便捷地进行多机多卡的模型开发，相比使用原始通信算子，可以成倍提高超大模型的开发效率。

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

### 由 Local Tensor 转换得到 Global Tensor

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

然后定义 placement 为 rank 0 和 1 上的 cuda 设备，SBP 为 tensor 第 0 维的切分，原本 Local Tensor 经过 `to_global` 变换后，就变成一个名为 `x_global` 的 Global Tensor。

可以观察到 `x_global` 的 shape 变为了 `(4, 5)`，这里显示的 shape 是 Global Tensor 的 shape。Local Tensor 的 `to_global` 方法提供了 tensor 类型的转换，原本的 Local Tensor 是要转换成的总量（Global Tensor） 在本 rank 的分量（Local Tensor）。分量和总量的关系是在 placement 上按 SBP 转换而来的，比如这里 `x` 和 `x_global` 的关系是在 0 和 1 号 GPU 上，按 `x_global` 的第 0 维 split 而得到 `x`。因此 `to_global` 可以从 x 的 shape 推理出 x_global 的 shape：把原 Local Tensor 的 shape 在第 0 维拼接。这里说的 Global Tensor 的 shape，准确地讲是 global shape。

Global Tensor 除了 shape，还有数据部分。对于一个 Global Tensor 的内部，在每个 rank 上都内含了一个 Local Tensor 作为其本地分量，这个 Local Tensor 就是 Global Tensor 在每个 rank 的物理数据。这也是我们期待的，物理上每个 rank 只需要保存一个分量的数据。
### 由 Global Tensor 得到 Local Tensor
如果想得到 Global Tensor 的本地分量，可以通过 [to_local](https://oneflow.readthedocs.io/en/master/tensor.html#oneflow.Tensor.to_local) 方法得到这个对应的 Local Tensor。接上面的程序，增加 `print(x_global.to_local())`，在不同的 rank 分别得到一个 shape 为 `(2, 5)` 的本地分量 tensor。

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

### 由 Global Tensor 转成 另外一个 Global Tensor

进行分布式计算通常都需要在正常的计算逻辑之间插入通信操作，而使用 OneFlow 时只需要做 Global Tensor 的转换。因为 Global Tensor 中的 `sbp` 参数指定了数据的分布情况：
- s，即 split(dim)， 表示在 dim 维度切分的分布关系；

- b，即 broadcast，表示广播的数据分布关系；

- p，即 partial_sum，表示 element-wise 的部分累加分布关系；

详情参考 [SBP](https://docs.oneflow.org/master/parallelism/02_sbp.html#sbp)。

因为 Global Tensor 中含有数据分布的信息，如果需要变成另外一种数据分布，只需要创建另外一个 Global Tensor 就好了。创建另外一个 Global Tensor 的过程，其中需要的通信会被自动推理和执行，从而避免了手写通信操作。自动推理并执行通信背后依赖的是 OneFlow 的 [Boxing](https://docs.oneflow.org/master/parallelism/03_consistent_tensor#boxing-sbp)，一种自动做数据 re-distribution 的机制。

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

可以看到，`x_global` 到 `x_global_b` 的变化就是 SBP 从 `flow.sbp.split(0)` 变成了 `flow.sbp.broadcast`。他们的 global shape 都是 `(4, 5)`，但是本地分量从一个分片变成了一个完整的数据，这个变化可以从对 `to_local()` 的打印结果观察到。这里的 `to_global` 变换完成了物理数据的归并，通常来讲，这需要用户手写一个 `all-gather` 集合通信来完成同样的操作，而在 OneFlow Global Tensor 中，这个通信操作的推理和执行被自动完成了，用户只需要指定期望的 Global Tensor 的数据分布就好。

通过指定期望的数据分布，就自动完成通信操作的推理和执行。让算法开发者可以 `thinking in data distribution` 而不是 `thinking in data communication operation`，从而极大提高分布式程序的开发效率。

这里补充介绍一下 Global Tensor 的 `numpy()` 方法，对于任意的 Global Tensor 如 `x_global`，`x_global.numpy()` 等价于 `x_global.to_global(spb=flow.sbp.broadcast).to_local().numpy()`，即内部隐含了一次将原 Global Tensor 转成 SBP 为 flow.sbp.broadcast() 的 Global Tensor，然后进行一次 to_local 操作，最后对这个 Local Tensor 调用 `numpy()` 方法。所以 `x_global.numpy()` 得到的是一个完整的数据。

### global tensor 参与计算
前面了解了global tensor的概念、创建、和 local tensor 的转换、和 global tensor 的转换，这部分介绍 global tensor 如何参与实际计算。这里以 global tensor 参与乘法计算为例，构造如下程序：
```python
import oneflow as flow

placement = flow.placement(type="cuda", ranks=[0,1])
x = flow.randn(4, 5, placement=placement, sbp=flow.sbp.split(dim=0))
w = flow.randn(5, 8, placement=placement, sbp=flow.sbp.broadcast)
y = flow.matmul(x, w)
print(y.is_global) # True
print(y.shape) # (4, 8)
print(y.sbp) # (flow.sbp.split(dim=0))
print(y.to_local().numpy())
```
上面的程序先创建了两个 global tensor，分别是 `x` 和 `w`，然后他们参与 `flow.matmul` 计算得到 y。

OneFlow 中的大部分算子都支持对 global tensor 的计算，所以可以看到 `flow.matmul` 在接口上并无特殊之处。可以认为 OneFlow 中的算子都是多态的，在输入 local tensor 时，采用的是普通的单卡执行模式，而输入 global tensor 时，采用的是特殊的 global view（多机多设备分布式）执行模式。这个特性对于把单卡代码改成分布式代码提供了极大便利：只需要把输入部分的 tensor 转换成 global tensor 就可以完成分布式程序改造的主要工作。

上面的程序中的 `flow.matmul` 的输出 `y` 也是一个 global tensor。这个算子可以顺利执行，有个前置条件是输入的 `x` 和 `w` 的 placement 相同，这个约束和单设备执行时要求设备相同类似。中间计算时，`flow.matmul` 算子会自动做输出的 `y` 的 placement 和 sbp 的推理：
- placement，输出和输入的 placement 相同；
- spb，输出的 sbp 的推理规则，因算子类型而异，类似每个算子输出的数据的 shape 也因算子而异，这个推理规则是 OneFlow 内置的；

在这里，flow.sbp.split(0) 和 flow.sbp.broadcast 相乘的输出数据会被推理成 flow.sbp.split(0)。`x` 在每个 rank 上是一个分片数据，`w` 是一个完整的数据，做完乘法得到的 `y` 是一个分片的数据。看到这里，了解常见并行方式的朋友应该可以发现：这里实现了一个数据并行的前向计算，`x`是切片的数据，`w`是完整的参数数据。

到此，本文从用Global Tensor的创建开始，完成了一个数据并行的算子计算。更多并行方式和SBP的推理逻辑，在后面内容继续展开。
## 扩展阅读

### OneFlow 多机多卡启动 和 依赖的环境变量
OneFlow Global Tensor 执行采用多客户端模式(Multi-Client)，即每个设备对应一个进程。n 机 m 卡 的环境，就对应 n * m 个进程。每个进程都有一个进程 rank 编号，global tensor 中的 placement 参数中的 ranks 对应的就是这个 rank 编号。进程 rank 编号隐式的也是设备编号，rank 编号加上设备类型就能标识一个设备，比如flow.placement(type="cuda", ranks=[k]) 就会对应上 k / m 号机器的编号为 k % m 的 cuda 设备。

因为采用多客户端模式，所以需要给每个设备对应启动一个进程。在 OneFlow 中，把一个同样的脚本程序，启动多次就好了，唯一需要注意的是，每个脚本程序的进程启动需要不同的环境变量，以区分进程编号和建立通信连接。

使用环境变量启动参数虽然参数繁琐，但是适用性广，可以采用任意的方式来启动进程，只要提供好 OneFlow 分布式执行提供的环境变量就好。另外为了方便使用，OneFlow 也提供了一个分布式启动多进程且自动构建环境变量的工具 [oneflow.distributed.launch](./04_launch.md)。这里主要说明采用环境变量的启动方式：
- `MASTER_ADDR`：多机训练的第0号机器的 IP；
- `MASTER_PORT`：多机训练的第0号机器的监听端口，不与已经占用的端口冲突即可；
- `WORLD_SIZE`：整个集群中计算设备的数目，因为目前还不支持各个机器上显卡数目不一致，因此 `WORLD_SIZE` 的数目实际上是 $机器数目 \times 每台机器上的显卡数目$。如我们这个例子中，是单机2卡的情况，因此 `WORLD_SIZE=2`

`RANK` 和 `LOCAL_RANK` 都是对计算设备的编号，不同的是 `RANK` 是集群内所有机器下的进程编号，`LOCAL_RANK` 某个机器内的进程编号。当是单机训练（单机单卡或单机多卡）时，两者相等。以上的例子中，有两个显卡，分别是0号和1号。

当是多机训练时，每台机器上的 `LOCAL_RANK` 的上限，就是每台机器上的计算设备的数目；`RANK` 的上限，就是所有机器上所有计算设备的总和，它们的编号均从0开始。（因为编号从0开始，所以不包含上限）。

以两台机器、每台机器上有两张显卡为例，可以整理出每张显卡的 `LOCAL_RANK` 与 `RANK` 对应情况：

|                  | RANK | LOCAL_RANK |
| ---------------- | ---------- | ---- |
| 机器0的第0张显卡 | 0          | 0    |
| 机器0的第1张显卡 | 1          | 1    |
| 机器1的第0张显卡 | 2          | 0    |
| 机器1的第1张显卡 | 3          | 1    |
