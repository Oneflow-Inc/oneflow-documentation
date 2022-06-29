# 使用 Global Tensor 进行多机多设备编程
By [YaoChi](https://github.com/doombeaker), [Xu Xiaoyu](https://github.com/strint)

Global Tensor 是为了方便多机多设备分布式执行的 Tensor，是实现全局视角（Global View）编程的接口。

在大部分编程语言中，Global 的含义是进程内的全局可见，比如[全局变量（Global Variable）](https://en.wikipedia.org/wiki/Global_variable)。但是 Global Tensor 中的 Global 的含义是进程间全局可见，所以 Global Tensor 更为准确的的说法是 Global (on all processes) Tensor，即所有进程可见的 Tensor. Global Tensor 使用中会在每个进程（也叫 Rank）有一个对应设备（如 GPU），在所有进程被算子执行时，算子就会自动完成对该 Tensor 的多机多设备分布式执行。

当前常见使用的 Tensor，是进程内可见的，存在一个设备设备上。为了区分，会把这种 Tensor 叫做 Local Tensor。Local 是 相对 Global 而言的，所以 Local Tensor 可以认为是 Local (on one process) Tensor。

在 OneFlow 中，同一个算子，大部分都同时支持输入 Local Tensor 和 Global Tensor。输入 Local Tensor 时，进行的是单进程单设备执行；但是输入 Global Tensor时，就进行的是多进程多设备执行。Local Tensor 可以便捷的转化为 Global Tensor。如此，单机单卡执行的代码可以平滑的转换成多机多卡执行的代码。

使用 Global Tensor，可以非常便捷的进行多机多卡的模型开发，相比使用原始通信算子，可以成倍的提高超大模型的开发效率。

## 创建 Global Tensor

现在尝试在2张 GPU 显卡的主机上创建一个 global tensor。以 `randn` 算子为例，创建一个 Python 文件`test_randn_global.py`，加入以下内容：
```
import oneflow as flow

# Place a global tensor on cuda device of rank(process) 0 and 1
placement = flow.placement("cuda", [0,1]) 
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

最后，在两个 Terminal 下个个启动一下`test_randn_global.py`观察 global tensor 的创建结果：
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

## 扩展阅读

### 多机训练时的环境变量

本文的例子，通过设置环境变量配置分布式训练，仅仅是为了在交互式 Python 环境下方便查看实验效果。
如果不是学习、试验目的，而是生产需求，可以直接通过 [oneflow.distributed.launch](./04_launch.md) 启动分布式训练，该模块内部根据命令行参数自动设置了必要的环境变量。

- `MASTER_ADDR`：多机训练的第0号机器的 IP
- `MASTER_PORT`：多机训练的第0号机器的监听端口，不与已经占用的端口冲突即可
- `WORLD_SIZE`：整个集群中计算设备的数目，因为目前还不支持各个机器上显卡数目不一致，因此 `WORLD_SIZE` 的数目实际上是 $机器数目 \times 每台机器上的显卡数目$。如我们这个例子中，是单机2卡的情况，因此 `WORLD_SIZE=2`

`RANK` 和 `LOCAL_RANK` 都是对计算设备的编号，不同的是 `RANK` 是“全局视角”的编号，`LOCAL_RANK` 某个特定机器上的“局部视角”的编号。当是单机训练（单机单卡或单机多卡）时，两者是没有区别的。以上的例子中，有两个显卡，分别是0号和1号。

当是多机训练时，每台机器上的 `LOCAL_RANK` 的上限，就是每台机器上的计算设备的数目；`RANK` 的上限，就是所有机器上所有计算设备的总和，它们的编号均从0开始。（因为编号从0开始，所以不包含上限）

以两台机器、每台机器上有两张显卡为例，可以整理出每张显卡的 `LOCAL_RANK` 与 `RANK` 对应情况：

|                  | RANK | LOCAL_RANK |
| ---------------- | ---------- | ---- |
| 机器0的第0张显卡 | 0          | 0    |
| 机器0的第1张显卡 | 1          | 1    |
| 机器1的第0张显卡 | 2          | 0    |
| 机器1的第1张显卡 | 3          | 1    |

### Boxing（自动转换 SBP）

我们已经通过以上代码的例子，知道一个算子会根据输入 tensor 的 SBP 属性以及算子内置的 SBP Signature，自动设置输出 tensor 的 SBP。

但是，细心的用户可能会进一步思考，如果上游算子输出 tensor 的 SBP，与下游算子输入的需要不一致时，怎么办呢？

比如，假设在模型并行中，有2层矩阵乘法，在第一层和和第二层都做模型并行。

![multi-layer-matmul](./imgs/multi-matmul.png)


因为第一层的输出的 SBP（`split(1)`），并不是第二层输入所期待的（`broadcast`），这时候，OneFlow 会自动在上一层的输出和下一层的输出之间，插入 Boxing 操作，利用集合通信进行必要的数据转换。

从 `split(1)` 转换为 `broadcast`，相当于做了一次 `AllGather` 操作。如下图所示。

![s2b](./imgs/boxing_s2b.png)

因为有 Boxing 机制的存在，使得用户只用关心少数关键地方（如 source 算子）的 SBP 设置，剩下的全部都可以交给 OneFlow 框架。
