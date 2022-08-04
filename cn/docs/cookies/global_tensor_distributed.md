# 使用 Global Tensor 进行多机多设备编程：分布式并行策略

> 首先简单介绍分布式训练的重要性

深度学习是通过神经网络学习样本数据的内在规律和表现层次的一种复杂机器学习算法。计算过程主要涉及数据和模型两部分。

随着深度学习的广泛应用，模型规模不断扩大，对硬件（算力、内存）的需求也在不断提高。然而，受限于物理定律，持续提高芯片的集成越来越困难，单一设备的算力及容量难以跟上模型扩大的需求。

为解决算力增速不足的问题，多节点集群的分布式训练方式逐渐受到重视，高效易用的分布式并行策略的提出势在必行。


## 并行策略

值得注意的是，简单的设备堆叠并不一定会带来算力的增长。因为神经网络的训练并不是单纯的“把原来一个设备做的事情，现在分给多个设备各自做”，它不仅需要多个设备进行计算，还涉及到设备之间的数据传输，只有协调好集群中的计算与通信，才可以实现高效的分布式训练。

> 对三种并行方式进行简要概括

常见的并行策略包括**数据并行**、**模型并行**和**流水并行**，特点如下：

- 数据并行：对**数据**进行切分，不同设备数据不同，但模型相同
- 模型并行：对**模型**进行切分，不同设备数据相同，但模型不同
- 流水并行：将**模型**分为多个阶段，分发到不同设备，各个设备之间以“流水线”的方式完成训练

除上述三种策略外，**混合并行**也是一种常见的并行策略，通过上述两种或三种方式的混合使用完成训练目的。

> 这里考虑加一段 Global Tensor 实现并行的优势介绍（简单、高效、……）

待定

> matmul 基础代码和示例，后续所有示例都以这个为基础修改

本文以矩阵乘法为例，解释并行策略间的区别，以及如何利用 Global Tensor 实现不同的并行方式。

假设神经网络中的某一层是进行矩阵乘法计算，其中，输入 $x$ 的形状为 $4\times5$，模型参数 $w$ 的形状为 $5\times8$，那么，矩阵乘法输出形状为 $4\times8$。

基础代码：

```python
import oneflow as flow

x = flow.randn(4, 5)
w = flow.randn(5, 8)
out = flow.matmul(x, w)
print(out.shape) # (4, 8)
```

示意图如下：

![matmul](../parallelism/imgs/matmul_logical.png)

单设备的训练中，以上矩阵乘法计算得到 $out$ 后会传递到下一层，并最终计算得到 $loss$。然后，在反向传播过程中，得到 $\frac{\partial loss}{\partial w}$，用于更新 $w$。

### 数据并行

> 接下来以以例子和图片结合的方式，分别介绍各种并行策略

数据并行是将数据进行切分输入不同设备，而每个设备上的模型保持完整和一致。

OneFlow 特有的 Global Tensor 采用 `placement` 与 `sbp` 结合的方式完成分布。其中 `placement` 表示 Global Tensor 分布的物理设备，`sbp` 表示 Global Tensor 分布的方式（详情可见：[创建 Global Tensor](./global_tensor.md/#global-tensor_2)）。

以两卡数据并行为例，Global Tensor 的设计方式使得上述矩阵乘法案例的修改非常简单：

1. 数据 $x$ 按第 0 维度切分(`sbp=flow.sbp.split(dim=0)`)，分布在两卡设备上(`placement=flow.placement(type="cuda", ranks=[0, 1])`)
2. 模型 $w$ 保持完整(`sbp=flow.sbp.broadcast`)，分布在两卡设备上(`placement=flow.placement(type="cuda", ranks=[0, 1])`)

修改后，完整代码如下：

```python
import oneflow as flow

placement = flow.placement(type="cuda", ranks=[0, 1])
x = flow.randn(4, 5, placement=placement, sbp=flow.sbp.split(dim=0))
w = flow.randn(5, 8, placement=placement, sbp=flow.sbp.broadcast)
out = flow.matmul(x, w)
print(out.shape) # (4, 8)
```

数据并行示意图：

![Data Paralelism](../parallelism/imgs/matmul_data_paralelism.png)


> 这里在考虑要不要放数据并行的其他介绍，例如：
>> 数据并行策略下，在反向传播过程中，需要对各个设备上的梯度进行 AllReduce，以确保各个设备上的模型始终保持一致

>> 当数据集较大，模型较小时，由于反向过程中为同步梯度产生的通信代价较小，此时选择数据并行一般比较有优势，常见的视觉分类模型，如 ResNet50，比较适合采用数据并行。

### 模型并行

当神经网络非常巨大时，数据并行同步梯度的代价很大，此时可以考虑采用模型并行策略。

与数据并行相反，模型并行是将模型进行切分输入不同设备，而每个设备上的数据保持完整和一致。

同样以两卡为例，模型并行修改方式为：

1. 数据 $x$ 保持完整(`sbp=flow.sbp.broadcast`)，分布在两卡设备上(`placement=flow.placement(type="cuda", ranks=[0, 1])`)
2. 模型 $w$ 按第 1 维度切分(`sbp=flow.sbp.split(dim=1)`)，分布在两卡设备上(`placement=flow.placement(type="cuda", ranks=[0, 1])`)

修改后，完整代码如下：

```python
import oneflow as flow

placement = flow.placement(type="cuda", ranks=[0, 1])
x = flow.randn(4, 5, placement=placement, sbp=flow.sbp.broadcast)
w = flow.randn(5, 8, placement=placement, sbp=flow.sbp.split(dim=1))
out = flow.matmul(x, w)
print(out.shape) # (4, 8)
```

模型并行示意图：

![Data Paralelism](../parallelism/imgs/matmul_model_paralelism.png)

### 流水并行

当神经网络过于巨大，无法在一个设备上存放时，可以选择流水并行策略。 流水并行将网络切分为多个阶段，并分发到不同的计算设备上，各个计算设备之间以“流水线”的方式完成训练。

以两卡流水并行为例，构造两阶段示例程序：

```python
import oneflow as flow

P0 = flow.placement(type="cuda", ranks=[0])
P1 = flow.placement(type="cuda", ranks=[1])
BROADCAST = flow.sbp.broadcast

# 模型第一阶段分布在第 0 卡
w0 = flow.randn(5, 8, placement=P0, sbp=BROADCAST)
# 模型第二阶段分布在第 1 卡
w1 = flow.randn(8, 3, placement=P1, sbp=BROADCAST)

# 随机生成数据模拟输入
x = flow.randn(4, 5)

# 利用 to_global 将第一阶段的数据分布在第 0 卡
in_stage0 = x.to_global(placement=P0, sbp=BROADCAST)
out_stage0 = flow.matmul(in_stage0, w0)
print(out_stage0.shape) # (4, 8)

# 利用 to_global 将第二阶段的数据分布在第 1 卡
in_stage1 = out_stage0.to_global(placement=P1, sbp=BROADCAST)
out_stage1 = flow,matmul(in_stage1, w1)
print(out_stage1.shape) # (4, 3)
```

以上程序采用矩阵乘法，模拟了一个两阶段神经网络。与数据并行和模型并行不同，流水并行中的数据和模型均未被切分，而是分别将两个阶段分布在不同的设备上进行计算。

Global Tensor 的设计，使得计算过程中，只需通过 `to_global` 方法调整上一阶段的输出数据的分布策略，作为下一阶段的输入数据即可。

> 这里要不要写“ Stage ID 及梯度累积设置”

### 混合并行

> 这里想的是直接放 GPT-3 示例

在网络的训练中，也可以将多种并行策略混用，以 GPT-3 为例，以下是它训练时的设备并行方案：

首先将模型分为 64 个阶段，进行流水并行。每个阶段都运行在 6 台 DGX-A100 主机上。在 6 台主机之间，进行的是数据并行训练；每台主机有 8 张 GPU 显卡，同一台机器上的 8 张 GPU 显卡之间是进行模型并行训练。

![gpt-3](../parallelism/imgs/gpt3-overview.png)

## 结语

并行策略的选择影响着训练效率，框架对并行训练的接口支持程度，决定了算法工程师的开发效率。

本文介绍了数据并行、模型并行、流水并行以及混合并行这些分布式并行策略，通过示例展示了 OneFlow 针对分布式训练所做的系统级设计和创新，以便于用户轻松上手分布式训练。
