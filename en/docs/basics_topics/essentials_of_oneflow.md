# OneFlow系统设计

本文的主要内容如下：

* OneFlow 的设计目标

* OneFlow 的特色一：Actor 机制

* OneFlow 的特色二：SBP 机制

* 总结

## 一、OneFlow 的设计目标
OneFlow 的设计目标是追求极致的性能，特别是分布式多机多卡环境下的横向扩展性，希望能让用户使用多机多卡就像使用单机单卡一样容易，且享受线性加速的运行效率。

为什么 OneFlow 要聚焦于分布式场景的性能和易用性呢？随着深度学习的发展，模型越来越大，训练深度学习模型所需的算力会越来越高，同时模型增大的速度要大于 GPU 单卡显存扩容的速度；训练对算力的增长要求要大于 GPU 单卡算力增长的速度。单卡的计算能力和显存远远不能满足深度学习模型训练的需求，必须借助多机多卡并行加速。

若深度学习框架可以让互联的多个 GPU 协同工作好，实现线性加速比，即使每块 GPU 性能“一般”，也可满足任意规模的算力需求，这就是所谓的横向扩展性，我们坚信这是算力增长的解决之道。

但是，已有框架都是首先聚焦于单卡的用户体验，仅对适合数据并行的多机多卡场景处理的较好，即把单卡的计算图镜像复制到多机多卡上，各个卡和机器之间辅助于模型同步的模块。

对于 BERT/GPT-3 等参数量巨大的模型，用户在使用已有深度学习框架时常常会遇到多机多卡不好用、效率低下或无法实现等问题。用户做分布式训练常常需要较高的学习成本，还需要关心多机多卡之间模型的同步问题。业界为解决分布式深度学习的痛点，不仅改进深度学习框架自身，还研发了多种第三方插件，譬如 NCCL，Horovod，BytePS，HugeCTR，Mesh-tensorflow，Gpipe 等等，但仍不能满足用户极致的性能需求。

OneFlow 的核心设计理念是，让多机多卡分布式训练高效地协同运转，同时要让用户在多机多卡的训练体验上就像单卡一样简单容易。下面我们来介绍OneFlow 实现此目标最核心的两点想法，来说明 OneFlow 是如何看待分布式场景下的深度学习训练的。

## 二、Actor：一套简洁的机制解决几乎所有技术难题

关键特性：

* 去中心化调度

* 流水线

* 数据搬运是一等公民

* 传输被计算掩盖

* 控制逻辑被执行逻辑掩盖

在 OneFlow 的设计中，分为 Compile 和 Runtime 两个时期，Compile 时期把用户定义的神经网络、分布式环境信息等编译成一个静态图的执行计划 Plan，Plan 由执行单元 Actor 的描述信息组成；Runtime 时期，各个机器根据 Plan 里的 Actor 描述信息真实地创建属于自己机器的众多 Actor 实例，然后启动 Actor 运行系统。整个深度学习训练期间，OneFlow 的执行基本单元就是 Actor，Actor 对应静态执行图上的节点，Actor 之间生产、消费的数据存储在 Register 中，Actor 之间通过消息传递来协作运行。

### 1. Actor 机制实现去中心化调度
OneFlow 的运行时去中心化调度就是用 Actor 机制实现的。在整个由 Actor 构成的静态图中，没有一个中心的调度节点，每个 Actor 都只需要关心自己所需数据的生产者（上游 Actor ）和自己生产的数据的消费者（下游 Actor）即可。这样在超大规模的分布式训练场景下， **完全的去中心化** 调度可以避免中心调度的单点性能瓶颈问题。

每个 Actor 内部都有一个 **状态机** ，Actor 收发的消息、执行的情况都会改变自己的状态。需要注意的是，Register 是存储块，存放了 Actor 生产出来的数据，而消息是包含了 Register 存储块的内存地址的轻量级数据，Actor 之间传递的是消息，而不是 Register，这样就实现了 zero-copy。

当 Actor 收到了新消息，判断它执行所需要消费的 Register 已经就绪，且它将要生产的数据有空闲的 Register 可以写入时，这个 Actor 就执行（Act）一次，生产出一个 Register。

生产完以后，该 Actor 就向需要消费这个 Register 的那些消费者 Actor 们发消息，表示 “你们可以来读取我生产的数据了” ；同时该 Actor 还需要把它消费完的那些 Register 还给这些 Regsiter 的生产者 Actor 们，表示 “我用完了你们的数据，你可以回收了” 。Actor 内部的状态机如图1 所示。 

<div align="center">
    <img src="imgs/actor_state_machine.png" align='center'/>
</div>

<center>
图1 Actor 内部状态机
</center>

在 Actor 启动之后，会根据与其他 Actor 之间收发消息来切换自己的两个状态：**等待状态** 和 **执行状态** 。

一个 Actor 收到的消息一般分为几个类型：

* 上游的生产者 Actor 发来消息说：你可以来读我生产的数据了；

* 下游的消费者 Actor 发来消息说：我用完你生产的数据了。

当这个数据被所有消费者都用完以后，就可以回收成为空闲块等待下一次被该 Actor 重新生产一份新的数据。

一个 Actor 收到消息以后都会去尝试判断当前是否满足执行条件，执行条件一般有两个：

* 需要读取的数据是否都到齐了；

* 是否有空闲块可以拿来被生产。当满足执行状态以后 Actor 就开始调用自己内部的 Kernel 真实的去读写数据。

执行完毕后 Actor 会向上下游发消息：

* 向下游的消费者 Actor 发消息说：我刚生产了一块数据，你可以来读了；

* 向上游的生产者 Actor 发消息说：我刚用完了你之前发给我的数据了。

Actor 只需要关心上下游的消息就能判断自己能不能执行。每个 Actor 都通过自己内部的状态机和收发消息机制实现了 **完全去中心化** 的分布式协同工作。

### 2. Actor 机制实现流水线

上面我们介绍了 Actor 的内部状态机，Actor 之间的消息传递和数据传递是依赖 Register 实现的。一个 Actor 是否能执行，只跟两个条件相关：

* 自己消费的那些 Register 是否可读；

* 自己生产的那些 Register 是否有空闲块可写。

对于一个 Register，如果我们运行时给它分配多个空闲块，那么相邻的两个 Actor 就可以同时工作，工作时间重叠起来，这样就实现了各个 Actor 之间的流水线。理想状态下整个静态执行图的执行时间就是整个系统中是性能瓶颈的那个 Actor 运行的总时间，其余 Actor 的执行时间都被流水线掩盖起来了。

我们举一个例子来解释 Actor 机制下的流水线是如何运转起来的。图2是一个由3个 Actor（a, b, c）组成的计算图的执行时序图。其中深绿色的 Regst方块表示正在被使用的 Register 块，白色的 Regst 方块表示同一个 Register 的备用空闲块。

* 1）在 Time0 时刻，Actor a 产出了一个 Regst_a_0，Actor b 和  Actor c 由于没有可读的 Register，所以处在等待状态。假设每个 Actor的执行时间都是单位时间。

* 2）到 Time1 时刻，Actor a 给 Actor b 发消息说你可以来读我产出的 Regst_a_0 了，Actor b 收到了消息，并检查自己生产的 Register b 是否有空闲 Regst 块可用，发现有可用的 Regst_b_0，于是 Time1 时刻Actor b 执行，读取 Regst_a_0，写 Regst_b_0；同时 Actor a 还会去看自己是否有空闲块可写，发现有，Time1 时刻 Actor a 也在执行，写 Regst_a_1（这里需要说明的是，Regst_a_0 和 Regst_a_1 逻辑上是属于同一个 Register，只是空间上分成了不同的空闲块备份而已。在深度学习训练任务中，Regst_a_0 和 Regst_a_1 里存放的是同一个 operator 产出的不同batch的数据）。于是 Actor a 和 Actor b 就并行工作起来了。Actor c 由于没有数据可读，仍在等待。

* 3）到 Time2 时刻，Actor b 生产出了 Regst_b_0，于是给下游的消费者Actor c 发消息说你可以来读我生产的 Regst_b_0，同时给上游的生产者Actor a 发消息说我用完了你的 Regst_a_0。此时 Actor a 已经把刚刚生产的 Regst_a_1 又发给了 Actor b，Actor b 检查自己仍有 Regst_b_1 空闲，于是 Actor b 开始读 Regst_a_1，写 Regst_b_1；Actor c 收到 Regst_b_0，发现自己有 Regst_c_0 空闲，于是 Actor c 开始执行，读 Regst_b_0，写 Regst_c_0；Actor a 收到了 Actor b 用完还回来的 Regst_a_0，检查 Regst_a_0 所有的消费者都用完了，于是将 Regst_a_0 回收，标记为空闲块，同时 Actor a 还可以继续执行，写 Regst_a_2。

<div align="center">
    <img src="imgs/actor_time_sequence.png" align='center'/>
</div>

<center>
图2 Actor 生产消费关系和执行时序图
</center>

在上面的例子中，到了 Time2 时刻，其实 Actor a、b、c 都在工作，在深度学习训练任务中，Time2 时刻 Regst_b_0、Regst_c_0 存放的是 Batch 0 的数据，Regst_a_1、Regst_b_1 存放的是 Batch 1 的数据，Regst_a_2 存放的是 Batch 2 的数据。通过一个 Register 有多个空闲块的设计，Actor 机制就实现了流水并行。

在这里我们抛出一个更进一步深入的问题：整个数据流的执行像一个网络，数据在网络中的流动就完成了计算，如何避免生产者生产太快，消费者消费不及，以及如何避免生产者生产太慢，消费者感到饥饿的问题，这涉及到对计算、内存、传输带宽的规划，尽可能使系统的瓶颈之处最宽，需要解决流控（flow control）的问题以及资源分配问题（如每个 Actor 的 Register 到底分配几个内存块配额），这非常关键，也是 OneFlow 系统已解决的问题。

### 3. 数据搬运是一等公民

在多机多卡的分布式环境中，各个机器和各个设备之间的数据传输往往是影响系统的横向扩展性的最重要因素，如果传输开销可以被计算开销掩盖，那么分布式深度学习训练就可以达到理想的线性加速比。相较于其他的框架，OneFlow 把数据搬运视为跟数据计算同等地位的操作，提出 **数据搬运是一等公民** 的思想。

已有框架在编译期的关注焦点是数据计算，认为数据搬运是背后隐式发生的，因此在静态分析计算图时略过计算和搬运的重叠编排，OneFlow 在计算图中显式表达了数据搬运，而且在静态分析时同等对待数据搬运和数据计算，以最大化重叠搬运和计算。

在最终的执行图中，数据搬运操作也是一个个 Actor。除了在设备上做数据计算用的 Actor 以外，还有计算机内存到 GPU 显存之间的数据拷贝 Actor，机器之间做网络通信的网络 Actor，负责数据的切分、合并、复制的Actor，负责读取磁盘数据的 Actor，负责加载保存模型的 Actor 等等。很多其他框架都把数据加载、多卡模型梯度的同步、网络、模型加载更新等分别做成一个单独的模块，而 OneFlow 的设计是所有的功能都在一张由Actor组成的静态执行图里实现了。OneFlow 这样的设计不仅简洁、优雅，还非常高效。

<div align="center">
    <img src="imgs/data_transport.png" align='center'/>
</div>

<center>
图 3 数据是如何从一个设备搬运到另一个设备上的
</center>

图3表示了没有 GPU-Direct 的况下，在 OneFlow 的 Runtime 阶段，一个设备上的计算节点如果消费了另一个设备的计算节点，数据是如何搬运过去的。

### 4. 尽可能并行

在 OneFlow 的设计中，所有的出发点都是希望可以尽可能并行，从而达到最优的分布式性能。比如考虑到分布式训练模型梯度同步时，显存到内存的传输带宽高于机器之间的网络传输带宽，OneFlow 会做两级的 scatter 和 gather 操作（本机的和各个机器之间的），用于增加 locality，提高整体性能。

又比如在异步启动深度学习训练时，Python 端用户的控制逻辑跟 OneFlow 运行时的执行图是并行执行的，同时 OneFlow 有一套互斥临界区的设计保证执行的高效性和正确性。

数据加载部分无论是从磁盘读数据还是从 Python 端喂数据，OneFlow 都能保证尽可能并行，使得计算设备不会因为要等数据而导致性能下降。

已有框架如果想要尽可能重叠数据搬运和计算，一般借助多层回调（Callback）函数，当嵌套层次过多时，会遇到所谓的 **Callback Hell** 麻烦，正确性和可读性都可能下降。但在 OneFlow 中，以上的这些并行并发特性，都是在这一套简洁的 Actor 机制下实现的，解决了令人头秃的 Callback Hell 问题。

此外，在多机的网络通信部分，OneFlow 底层的网络通信库原生支持 RDMA 的高性能通信，也有一套基于 epoll 的高效通信设计。而目前最流行的 Pytorch，多机还需要通过 RPC 来做数据同步。

## 三、OneFlow 如何做到分布式最易用
OneFlow 是目前分布式场景中支持数据并行、模型并行、流水并行等最易用的深度学习框架。用户只需要像单卡一样去搭建网络模型，并告诉 OneFlow 有哪些机器哪些卡，OneFlow 就会用最高效的方式把这些机器和设备使用起来。

这源于 OneFlow 的一套独特的设计：ConsistentView（一致性视角）。对于多机多卡，OneFlow 会 **把它抽象成一个超级大的设备** ，我们称之为逻辑上的设备，这个逻辑设备的显存是实际多个物理设备的显存之和，这个逻辑设备的算力也是实际多个物理设备的算力之和。

用户只需要在这个逻辑上的超级设备里，定义深度学习模型是如何构建的，其余的便不需要用户来操作，由 OneFlow 来完成逻辑上的设备到物理上的设备的映射。

这里先明确两个概念：“逻辑上的”和“物理上的”。“逻辑上的”表示 OneFlow 把分布式集群抽象成一个超级计算机之后的计算和数据，“物理上的”表示那些真实的部署到各个机器和设备上的计算和数据。

深度学习网络是由 Op 构成的计算图，Op 之间生产消费 Tensor 数据。在多机多卡的环境下，一个逻辑上的 Op 会对应多个真实的物理上的 Op，每个物理上的 Op 实际执行的计算都是这个逻辑 Op 计算的一部分，一个逻辑上的 Tensor 也会对应多个物理上的 Tensor，每个物理上的 Tensor 都是逻辑 Tensor 的一部分。

对于其他的框架定义的分布式训练，每张卡是一个“world”，多卡之间根据暴露出来的接口来同步模型梯度；而对于 OneFlow 而言，多机多卡也都是一个“world”，我们使用一套 Placement+SBP 的方式做全局的统筹管理。

### Placement
在 OneFlow 的计算图搭建过程中，每个计算 Op 都有一个属性叫做 Placement，表示了该逻辑上的 Op，是要部署到哪些机器哪些设备上的。对于常见的数据并行，就是所有的 Op 都部署到所有的设备上。但 OneFlow 也支持用户指定 Op 的 Placement，比如当网络过大单卡根本放不下的时候，在 OneFlow 可以让网络的前一部分在一张卡上，后一部分在另一张卡上，用一种“接力”的方式工作，实现流水并行。

图4展示了一种可能的 Placement 例子。用户定义了一个由3个 Op 组成的网络：Op_0 -> Op_1 -> Op_2。

其中 Op_0 和 Op_1 的 Placement 是 Device 0，Op_2 的 Placement 是 Device 1，这就是一个流水并行的例子，Oneflow 会自动在 Op_1 和 Op_2 之间插入需要的数据搬运的 Copy Op。

<div align="center">
    <img src="imgs/pipeline_placement.png" align='center'/>
</div>

<center>
图4 一个流水并行的Placement示例图
</center>

### SBP
SBP 是 OneFlow 独有的概念，他是三个单词的首字母组合：Split、Broadcast、PartiaSum（以 PartialSum 为例，实际上还可以是PartialMin、 PartialMax 等 reduce 操作），全称叫 SbpParallel，表示一种逻辑上的 Tensor 跟物理上的多个 Tensor 的映射关系。

其中 Split 表示物理上的 Tensor 是逻辑 Tensor 按照某一维度切分后得到的， Split 有个参数 axis，表示切分的维度，如果把多个物理上的 Tensor 按照 Split 的维度进行拼接，就能还原出逻辑 Tensor。

Broadcast 表示物理上的 Tensor 是跟逻辑上的 Tensor 完全相同的。

PartialSum 表示物理上的 Tensor 虽然跟逻辑上的 Tensor 形状一致，但是物理上的 Tensor 里的值是逻辑 Tensor 里对应位置的一部分，如果把物理上的多个 Tensor 按照对应位置相加，即可还原出逻辑上的 Tensor。

图5展示了 SBP 的简单示例。

<div align="center">
    <img src="imgs/sbp_parallel.png" align='center'/>
</div>

<center>
图5 几种 SbpParallel 的简单情形
</center>

SbpSignature 是一个 SbpParallel 的集合，在 OneFlow 的设计里是 Op 的属性，它描绘了一个逻辑上的 Op 被映射成各个设备上的多个物理上的Op以后，这些物理上的 Op 是如何看待他们输入输出Tensor在逻辑上和物理上的映射关系的。一个 Op 会有多个合法的 SbpSignature，一个最简单的合法 signature 就是输入输出都是 Broadcast，这表示了这个 Op 需要整个逻辑上的 Tensor 数据。

当用户构建的逻辑上的计算图确定以后，OneFlow 在 Compiler 生成分布式的物理上的执行图时，会考虑每个 Op 的 Placement 和该 Op 允许的合法 SbpSignature 列表，在其中选择一个传输开销最小的 SbpSignature 作为本次训练的 SbpSignature，用于指导 Compiler 生成最高效的执行图。

关于 Op 的合法 SbpSignature 的列表，我们举一个矩阵乘法（matmul）的Op的例子。

定义: `Y = matmul(A, B)` , `A`, `B`, `Y` 都是 `Tensor`，表示 `Y = AB`。那么至少存在两种合法的 SbpSignature：

* 1) Y: `Split(0)`, A: `Split(0)` , B: `Broadcast`

* 2) Y: `Split(1)`, A: `Broadcast`, B: `Split(1)`

两种合法的 signature 在两个设备上的示意图如图6所示。假设逻辑上的 MatMul 的输入输出 Tensor 的形状是：

```text
A(64, 10) × B(10, 50) -> Y(64, 50)
```

<div align="center">
    <img src="imgs/sbp_signature.png" align='center'/>
</div>

<center>
图6 MatMul的两种合法SbpSignature
</center>

且该 Op 分布在两个设备上。在第一种 SbpSignature 下，0号设备上的A是逻辑上 A 的前一半，1号设备上的 A 是逻辑 A 的后一半（按照第0维切分），而两个设备上的 B 跟逻辑上的 B 完全一致，两个设备输出的 Y 分别是逻辑上的 Y 的前一半和后一半。同样可以分析第二种 SbpSignature。

值得一提的是，当 A 是数据，B 是模型的时候，第一种 SbpSignature 就是 **数据并行** ，第二种 SbpSignature 就是 **模型并行** 。如果两个相邻的 MatMul op，前一个使用第一种 SbpSignature，后一个使用第二种 SbpSignature，整个网络就实现了 **混合并行** 。

图7是一个混合并行的示例，定义了 Y0 = MatMul_0(A0, B0) , Y1 = MatMul_1(Y0, B1) 这样一个由两个op组成的计算图，其中A0, Y0, Y1是数据Tensor，B0, B1 是模型Tensor。

<div align="center">
    <img src="imgs/mixed_parallel.png" align='center'/>
</div>

<center>
图7 混合并行示例
</center>

在图7中 MatMul_0 产出的 Y0 被 MatMul_1 消费，但是这两个 op 对同一个 Tensor 的 SBP 看待方式是不一样的，MatMul_0 认为 Y0 是 Split(axis=0) 切分，但是 MatMul_1 需要一个 Broadcast 的 Y0 输入。这时候OneFlow会自动插入一个“万能”的 Boxing Op 做必要的数据裁剪、拼接、搬运和求和等操作，使得所有的Op都可以在分布式环境下高效的拿到自己想要的数据。

另外在数据并行的时候，训练的前向模型 Tensor 的是 Broadcast，对应反向传播的梯度就是PartialSum，当 Optimizer 需要全部的梯度来更新模型时，就会触发 OneFlow 的 Boxing 机制进行高效的梯度同步工作。

### 最易用的分布式并行框架

OneFlow 的这套 Placement + SBP + Boxing 的机制，可以使得用户定义的计算图中的 Op、Tensor 以任意的方式分布在各个机器和各个设备上，无论是数据并行、模型并行还是流水并行，对于 OneFlow 而言，都只是一个特定 Placement 下的特定 SbpSignature 的组合而已，用户可以方便的配置，也可以交给 OneFlow 来做自动的处理。

另外，早在微软推出 ZeRO-2 框架之前，OneFlow 就已经支持了类似的功能，多机多卡情况下，每个模型 Tensor 都只保存在其中一个设备上，降低梯度计算中的内存占用。

## 四、总结
综上，在编译期，OneFlow 通过设计一套数学上严谨的形式系统来表示所有合法的并行模式，并支持编译器较方便地自动搜索最优并行方案。

在运行期，OneFlow 通过 Actor 系统最优地、灵活地支持并行、并发执行。OneFlow 的内核具有简洁、高效和高扩展性的优点。

基于此设计，OneFlow 使得分布式训练的性能达到极致，且分布式训练跟单卡一样简单易用。