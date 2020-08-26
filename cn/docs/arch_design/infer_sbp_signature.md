# SBP Signature 自动推导
在 [OneFlow 系统设计](../basics_topics/essentials_of_oneflow.md#sbp)一文中，我们已经从设计角度介绍过 SBP 的概念。
在本文中，将结合源码更详细地介绍 SBP 以及 OneFlow 中的 SBP Signature 自动推导机制。

## SBP 与 SBP Signature

### SBP
在 [OneFlow 如何做到分布式最易用](../basics_topics/essentials_of_oneflow.md#oneflow_2) 中介绍了 OneFlow 并行特色中“逻辑上”与 “物理上”两个概念：

> 这里先明确两个概念：“逻辑上的”和“物理上的”。“逻辑上的”表示 OneFlow 把分布式集群抽象成一个超级计算机之后的计算和数据，“物理上的”表示那些真实的部署到各个机器和设备上的计算和数据。

当我们进行分布式训练时，有多种方式将逻辑上的数据分发到物理设备上。可以是：

- 数据被切分到各个物理设备（Split），这样，每个物理设备拥有逻辑上数据的一部分，物理上的数据拼接后可以得到逻辑上的数据
- 数据被广播到各个物理设备（Broadcast），这样，每个物理设备拥有逻辑上全部的数据
- 数据以 Partial 的方式分发到各个物理设备上，这样，每个物理设备上的数据与逻辑上的数据的形状一致，但是需要对所有物理设备上的数据经过特定运算后，才可以得到逻辑上的数据，这种分发方式有 PartialSum（物理上的数据按对应位置相加得到逻辑上的数据）、PartialMax（取物理上的数据对应位置的最大值得到逻辑上的数据）等

为了表达逻辑上与物理上的数据的映射关系， OneFlow 发明了 SBP 的概念，SBP 是数据（Tensor，OneFlow 中也常称作 Blob）的属性。

以上内容的图示，可以参阅 [SBP 简单示例](https://docs.oneflow.org/basics_topics/essentials_of_oneflow.html#sbp)。

SBP 在代码中的类型名为 `SbpParallel`，它定义在 [sbp_parallel.proto](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/job/sbp_parallel.proto) 文件中：
```text
message SplitParallel {
  required int64 axis = 1;
}

message BroadcastParallel {
}

message PartialSumParallel {
}

message SbpParallel {
  oneof parallel_type {
    SplitParallel split_parallel = 1;
    BroadcastParallel broadcast_parallel = 2;
    PartialSumParallel partial_sum_parallel = 3;
  }
}
```

如以上所示，目前 SBP 属性可以是 `SplitParallel`、`BroadcastParallel`、`PartialSumParallel` 中的一种；若 SBP 属性为 `SplitParallel`，则还需要指定 `axis`，`axis` 指定了数据按照哪个维度进行切分。

### Operator 类
在 OneFlow 中，对数据的操作都抽象成为了 operator，简称 Op。Op 接受一个或多个输入 Blob，进行处理后，输出一个或多个 Blob。

OneFlow 将 OP 封装为 `Operator` 类，
在 [operator.h](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/operator/operator.h) 及其对应的 cpp 文件中。

```cpp
class Operator {
 public:
 ...
   // bn_in_op <-> lbi
  const LogicalBlobId& BnInOp2Lbi(const std::string& bn_in_op) const;
 ...
  const std::string& op_name() const { return op_conf().name(); }
  DeviceType device_type() const;
 ...
```
可以看到 `Operator` 的成员及方法，描述了一个 Op 所需要的诸如输入、输出等信息。

也有一系列 `InferXXX` 方法，它们对应了构图时的推导工作，比如本文将要介绍的 SBP Signature 推导过程，就与 `InferSbpSignatureIf` 等方法有关。

当然，`Operator` 还包括了我们即将介绍的 `SBP Signature` 成员：
```cpp
  Maybe<const SbpSignature*> sbp_signature() const;
```

### SBP Signature
Op 描述了在 **逻辑上** 如何处理数据，当分布式系统运行时，OneFlow 根据数据的 SBP 属性，将数据分发到各个物理设备，进行计算，并输出结果。

对于一个孤立的数据，其 SBP 属性可以随意设置，对于一个有输入、输出数据的 Op，我们可以随意设置它的输入、输出的 SBP 属性吗？
答案是，不可以。因为随意设置一个 Op 输入输出的 SBP 属性，可能不符合逻辑上 Op 的运算法则。

我们以逻辑上的矩阵乘法为例，假设分布式系统中2个设备，研究矩阵乘法的输入、输出的 SBP 要如何组合才合法，如何组合不合法。

逻辑上，一个性质为 `(m, k)` 的矩阵 `A` 与形状为 `(k, n)` 的矩阵 `B` 相乘得到 `Y`，Y的形状必然为 `(m, n)`
```text
  A     *     B     =     Y
(m, k)      (k, n)      (m, n)
```

依据矩阵乘法的规律，我们可以将矩阵 `A` 按第0维进行切分，切分为形状分别为 `(m0, k)`、`(m1, k)` 的两个矩阵：`A0` 和 `A1`，然后分布到2个设备上分别计算：
```text
device 0:
  A0     *     B     =     Y0
(m0, k)     (k, n)      (m0, n)

device 1:
  A1     *     B     =     Y1
(m1, k)     (k, n)      (m1, n)
```
我们容易得到物理设备上的 `A0`、`A1` 与逻辑上的 `A` 的关系，以及 `Y0`、`Y1` 与逻辑上的 `Y` 的关系：
```text
A == A0 + A1
Y == Y0 + Y1
```
可见，按照以上的方式，将逻辑上的数据分发到各个物理设备上，是能够完成运算，并且最终得到逻辑上的正确结果的。

以上较长的篇幅，若使用 SBP 来描述，会变得异常简单： 

`A` 为 Split(0)， `B` 为 Broadcast，运算结果 `Y` 为 Split(0)。

可见，对于矩阵乘法而言，其输入输出的 SBP，按以上方式组合，是合法的。对于矩阵乘法而言，合法的 SBP 组合不止这一种，比如还可以是：

> `A` 为 Broadcast， `B` 为 Split(1)，运算结果 `Y` 为 Split(1)。

以及

> `A` 为 Split(1)， `B` 为 Split(0)，运算结果 `Y` 为 PartialSum。

以上几种合法的 SBP 组合，来自 [sbp_parallel.proto](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/job/sbp_parallel.proto) 中的注释部分，更详细地说明可以查阅该文件。

虽然展示了多个合法的 SBP 组合，但是并不是任意的 SBP 组合都是合法的，比如对于矩阵乘法，若将 `A` 和 `B` 均按照 Split(0) 切分：
```
 A     == A0     +     A1
(m, k)    (m0, k)     (m1, k)

 B     == B0     +     B1
(k, n)    (k0, n)     (k1, n)
```
那么，因为矩阵乘法要求左矩阵的列数目与右矩阵的行数目相等，而 `A0`、`A1` 与 `B0`、`B1` 之间无法满足这个条件，所以它们无法分配到各个物理设备上完成矩阵乘法。我们可以说， `A` 为 Split(0)， `B` 为 Split(0) 的 SBP 组合不合法。

我们将上文出现的，对于某个 Op，其输入输出的 **合法的 SBP 属性组合**，称为 **SBP Signature**。

SBP Signature 描绘了 Op 如何看待逻辑上的输入输出与物理上的映射关系。

## SBP Signature List
有了 SBP Signature 的概念后，我们可能会提出两个问题：

- Op 的所有 Sbp Signature 是由 OneFlow 推导出来的吗？
- 如果有多个 Sbp Signature 可供选择，那么应该选择哪一个呢？

对于前一个问题，答案是否定的，因为 Op 输入输出的 SBP 属性的组合是否合法，与 Op 的运算规则有关，属于业务逻辑的范畴，OneFlow 不可能预先知晓所有已经存在的、还未发明的 Op 的运算规则。

因此，SBP Signature 的所有可能，交给了 Op 作者，OneFlow 预留了相关接口，使得 Op 的作者可以为自己的 Op 注册合法的 SBP Signature。

以矩阵乘法 [matmul_op.cpp](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/ops/matmul_op.cpp#L152) 的实现为例：

```cpp
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      // (m, k_a) * (k_b, n) where k_a == k_b
      ...
      ctx->NewBuilder()
          .Split(user_op::OpArg("a", 0), m_axis)
          .Broadcast(user_op::OpArg("b", 0))
          .Split(ctx->outputs(), 0)
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("a", 0))
          .Split(user_op::OpArg("b", 0), n_axis)
          .Split(ctx->outputs(), 1)
          .Build();
      ctx->NewBuilder()
          .Split(user_op::OpArg("a", 0), k_a_axis)
          .Split(user_op::OpArg("b", 0), k_b_axis)
          .PartialSum(ctx->outputs())
          .Build();
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("a", 0))
          .Broadcast(user_op::OpArg("b", 0))
          .PartialSum(ctx->outputs())
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("a", 0))
          .PartialSum(user_op::OpArg("b", 0))
          .PartialSum(ctx->outputs())
          .Build();
```
以上代码，就注册了：
- `a` 为 Split, `b` 为 Broadcast, 输出为 Split
- `a` 为 Broadcast, `b` 为 Split, 输出为 Split
- `a` 为 Split, `b` 为 Split, 输出为 PartialSum
- `a` 为 PartialSum, `b` 为 Broadcast, 输出为 PartialSum
- `a` 为 Broadcast, `b` 为 PartialSum, 输出为 PartialSum

5种 SBP Signature。

接着，我们来看第二个问题，既然一个 Op 可能存在多个 SBP Signature，那么在分布式训练时，是不是需要用户依据神经网络的情况而自己指定呢？

答案是：用户可以自己指定，但绝大多数情况下并没有这个必要。因为在作业函数构图阶段，OneFlow 会根据设备信息与输入数据的情况，在所有 SBP Signature 中，自动选择一个在分布式系统中，传输代价最小的 SBP Signature。

在 OneFlow 中，依据输入的 SBP 属性，选择最优的 SBP Signature，称为 SBP Signature 推导。接下来我们将结合源码，介绍 SBP Signature 推导的细节。

## SBP Signature 推导
### 流程概述

把以下绘图
```
InferOpNodeSbpSignature/InferOpOutSbpParallel 
-> InferOpSbpSignature 
-> InferSbpSignatureIf 
-> InferSbpSignature
```

### 代码解读