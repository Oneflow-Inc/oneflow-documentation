# SBP Signature 自动推导
在 [OneFlow 系统设计](../basics_topics/essentials_of_oneflow.md#sbp)一文中，我们已经从设计角度介绍过 SBP 的概念。
在本文中，将结合源码更详细地介绍 SBP 以及 OneFlow 中的 SBP Signature 自动推导机制，包括：

- 数据的 SBP 属性
- Op 的 SBP Signature 属性
- OneFlow 自动推导 SBP Signature 的流程
- SBP Signature 的代价模型

## 基础概念介绍

### SBP
在 [OneFlow 如何做到分布式最易用](../basics_topics/essentials_of_oneflow.md#oneflow_2) 中介绍了 OneFlow 并行特色中“逻辑视角”与 “物理视角”两个概念：

OneFlow 的逻辑视角，意味着 OneFlow 可以将分布式集群中各物理设备上的数据和算力，抽象成一个逻辑上的超级计算机的数据和算力；而 OneFlow 的物理视角，可以关注到那些真实的部署到各个设备上的数据和算力。

当我们进行分布式训练时，有多种方式将逻辑视角的数据分发到物理设备上。可以是：

- 数据被切分到各个物理设备（Split），这样，每个物理设备拥有逻辑上数据的一部分，物理上的数据拼接后可以得到逻辑上的数据
- 数据被广播到各个物理设备（Broadcast），这样，每个物理设备拥有逻辑上全部的数据
- 数据以 Partial 的方式分发到各个物理设备上，这样，每个物理设备上的数据与逻辑上的数据的形状一致，但是需要对所有物理设备上的数据经过特定运算后，才可以得到逻辑上的数据，这种分发方式有 PartialSum（物理上的数据按对应位置相加得到逻辑上的数据）、PartialMax（取物理上的数据对应位置的最大值得到逻辑上的数据）等

为了表达逻辑视角与物理视角上的数据映射关系， OneFlow 发明了 SBP 的概念，SBP 是数据（Tensor，OneFlow 中也常称作 Blob）的属性。

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

OneFlow 将 Op 封装为 `Operator` 类，
在 [operator.h](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/operator/operator.h) 及其对应的 cpp 文件中。

```cpp
class Operator {
 ...
};
```
可以看到 `Operator` 的成员及方法，描述了一个 Op 所需要的诸如输入、输出等信息。

也有一系列 `InferXXX` 方法，它们对应了构图时的推导工作，比如本文将要介绍的 SBP Signature 推导过程，就需要调用 `InferSbpSignatureIf` 方法推导最优的SBP Signature。

当然，`Operator` 还包括了我们即将介绍的 `SBP Signature` 成员，它对应了最终推导的结果：
```cpp
  Maybe<const SbpSignature*> sbp_signature() const;
```

### SBP Signature
Op 描述了在 **逻辑视角** 上如何处理数据，当分布式系统运行时，OneFlow 根据数据的 SBP 属性，将数据分发到各个物理设备，进行计算，并输出结果。

对于一个孤立的数据，其 SBP 属性(`SbpParallel`)可以随意设置，对于一个有输入、输出数据的 Op，我们可以随意设置它的输入、输出的 SBP 属性吗？

不可以。因为随意设置一个 Op 输入输出的 SBP 属性，可能不符合逻辑上 Op 的运算法则。

让我们以矩阵乘法为例，讨论这个问题。看看在有2个设备的分布式系统中，矩阵乘法的输入、输出的 SBP 要如何组合才合法，如何组合不合法。

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
 A     ==     A0     +     A1
(m, k)     (m0, k)        (m1, k)

 Y     ==     Y0     +     y1
(m, n)     (m0, n)        (m1, n)
```
以上的“+”表示拼接，下同。

可见，按照以上的方式，将逻辑上的数据分发到各个物理设备上，是能够完成运算，并且最终得到逻辑上的正确结果的。

以上较长的篇幅，若使用 SBP 来描述，会变得异常简单： 

> `A` 为 Split(0)， `B` 为 Broadcast，运算结果 `Y` 为 Split(0)。

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

我们将上文出现的，对于某个 Op，其输入输出的 **一个特定的、合法的 SBP 属性组合**，称为这个 Op 的一个 **SBP Signature**。

SBP Signature 描绘了 Op 如何看待逻辑视角的输入输出与物理视角的映射关系。

## 选择最优的 SBP Signature
有了 SBP Signature 的概念后，我们可能会提出两个问题：

- Op 的所有 Sbp Signature 是由 OneFlow 推导出来的吗？
- 如果有多个 Sbp Signature 可供选择，那么应该选择哪一个呢？

对于前一个问题，答案是否定的，因为 Op 输入输出的 SBP 属性的组合是否合法，与 Op 的运算规则有关，属于业务逻辑的范畴，OneFlow 不可能预先知晓所有已经存在的、还未发明的 Op 的运算规则。

因此，OneFlow 将罗列所有可能的 SBP Signature 的工作，交给了 Op 作者，OneFlow 预留了相关接口，使得 Op 的作者可以为自己的 Op 注册合法的 SBP Signature。

以矩阵乘法 [matmul_op.cpp](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/ops/matmul_op.cpp#L152) 为例：

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
      ...
```
以上代码，就注册了：

- `a` 为 Split, `b` 为 Broadcast, 输出为 Split
- `a` 为 Broadcast, `b` 为 Split, 输出为 Split
- `a` 为 Split, `b` 为 Split, 输出为 PartialSum
- `a` 为 PartialSum, `b` 为 Broadcast, 输出为 PartialSum
- `a` 为 Broadcast, `b` 为 PartialSum, 输出为 PartialSum

5种 SBP Signature。OneFlow 中准备了数据结构 `SbpSignatureList` 用于存放多个 SBP Signature：

```text
message SbpSignatureList {
  repeated SbpSignature sbp_signature = 1;
}
```

接着，我们来看第二个问题，既然一个 Op 可能存在多个 SBP Signature，那么在分布式训练时，是不是需要用户依据神经网络的情况而自己指定呢？

答案是：用户可以自己指定，但绝大多数情况下并没有这个必要。因为在构图阶段，OneFlow 会根据设备信息与输入数据的情况，在所有 SBP Signature 中，自动选择一个最优的 SBP Signature。

在 OneFlow 中，依据输入的 SBP 属性，选择最优的 SBP Signature，称为 **SBP Signature 推导** 。接下来我们将结合源码，介绍 SBP Signature 推导的细节。

## SBP Signature 推导
所谓的 SBP Signature 推导，就是在多个合法 SBP Signature 中，为 Op 选择一个最优的。目前，OneFlow 对于“最优”的默认标准是传输代价最小。

### 流程概述

在 Lazy 模式下，函数 `::InferOpSbpSignature` 是SBP Signature 推导的入口，在 [job_build_and_infer_ctx.cpp](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/job/job_build_and_infer_ctx.cpp) 的 `JobBuildAndInferCtx::InferOpOutSbpParallel` 以及 [op_graph.cpp](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/graph/op_graph.cpp) 的 `OpGraph::InferOpNodeSbpSignature` 中都会调用它。

前者发生在 OneFlow 构建用户 Python 端的网络时，后者发生在 OneFlow 对用户的网络进行进一步的编译优化时。以前者为例，调用关系为：
```
JobBuildAndInferCtx::InferOpOutSbpParallel
  -> ::InferOpSbpSignature
    -> Operator::InferSbpSignatureIf 
      -> Operator::InferSbpSignature
```

各函数（方法）的接口及主要工作罗列如下，需要提前说明：下文出现的名如 `XX4YY` 的函数，均为对象转化方法(Get XX for YY)，比如 `ConstBlobDesc4Ibn` 就是根据 Ibn (input blob name) 得到 const blob description。

* JobBuildAndInferCtx::InferOpOutSbpParallel
```cpp
Maybe<void> 
JobBuildAndInferCtx::InferOpOutSbpParallel(Operator* op,
                    const SbpSignature& sbp_sig_conf,
                    const ParallelDesc& parallel_desc);
```
在 `JobBuildAndInferCtx::InferOpOutSbpParallel` 接受 Op、用户指定的 SBP Signature（如果有的话）、并行配置信息作为参数，并且在内部整理 Op 的输入的 SBP ，将这些信息一起传递给下一层的 `InferOpSbpSignature`。

* InferOpSbpSignature
```cpp
Maybe<void> InferOpSbpSignature(
    Operator* op, const SbpSignature& sbp_sig_conf, 
    const ParallelDesc& parallel_desc,
    const HashMap<std::string, SbpInferHint>& ibn2sbp_infer_hint,
    std::function<Maybe<const OptInt64*>(const std::string&)> BatchAxis4BnInOp);
```
在 `InferOpSbpSignature` 中主要做准备工作：它设计了一个 cost model，为各个可选的 SBP Signature 进行打分，分数最低的 SBP Signature 意味着传输代价最小。这个函数中设计的 cost model 将会在下一层 `Operator::InferSbpSignatureIf` 中使用。

* Operator::InferSbpSignatureIf
```cpp
Maybe<void> Operator::InferSbpSignatureIf(
    const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc);
```
在 `Operator::InferSbpSignatureIf` 中将根据是单机还是分布式情况进行不同处理：
如果是单机情况，则输入输出均采用 Split(0) 即可；如果是分布式情况，则调用下一层的 `Operator::InferSbpSignature`，根据上一层设计的 cost model，挑选出代价最小的 SBP Signature。

* Operator::InferSbpSignature
```cpp
Maybe<void> Operator::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc);
```
在 `Operator::InferSbpSignature` 中，将获取 Op 作者注册的所有SBP Signature，然后根据 cost model 对它们进行打分并排序，选择代价最小者。


### SBP Signature 的代价模型
SBP 的推导，发生在计算图构图过程中，目前采用的是贪心算法，基本流程是：

- 通过计算图的拓扑序遍历每个 Op
- 对于计算图的 source 节点（无输入的节点），如果是模型（variable），SBP 属性默认设置为是 Broadcast；如果是数据，SBP 属性默认为 Split
- 除了source节点外的其他所有结点，在拓扑序遍历过程中，则会对 Op 的 SBP Signature 进行推导，寻找一个 cost model 最小的的 SBP Signature

结合以上内容及在流程概述中的介绍，我们知道 SBP Signature 推导的关键在 cost model 如何评价 SBP Signature。我们将结合代码重点介绍 OneFlow 如何计算 SBP Signature 的代价。

在 [InferOpSbpSignature](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/operator/operator.cpp) 中，有对应的 cost model，用于计算 SBP Signature 的代价，采用的具体算法如下。

首先， OneFlow 准备了三个函数，分别从三个角度，根据输入以及待选的 SBP Signature 中的 SBP 属性进行打分：
```cpp
  auto OrderValue4HasBatchAxis = [&](const std::string& bn,
                                     const SbpParallel& sbp_parallel) -> int32_t {
    const auto& batch_axis = *BatchAxis4BnInOp(bn);
    return -1
           * (batch_axis.has_value() && sbp_parallel.has_split_parallel()
              && sbp_parallel.split_parallel().axis() == batch_axis.value());
  };
  auto OrderValue4HasNoBatchAxis = [&](const std::string& ibn,
                                       const SbpParallel& sbp_parallel) -> int32_t {
    const auto& batch_axis = *BatchAxis4BnInOp(ibn);
    return -2
           * (batch_axis.has_value() == false
              && SbpInferHint4Ibn(ibn)->sbp_parallel().has_split_parallel() == false
              && sbp_parallel.has_split_parallel() == false);
  };
  auto OrderValue4SbpHint = [&](const std::string& ibn,
                                const SbpParallel& sbp_parallel) -> int32_t {
    return -3 * (SbpInferHint4Ibn(ibn)->sbp_parallel() == sbp_parallel);
  };
```
因为三个函数的返回值都是 `数字*bool` 的形式，所以返回值为 -3，-2，-1，0中的某个。

比如，若以下表达式为 `true`：
```cpp
(SbpInferHint4Ibn(ibn)->sbp_parallel() == sbp_parallel)
```
则意味着当前输入的 SBP 属性，与待选择的 SBP Signature 中的对应位置的 SBP 属性是一致的，那么传输代价最小，分数为-3。

以上三个函数，只是对于单个 input blob 进行代价评估，之后，使用了一个 `CalcOrderValue4SbpSig` 函数，遍历 Op 的所有输入，综合以上多个角度的结果，得到代价的综合分数：

```cpp
CalcOrderValue4SbpSig = [&](const SbpSignature& sbp_signature) -> int32_t {
  int32_t order_value = 0;
  for (const auto& ibn : op->input_bns()) {
    // 待计算代价的 SBP Signature 中，对应的 SBP 属性
    const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(ibn);

    // 根据 input blob 和 SBP Signature 中的 SBP，进行打分
    order_value += OrderValue4HasBatchAxis(ibn, sbp_parallel_it->second);
    order_value += OrderValue4HasNoBatchAxis(ibn, sbp_parallel_it->second);
    order_value += OrderValue4SbpHint(ibn, sbp_parallel_it->second);
  }
  for (const auto& obn : op->output_bns()) {
    const auto& sbp_parallel_it = sbp_signature.bn_in_op2sbp_parallel().find(obn);
    order_value += OrderValue4HasBatchAxis(obn, sbp_parallel_it->second);
  }
  return order_value;
};
```

以上准备了 cost model 的评价标准，真正的评价时机，发生在 `Operator::InferSbpSignatureIf` 中：
```cpp
Operator::InferSbpSignatureIf(...) {
  if (parallel_desc.parallel_num() == 1) {
    auto* bn2sbp = mut_sbp_signature()->mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : input_bns()) { (*bn2sbp)[ibn].mutable_split_parallel()->set_axis(0); }
    for (const auto& obn : output_bns()) { (*bn2sbp)[obn].mutable_split_parallel()->set_axis(0); }
  } else if (parallel_desc.parallel_num() > 1) {
    return InferSbpSignature(mut_sbp_signature(), 
              sbp_sig_conf, 
              CalcOrderValue4SbpSig,
              SbpInferHint4Ibn, 
              parallel_desc);
  }
}
```
其逻辑非常简单:

- 当 `parallel_desc.parallel_num() == 1` 时，说明是单机单卡情况，此时 Split、Broadcast、PartialSum 是等价的，SBP 属性设置为哪种都正确，因此不妨将所有的输入、输出的 SBP 属性设置为 Split(0) 即可
- 当并行数目大于1时，则调用 `Operator::InferSbpSignature`，依据 cost model，选择代价最小的 SBP Signature

在 `Operator::InferSbpSignature` 中：
```cpp
Operator::InferSbpSignature(...){
  // get op sbp signatures
  ...
  SbpSignatureList sbp_sig_list;
  GetSbpSignaturesIf(LogicalBlobDesc4Ibn, parallel_desc, &sbp_sig_list);
  
  ...

  // sort sbp signatures by copy cost, then return the one with least cost
  std::vector<const SbpSignature*> sorted_sbp_signatures;
  SortSbpSignatureListByCopyCost(filtered_sbp_sigs_by_conf,
                    input_bns(), 
                    SbpInferHint4Ibn,
                    CalcOrderValue4SbpSig, 
                    &sorted_sbp_signatures);
  *sbp_signature = *sorted_sbp_signatures.at(0);
  return Maybe<void>::Ok();
}
```
先通过 `GetSbpSignaturesIf` 获取 Op 作者设置的所有 SBP Signature，然后调用 `SortSbpSignatureListByCopyCost`，在这个函数内部，将调用 `CalcOrderValue4SbpSig` 对所有的 SBP Signature 进行打分，并排序，排序后的结果，按代价升序放置在 `sorted_sbp_signatures` 中。

因此，最终选择代价最小的 `sorted_sbp_signatures.at(0)` 作为 Op 的 SBP Signature。

值得一提的是，默认的 cost model 虽然简单，但经过实践证明已经有非常不错的效果。此外，如果想使用自定义的标准选择 SBP Signature，只需要重写虚函数 `Operator::InferSbpSignature` 即可。

最后，更值得一提的是，除了本文介绍的 SBP Signature 推导方法外，OneFlow 团队正在研发一种寻求全局最优解的自动并行方法，敬请期待。

