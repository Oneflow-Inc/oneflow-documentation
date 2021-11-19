# 集群的一致性视角

OneFlow 提出了 **一致性视角（consistent view）** 的概念，用于简化分布式训练。简单而言，在 OneFlow 的一致性视角下，集群被抽象为一台“超级计算设备”。

用户不用关系集群中计算、通信的细节，只用关心逻辑上的数据与计算，依然像单机单卡那样思考、编程，就能进行分布式训练。

![consistent view](./imgs/consistent-view.png)

OneFlow 的一致性视角，依赖几个重要概念：Placement、SBP 与 SBP Signature。

## Placement
OneFlow 一致性视角下的 Tensor 有 `placement` 属性，通过 `placement` 属性可以指定该 Tensor 存放在哪个物理设备上。

OneFlow 会自动为集群中的计算设备编号。比如，如果集群中有4台主机，每台主机上有8张显卡，那么4台主机分别对应了 ID：0、1、2、3；每台主机上的显卡分别对应了编号0到7。如果想将 Tensor 放置在第0台机器的前4张显卡上，只需要配置：`placement("cuda", {0: [0, 1, 2, 3]})`。

`placement` 使得 OneFlow 很容易支持流水并行，我们将在本专题的其它文章中看到与 `placement` 有关的实际例子。

## SBP

SBP 是 OneFlow 发明的概念，描述了“超级计算设备”一致性视角下的数据与集群中真实的物理设备上的数据的映射关系，它由 `split`, `broadcast`, `partial` 的首字母组合而成。

详细而言：

- `split` 表示物理设备上的 Tensor，是将一致性视角的 Tensor 切分得到的。切分时，需要指定切分的维度。物理设备上的 Tensor ，经过拼接，可以还原得到一致性视角的 Tensor 。
- `broadcast` 表示一致性视角下的 Tensor，会复制并广播到所有的物理设备上。
- `partial` 表示一致性视角下的 Tensor 与物理设备上的 Tensor 的 **形状相同**，但是物理设备上的值，只是一致性视角下 Tensor 的 **一部分**。以 `partial sum` 为例，如果我们将集群中所有设备的张量按位置相加，那么就可以还原得到一致性视角的 Tensor。除了 `sum` 外，`min`、`max` 等操作也适用于 `partial`。

下图中分别展示了 SBP 的情况，分别是 `split(0)`、`split(1)`、`broadcast`、`partial sum`。

![SBP Example](./imgs/sbp-example.png)

在创建 Consistent Tensor 时，可以指定 Tensor 的 SBP，实际的代码例子将在下一篇文章 [Consistent Tensor](./03_consistent_tensor.md) 中看到。

## SPB Signature

SBP 描述了一致性视角下的数据与物理设备上的数据的映射关系，当进行分布式训练时，OneFlow 根据数据的 SBP 属性，将数据分发到各个物理设备，进行计算，并输出结果。

对于一个孤立的 Tensor，我们可以随意设置它的 SBP 属性。
但是，对于一个有输入、输出数据的算子，我们却不可以随意设置它的输入、输出的 SBP 属性。这是因为随意设置一个算子输入输出的 SBP 属性，可能不符合一致性视角下算子的运算法则。

让我们以矩阵乘法为例讨论这个问题。看看在有2个设备的分布式系统中，矩阵乘法的输入、输出的 SBP 要如何组合才合法，如何组合不合法。

假设一致性视角下要，一个形状为 $(m, k)$ 的矩阵 $A$ 与形状为 $(k, n)$ 的矩阵 $B$ 相乘得到 $Y$，$Y$ 的形状必然为 $(m, n)$。

依据矩阵乘法的规律，我们可以将矩阵 $A$ 按第0维进行切分，切分为形状分别为 $(m_0, k)$、$(m_1, k)$ 的两个矩阵：$A_0$ 和 $A_1$，然后在2个设备上分别计算：

设备一：

$$
\begin{matrix}
A_0     \times     B     =     Y_0
\\
(m_0, k)     (k, n)      (m_0, n)
\end{matrix}
$$

设备二：

$$
\begin{matrix}
A_1     \times     B     =     Y_1
\\
(m_1, k)     (k, n)      (m_1, n)
\end{matrix}
$$

我们容易得到物理设备上的 $A_0$、$A_1$ 与一致性视角 $A$ 的关系，以及 $Y_0$、$Y_1$ 与一致性视角数据 $Y$ 的关系：

$$
\begin{matrix}
A &= concat&(A_0 ,& A_1) \\
(m,k) &  & (m_0, k) & (m_1, k)
\end{matrix}
$$

$$
\begin{matrix}
Y &= concat&(Y_0 ,& Y_1) \\
(m,n) &  & (m_0, n) & (m_1, n)
\end{matrix}
$$

> 注意：以上的 `concat` 表示拼接操作。

可见，按照以上的方式，将一致性视角的数据分发到各个物理设备上，是能够完成运算，并且最终得到一致性视角上的正确结果的。以上较长的篇幅，若 **使用 SBP 来描述，会变得异常简单** ：

$A$ 为 `split(0)`， $B$ 为 `broadcast`，运算结果 $Y$ 为 `split(0)`。

可见，对于矩阵乘法而言，其输入输出的 SBP，按以上方式组合，是合法的。对于矩阵乘法而言，**合法的 SBP 组合不止一种**，比如还可以是：

$A$ 为 `broadcast`， $B$ 为 `split(1)`，运算结果 $Y$ 为 `split(1)`。

或者：

$A$ 为 `split(1)`， $B$ 为 `split(0)`，运算结果 $Y$ 为 `partial sum`。

虽然展示了多个合法的 SBP 组合，但是并不是任意的 SBP 组合都是合法的，比如对于矩阵乘法，如果 $A$、$B$ 均为 `split(0)`，那么：

$$
\begin{matrix}
A &= concat&(A_0 ,& A_1) \\
(m,k) &  & (m_0, k) & (m_1, k)
\end{matrix}
$$

$$
\begin{matrix}
B &= concat&(B_0 ,& B_1) \\
(k,n) &  & (k_0, n) & (k_1, n)
\end{matrix}
$$

那么在物理设备上，因为 $A_0$ 与 $B_0$ 的形状，并不满足矩阵乘法的要求，也就无法在物理设备上完成矩阵乘法。我们可以说， $A$ 为 `split(0)`， $B$ 为 `split(0)` 的 SBP 组合是不合法的。

我们将上文出现的，对于某个算子，其输入输出的一个 **特定的、合法的 SBP 组合**，称为这个算子的一个 **SBP Signature**。

OneFlow 中的算子，都会由算子作者根据算子的运算法则，预设好该算子所有可能的 SBP Signature，用户只需要设置数据的 `placement` 和 `SBP` 属性，在运行时，OneFlow 会自动选择最优的 SBP Signature，这个选择过程对用户而言是透明的。

## 总结
`placement` 与 `SBP`、`SBP Signature` 是 OneFlow 分布式一致性视角的重要保证，OneFlow 的一致性视角使得 OneFlow 的分布式训练与单机单卡一样简单。

在下一篇 [Consistent Tensor](./03_consistent_tensor) 中，我们将看到一致性视角的编程例子。
