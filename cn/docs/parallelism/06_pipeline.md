# 流水并行训练

在 [常见的分布式并行策略](./01_introduction.md) 一文中介绍了流水并行的特点。

在 OneFlow 的 [一致性视角](./03_consistent_tensor.md) 下，通过简单的设置 Tensor 的 `placement` 属性，就可以实现流水并行。

可以用以下命令快速体验 OneFlow 的流水并行：

```shell

```