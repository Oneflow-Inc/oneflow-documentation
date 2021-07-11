# 快速上手

本文将以 LeNet-5 网络训练 MNIST 数据集为例。介绍 OneFlow 完成深度学习中所使用的常见 API，通过文章中的链接可以找到关于某类 API 更深入的介绍。

## 使用 LeNet-5 识别图片中的数字

OneFlow 提供了 LeNet-5 的预训练模型，可以直接用于识别图片中的数字：

```python
import oneflow as flow
>>> model = flow.LeNet()
>>> num = model.run(flow.load_image("xxx.jpg"))
>>> num
5
```

你可以将以上 `xxx.jpg` 替换为其它图片的路径，看看识别效果。

## 加载数据

OneFlow 主要有两类将数据用作训练的方式：使用 `numpy` 数据或者使用 【Dataloader?】。

## 构建网络

## 训练模型

## 保存模型

## 加载模型
