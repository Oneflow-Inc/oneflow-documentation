# 概述

## 分布式深度学习框架

本标准所述分布式深度学习框架如图1： 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201102203337228.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

深度学习框架从底层，实现，到应用，大致分为六个组件。



a) 分布式深度学习框架底层依赖于基础硬件库，包括系统资源（内存，线程资源管理），计算加速硬件（如CPU，GPU），网络通信库（如常用的 epoll，nccl）以及文件系统。

b) 支撑库包含了用于支持框架运行的组件，包括平台及通用套件，底层基础算子，张量计算库，可供选择的张量编译器（如XLA，TensorRT），为框架整体提供基础支撑。

c）编译时主要是框架在编译期间根据用户定义的网络结构构建计算图，进一步优化成执行计划，为程序运行做好准备。包括算子描述，自动求导，内存分配，自动并行机制，图优化，自动放置，代价计算，数据路由等操作。

d)  运行时包含了框架运行程序所需的组件，包含有执行体，状态机，控制平面，消息中枢等组件。

e）接口层主要是提供程序调用接口，包括静态图执行，动态图执行，模型格式转换以及上层提供的 Python 调用接口，用户不需要接触底层概念即可轻松使用。

f）模型库涵盖了多个人工智能领域的模型，供企业，开发者们使用。计算机视觉领域中包括常见的分类网络，目标检测，人脸识别模型；语音识别有基于时序的 RNN，LSTM 模型；自然语言处理领域有最常见的 BERT，TransFormer 模型；以及推荐系统常用的 Wide&Deep 模型。



# 接口说明

## 用户接口说明

深度学习框架需要提供包括 Python 在内，但不限于 Python 的接口供用户调用。包括以下几大类：

- 提供了配置分布式深度学习框架运行环境的接口，包括配置计算设备数量，线程池大小，网络线程数量等。

- 提供了常见的优化方法及优化目标接口，包括但不限于 SGD，Adam，RMSProp等优化器

- 提供了常见的基础数学算子，能够支持矩阵运算，三角函数，指数运算等数学操作
- 提供了神经网络算子及计算层库，能够支持卷积，池化等张量操作

- 提供了数据预处理操作，包括但不限于随机翻转，随机裁剪，随机缩放等

- 提供了参数正则化接口
- 提供系统配置接口，能够支持系统日志记录，选择数据放置，管理训练程序等功能
- 此外还需要提供方便用户进行程序验证，模型搭建的其他方法。包括但不限于数据集下载，命名空间，并行策略等

## 应用部署接口说明

### Serving

深度学习框架应兼容多种场景下，多种硬件架构下，多种操作系统下的应用部署，包括但不限于：

- 支持 POSIX 标准 (ISO/IEC9945) 操作系统部署
- 支持基于 http，grpc 等通信协议进行服务通信
- 支持基于 kubernetes 的一站式AI开发中枢, 提供框架/模型训练的快速部署
- 支持第三方模型格式 (如 ONNX) 转换，实现跨平台，高性能推理部署
- 用于部署的文件格式应考虑容错性，格式校验，安全性。需包含模型的训练参数，计算图拓扑关系以及其他拓展需求

### 上层应用

针对上层应用，深度学习框架应提供常见的研究价值大，商业价值大的预训练模型，包括不限于：

| 模型应用           | 数据集             | 目标                                       | 参考模型                       |
| ------------------ | ------------------ | ------------------------------------------ | ------------------------------ |
| 图像分类           | ImageNet           | 74.9分类准确率                             | Resnet-50 v1.5                 |
| 目标检测（重量级） | COCO 2017          | 0.377 Box min AP 0.339, Mask min AP        | Mask R-CNN                     |
| 目标检测（轻量级） | COCO 2017          | 21.2% mAP                                  | SSD (Resnet-34 backbone)       |
| 翻译（非递归）     | WMT English-German | 25.0 BLEU                                  | Transformer                    |
| 人脸识别           | MS1M-ArcFace       | 99.80% LFW，92.74% CFP-FP，97.76% AgeDB-30 | ArcFace(Resnet-50 backbone)    |
| 推荐系统           | MovieLens          | 0.51 HR@10                                 | Neural Collaborative Filtering |

# 缩略词补充

| 缩略词      | 解释                                                         |
| ----------- | ------------------------------------------------------------ |
| CPU         | 中央处理器（central processing unit）                        |
| GPU         | 图形处理器（Graphics Processing Unit）                       |
| epoll       | Linux 内核的 IO 多路复用实现                                 |
| NCCL        | 英伟达自研的多卡通信框架（Nvidia Collective multi-GPU Communication Library） |
| XLA         | 一种深度学习编译器（Accelerated Linear Algebra）             |
| TensorRT    | 英伟达自研的高性能深度学习推理框架                           |
| RNN         | 循环神经网络（Recurrent Neural Network）                     |
| LSTM        | 长短期记忆人工神经网络（Long Short Term Memory networks）    |
| TransFormer | 谷歌于2017年提出的一款文本模型                               |
| Wide&Deep   | 谷歌于2016年提出的一款推荐框架                               |
| SGD         | 随机梯度下降（Stochastic gradient descent）                  |
| Adam        | 自适应矩估计优化器                                           |
| Rmsprop     | 均方根优化器                                                 |
| http        | 超文本传输协议（HyperText Transfer Protocol）                |
| https       | 超文本传输安全协议（Hyper Text Transfer Protocol over SecureSocket Layer） |
| RPC         | 远程过程调用（Remote Procedure Call）                        |
| ONNX        | 开放神经网络交换（Open Neural Network Exchange）             |
|             |                                                              |
|             |                                                              |

