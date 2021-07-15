## 简介 Introduction

### 图像分类与CNN

 **图像分类** 是指将图像信息中所反映的不同特征，把不同类别的目标区分开来的图像处理方法，是计算机视觉中其他任务，比如目标检测、语义分割、人脸识别等高层视觉任务的基础。

ImageNet 大规模视觉识别挑战赛（ILSVRC），常称为 ImageNet 竞赛，包括图像分类、物体定位，以及物体检测等任务，是推动计算机视觉领域发展最重要的比赛之一。

在2012年的 ImageNet 竞赛中，深度卷积网络 AlexNet 横空出世。以超出第二名10%以上的top-5准确率，勇夺 ImageNet2012 比赛的冠军。从此，以 **CNN（卷积神经网络）** 为代表的深度学习方法开始在计算机视觉领域的应用开始大放异彩，更多的更深的CNN网络被提出，比如 ImageNet2014 比赛的冠军 VGGNet, ImageNet2015 比赛的冠军 ResNet。



### ResNet

[ResNet](https://arxiv.org/abs/1512.03385) 是2015年ImageNet竞赛的冠军。目前，ResNet 相对对于传统的机器学习分类算法而言，效果已经相当的出色，之后大量的检测，分割，识别等任务也都在 ResNet 基础上完成。

[OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark) 仓库中，提供 ResNet50 v1.5 的 OneFlow 实现。我们在 ImageNet-2012 数据集上训练90轮后，验证集上的准确率能够达到：77.318%(top1)，93.622%(top5)。

更详细的网络参数对齐工作，见 [OneFlow-Benchmark的cnns](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns) 部分

![resnet50_validation_acuracy](imgs/resnet50_validation_acuracy.png)



**关于 ResNet50 v1.5 的说明：**

> ResNet50 v1.5 是原始 [ResNet50 v1](https://arxiv.org/abs/1512.03385) 的一个改进版本，相对于原始的模型，精度稍有提升 (~0.5% top1)，详细说明参见[这里](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5) 。
>



准备好亲自动手，复现上面的结果了吗？



下面，本文就以上面的 ResNet50 为例，一步步展现如何使用 OneFlow 进行 ResNet50 网络的训练和预测。

主要内容包括：

- 准备工作
  - 项目安装和准备工作

- 快速开始
  - 预测/推理
  - 训练和验证
  - 评估
- 更详细的说明
  - 分布式训练
  - 混合精度训练与预测
- 进阶
  - 参数对齐
  - 数据集制作(ImageNet2012)
  - OneFlow 模型转 ONNX 模型



## 准备工作 Requirements

别担心，使用 OneFlow 非常容易，只要准备好下面三步，即可开始 OneFlow 的图像识别之旅。

- 安装 OneFlow，安装方式参考 [OneFlow项目主页](https://github.com/Oneflow-Inc/oneflow)

- 克隆/下载 [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark) 仓库。

  `git clone git@github.com:Oneflow-Inc/OneFlow-Benchmark.git`

  `cd  OneFlow-Benchmark/Classification/cnns`

- 准备数据集（可选）

  - 直接使用 synthetic 虚拟合成数据集
  - 下载我们制作的 Imagenet(2012) [迷你数据集](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/dataset/imagenet/mini-imagenet.zip) 解压放入data目录
  - 或者：制作完整 OFRecord 格式的 ImageNet 数据集（见下文进阶部分）

我们提供了通用脚本：`train.sh` 和 `inference.sh`，它们适用于此仓库下所有cnn网络模型的训练、验证、推理。您可以通过设置参数使用不同的模型、数据集来训练/推理。

 **关于模型的说明：**

> 默认情况下，我们使用resnet50，您也可以通过改动脚本中的--model参数指定其他模型，如：`--model="resnet50"`，`--model="vgg"` 等。

**关于数据集的说明：**


> 1）为了使读者快速上手，我们提供了 synthetic 虚拟合成数据，“合成数据”是指不通过磁盘加载数据，而是直接在内存中生成一些随机数据，作为神经网络的数据输入源。
>
> 2）同时，我们提供了一个小的迷你示例数据集。直接下载解压至 cnn 项目的 data 目录，即可快速开始训练。读者可以在熟悉了流程后，参考数据集制作部分，制作完整的 Imagenet2012 数据集。
>
> 3）使用 OFRcord 格式的数据集可以提高数据加载效率（但这非必须，参考[数据输入](../basics_topics/data_input.md)，OneFlow 支持直接加载 numpy 数据）。



## 快速开始 Quick Start

那么接下来，立马开始 OneFlow 的图像识别之旅吧！

首先，切换到目录：

```
cd OneFlow-Benchmark/Classification/cnns
```

### 预训练模型

#### resnet50

[resnet50_v1.5_model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/resnet_v15_of_best_model_val_top1_77318.tgz ) (validation accuracy: 77.318% top1，93.622% top5 )

### 预测/推理

下载好预训练模型后，解压后放入当前目录，然后执行：

```
sh inference.sh
```

此脚本将调用模型对这张金鱼图片进行分类：

<div align="center">
    <img src="imgs/fish.jpg" align='center'/>
</div>

若输出下面的内容，则表示预测成功：

```
data/fish.jpg
0.87059885 goldfish, Carassius auratus
```

可见，模型判断这张图片有87.05%的概率是金鱼 goldfish。

### 训练和验证（Train & Validation）

- 训练同样很简单，只需执行：

  ```
  sh train.sh
  ```

  即可开始模型的训练，您将看到如下输出：

  ```
  Loading synthetic data.
  Loading synthetic data.
  Saving model to ./output/snapshots/model_save-20200723124215/snapshot_initial_model.
  Init model on demand.
  train: epoch 0, iter 10, loss: 7.197278, top_1: 0.000000, top_k: 0.000000, samples/s: 61.569
  train: epoch 0, iter 20, loss: 6.177684, top_1: 0.000000, top_k: 0.000000, samples/s: 122.555
  Saving model to ./output/snapshots/model_save-20200723124215/snapshot_epoch_0.
  train: epoch 0, iter 30, loss: 3.988656, top_1: 0.525000, top_k: 0.812500, samples/s: 120.337
  train: epoch 1, iter 10, loss: 1.185733, top_1: 1.000000, top_k: 1.000000, samples/s: 80.705
  train: epoch 1, iter 20, loss: 1.042017, top_1: 1.000000, top_k: 1.000000, samples/s: 118.478
  Saving model to ./output/snapshots/model_save-20200723124215/snapshot_epoch_1.
  ...
  ```

  >  为了方便运行演示，我们默认使用synthetic虚拟合成数据集，使您可以快速看到模型运行的效果

  同样，你也可以使用[迷你示例数据集](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/dataset/imagenet/mini-imagenet.zip)，下载解压后放入 cnn 项目的 data 目录即可，然后修改训练脚本如下：

  ```
  rm -rf core.*
  rm -rf ./output/snapshots/*

  DATA_ROOT=data/imagenet/ofrecord

  python3 of_cnn_train_val.py \
      --train_data_dir=$DATA_ROOT/train \
      --num_examples=50 \
      --train_data_part_num=1 \
      --val_data_dir=$DATA_ROOT/validation \
      --num_val_examples=50 \
      --val_data_part_num=1 \
      --num_nodes=1 \
      --gpu_num_per_node=1 \
      --model_update="momentum" \
      --learning_rate=0.001 \
      --loss_print_every_n_iter=1 \
      --batch_size_per_device=16 \
      --val_batch_size_per_device=10 \
      --num_epoch=10 \
      --model="resnet50"
  ```

  运行此脚本，将在仅有50张金鱼图片的迷你 ImageNet 数据集上，训练出一个分类模型，利用它，你可以对金鱼图片进行分类。

  不要着急，如果您需要在完整的 ImageNet2012 数据集上进行训练，请参考：[OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns)仓库。



### 评估(Evaluate)

你可以使用自己训练好的模型，或者我们提供的 [resnet50_v1.5_model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/resnet_v15_of_best_model_val_top1_77318.tgz ) （解压后放入当前目录），对resnet50模型的精度进行评估。

只需运行：

```
sh evaluate.sh
```

即可获得训练好的模型在50000张验证集上的准确率：

```
Time stamp: 2020-07-27-09:28:28
Restoring model from resnet_v15_of_best_model_val_top1_77318.
I0727 09:28:28.773988162    8411 ev_epoll_linux.c:82]        Use of signals is disabled. Epoll engine will not be used
Loading data from /dataset/ImageNet/ofrecord/validation
validation: epoch 0, iter 195, top_1: 0.773277, top_k: 0.936058, samples/s: 1578.325
validation: epoch 0, iter 195, top_1: 0.773237, top_k: 0.936078, samples/s: 1692.303
validation: epoch 0, iter 195, top_1: 0.773297, top_k: 0.936018, samples/s: 1686.896
```

> 执行 `sh evaluate.sh` 前，确保准备了 ImageNet(2012) 的验证集，验证集制作方法请参考：[OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns)仓库。

从3轮的评估结果来看，我们的模型在 ImageNet(2012) 上已经达到了77.32+%的 top1 精度。

最后，恭喜你！完成了 Resnet 模型在 ImageNet 上完整的训练/验证、推理和评估，为自己鼓个掌吧！



## 更详细的说明 Details

### 分布式训练

**简单而易用的分布式，是 OneFlow 的主打特色之一。**

OneFlow 框架从底层设计上，就原生支持高效的分布式训练。尤其对于分布式的数据并行，用户完全不用操心算法从单机单卡扩展到多机多卡时，数据如何划分以及同步的问题。也就是说，使用 OneFlow，用户以单机单卡的视角写好的代码，**自动具备多机多卡分布式数据并行的能力。**


#### 如何配置并运行分布式训练？

还是以上面"快速开始"部分演示的代码为例，在 `train.sh` 中，只要用 `--num_nodes` 指定节点（机器）个数，同时用 `--node_ips` 指定节点的 IP 地址，然后用 `--gpu_num_per_node` 指定每个节点上使用的卡数，就轻松地完成了分布式的配置。

例如，想要在2机8卡上进行分布式训练，像下面这样配置：

```
# train.sh
python3 of_cnn_train_val.py \
    --num_nodes=2 \
    --node_ips="192.168.1.1, 192.168.1.2"
    --gpu_num_per_node=4 \
    ...
    --model="resnet50"
```

然后分别在两台机器上，同时执行：

```
./train.sh
```

程序启动后，通过 `watch -n 0.1 nvidia-smi` 命令可以看到，两台机器的 GPU 都开始了工作。一段时间后，会在 `--node_ips` 设置中的第一台机器的屏幕上，打印输出。


### 混合精度训练与预测

目前，OneFlow 已经原生支持 float16/float32 的混合精度训练。训练时，模型参数（权重）使用 float16 进行训练，同时保留 float32 用作梯度更新和计算过程。由于参数的存储减半，会带来训练速度的提升。

在 OneFlow 中开启 float16/float32 的混合精度训练模式，ResNet50 的训练速度理论上能达到`1.7`倍的加速。


#### 如何开启 float16 / float32 混合精度训练？

只需要在 `train.sh` 脚本中添加参数 `--use_fp16=True` 即可。

#### 混合精度模型

我们为您提供了一个在 ImageNet2012 完整训练了90个 epoch 的混合精度模型，Top_1：77.33%

您可以直接下载使用：[resnet50_v15_fp16](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/resnet_fp16_of_best_model_val_top1_77330.zip)



## 进阶 Advanced

### 参数对齐

OneFlow 的 ResNet50 实现，为了保证和[英伟达的 Mxnet 版实现](https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5)对齐，我们从 learning rate 学习率，优化器 Optimizer 的选择，数据增强的图像参数设定，到更细的每一层网络的形态，bias，weight 初始化等都做了细致且几乎完全一致的对齐工作。具体的参数对齐工作，请参考：[OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns) 仓库



###  数据集制作

#### 用于图像分类数据集简介

用于图像分类的公开数据集有CIFAR，ImageNet 等等，这些数据集中，是以 jpeg 的格式提供原始的图片。

- [CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html)
  是由Hinton 的学生 Alex Krizhevsky 和 Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。包括CIFAR-10和CIFAR-100。

- [ImageNet](http://image-net.org/index)
  ImageNet 数据集，一般是指2010-2017年间大规模视觉识别竞赛 (ILSVRC) 的所使用的数据集的统称。ImageNet 数据从2010年来稍有变化，常用 ImageNet-2012 数据集包含1000个类别，其中训练集包含1,281,167张图片，每个类别数据732至1300张不等，验证集包含50,000张图片，平均每个类别50张图片。

完整的 ImageNet(2012)制作过程，请参考 tools 目录下的[README说明](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/tools/README.md)



### OneFlow 模型转 ONNX 模型

#### 简介

 **ONNX (Open Neural Network Exchange)**  是一种较为广泛使用的神经网络中间格式，通过 ONNX 格式，OneFlow 模型可以被许多部署框架（如 OpenVINO、ONNX Runtime 和移动端的 ncnn、tnn、TEngine 等）所使用。这一节介绍如何将训练好的 ResNet50 v1.5 模型转换为 ONNX 模型并验证正确性。

#### 快速上手

我们提供了完整代码：[resnet\_to\_onnx.py](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/Classification/cnns/resnet_to_onnx.py)  帮你轻松完成模型的转换和测试的工作

 **步骤一：** 下载预训练模型：[resnet50_v1.5_model](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/resnet_v15_of_best_model_val_top1_77318.tgz ) ，解压后放入当前目录

 **步骤二：** 执行：`python3 resnet_to_onnx.py `

此代码将完成 OneFlow 模型 -> ONNX 模型的转化，然后使用 ONNX Runtime 加载转换后的模型对单张图片进行测试。测试图片如下：

<div align="center">
    <img src="imgs/tiger.jpg" align='center'/>
</div>
> ​                                             图片来源：https://en.wikipedia.org/wiki/Tiger

输出：

```python
Convert to onnx success! >>  onnx/model/resnet_v15_of_best_model_val_top1_77318.onnx
data/tiger.jpg
Are the results equal? Yes
Class: tiger, Panthera tigris; score: 0.8112028241157532
```



#### 如何生成 ONNX 模型

上面的示例代码，介绍了如何转换 OneFlow 的 ResNet 模型至 ONNX 模型，并给出了一个利用 onnx runtime 进行预测的例子，同样，你也可以利用下面的步骤来完成自己训练的 ResNet 或其他模型的转换。

**步骤一：将模型权重保存到本地**

首先指定待转换的 OneFlow 模型路径，然后指定转换后的 ONNX 模型存放路径，例如示例中：

```python
#set up your model path
flow_weights_path = 'resnet_v15_of_best_model_val_top1_77318'
onnx_model_dir = 'onnx/model'
```

**步骤二：新建一个用于推理的 job function**

然后新建一个用于推理的 job function，它只包含网络结构本身，不包含读取 OFRecord 的算子，并且直接接受 numpy 数组形式的输入。可参考 `resnet\_to\_onnx.py` 中的 `InferenceNet`。

**步骤三：调用 `flow.onnx.export `方法**

接下来代码中会调用 `oneflow_to_onnx()` 方法，此方法包含了核心的模型转换方法： `flow.onnx.export()`

 **`flow.onnx.export`** 将从 OneFlow 网络得到 ONNX 模型，它的第一个参数是上文所说的专用于推理的 job function，第二个参数是 OneFlow 模型路径，第三个参数是（转换后）ONNX 模型的存放路径

```python
onnx_model = oneflow_to_onnx(InferenceNet, flow_weights_path, onnx_model_dir, external_data=False)
```

#### 验证 ONNX 模型的正确性

生成 ONNX 模型之后可以使用 ONNX Runtime 运行 ONNX 模型，以验证 OneFlow 模型和 ONNX 模型能够在相同的输入下产生相同的结果。相应的代码在 resnet\_to\_onnx.py 的 `check_equality`。
