在进行分布式训练时，OneFlow框架提供了两种角度看待数据与模型的关系，被称作`consistent`策略与`mirrored`策略。

其中的`mirrored`策略其它框架（如TensorFlow中的`tf.distribute.MirroredStrategy`)，选择数据并行的方式进行分布式训练。

`consistent`策略在OneFlow项目早期就有成熟的设计，是OneFlow的一大特色，使得OneFlow可以在分布式任务中灵活地选择数据并行、模型并行或者混合并行。

本文将介绍：

* 数据并行与模型并行的区别及适用场景

* 在分布式任务中采用`mirrored`策略及其特点

* 在分布式任务中采用`consistent`及其特点

## 数据并行与模型并行
为了更好地理解OneFlow中的`consistent`和`mirrored`策略，我们需要了解分布式任务中的数据并行、模型并行两种模式的区别。

为了更直观地展示地展示两者的差别，我们先看一个矩阵相乘的例子，我们也可以将其看作是去掉了`bias`、激活函数等的简化版MLP模型。
【I × W = O】
【公式：output(N,C2) <= MatrixIn(N,C1) \times Para(C1, C2)】

其中的I矩阵，N行C1列，代表输入的样本数据。

## 在OneFlow中使用mirrored策略


## 在OneFlow中使用consistent策略

