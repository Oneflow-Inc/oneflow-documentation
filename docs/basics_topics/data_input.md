# 数据输入
机器学习是一种数据驱动的技术，数据加载和预处理需要兼顾效率和可扩展性。OneFlow提供了两种加载数据的方法：

- 一种是非常灵活的，以numpy ndarray为接口的方法，也就是说OneFlow的训练或者预测任务能够接收一组numpy数据作为输入。
- 另外一种方法是OneFlow的`数据流水线`，数据流水线方式只能够接受特定格式的数据文件，如：通过`ofrecord_reader`加载的OFRecord格式的数据(类似于TFRecord)

#### 优劣对比

numpy ndarray的方式简单方便，但适合小数据量的情况。因为但当数据量过大时，可能在准备numpy数据上遭遇时间瓶颈。因此，推荐在项目的初始阶段，数据结构没有确定的情况下采用此种方式；

OneFlow的数据流水线的方式，看上去稍显复杂，实际则采用了多线程和数据流水线等技术使得加载数据以及后面的数据预处理、数据增强的效率更高。因此，推荐成熟的项目使用。





## 使用Numpy作为数据输入
### 运行一个例子

在Oneflow中，你可以在训练/预测过程中，直接使用numpy/ndarray类型的数据作为输入，下面是一个完整的例子：

```python
# feed_numpy.py
import numpy as np
import oneflow as flow


@flow.global_function(flow.function_config())
def test_job(images=flow.FixedTensorDef((32, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((32,), dtype=flow.int32)):
    # do something with images or labels
    return images, labels


if __name__ == '__main__':
    images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
    labels_in = np.random.randint(-10, 10, (32,)).astype(np.int32)
    images, labels = test_job(images_in, labels_in).get()
    print(images.shape, labels.shape)
```

在上面的代码中，我们用@flow.global_function定义了一个预测任务—test_job()，其输入为images和labels，我们可以直接通过numpy格式的images和labels作为其数据输入。

你可以下载完整代码：[feed_numpy.py](../code/basics_topics/feed_numpy.py)，然后用python执行即可，如：

```bash
python feed_numpy.py
```
您将得到如下结果
```bash
(32, 1, 28, 28) (32,)
```
### 代码解析
当用户要用OneFlow进行一个深度学习的训练或者预测任务的时候，需要定义一个任务/作业函数（Job Function），然后再使用这个Job函数。先`定义`再`使用`是两个基本的步骤。要实现numpy数据输入功能，就需要在`定义`和`使用`任务函数的地方特殊处理一下，下面就分别说一下。

#### 定义
定义的地方需要声明一下有哪些输入，以及这些输入的形状和数据类型等信息。下面这段代码就是定义Job函数的地方。`test_job`是Job函数的函数名，例子中它有两个输入：`images`和`labels`，而且分别有自己的形状和数据类型。
```python
def test_job(images=flow.FixedTensorDef((32, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((32, ), dtype=flow.int32)):
```
#### 使用
在使用之前需要先准备好需要被输入的numpy的ndarray，例子中按照输入的形状和数据类型的要求随机生成了输入：`images_in`和`labels_in`。
```python
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
```

然后把`images_in`和`labels_in`作为`test_job`的输入传入并且进行计算，并返回计算结果保存到`images`和`labels`里。
```python
  images, labels = test_job(images_in, labels_in).get()
```

一般使用的地方都是在一个训练或者预测任务的循环中，这个简化的例子就使用了一次作业函数。

有关Job函数参数的两点说明：

* 1 - 例子中`FixedTensorDef`返回的是一个`占位符`，类似tensorflow中placeholder的概念，OneFlow中还可以用`MirroredTensorDef`方式生成占位符，这两种方式的区别参考[两类blob](https://github.com/Oneflow-Inc/oneflow-documentation/blob/model_mixed_parallel/docs/extended_topics/consistent_mirrored.md#%E4%B8%A4%E7%B1%BBblob);

* 2 - 作业函数支持多个参数，每个参数都必须是下面几种中的一种：1. 一个`占位符`  2. 一个由`占位符`组成的列表（list）、元组（tuple）或者字典(dict)

**总结**：在定义Job函数的时候把job函数的输入定义成`占位符`的形式，当使用job函数的时候输入相应的numpy数组对象，这样就实现了把numpy数据送入网络进行训练或者预测。

## 使用OneFlow数据流水线
OneFlow数据流水线解耦了数据的加载和数据预处理过程：

- 数据的加载目前支持`data.ofrecord_reader`和`data.coco_reader`两种，分别支持OneFlow原生的`OFRecord`格式的文件和coco数据集，其他格式的reader可以通过自定义扩展；

- 数据预处理过程采用的是流水线的方式，支持各种数据预处理算子的组合，数据预处理算子也可以自定义扩展。

### 运行一个例子
下面就给一个完整的例子，这个例子读取的是`OFRecord`数据格式文件，处理的是ImageNet数据集中的图片。完整代码可以点此下载：[of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

```python
# of_data_pipeline.py
import oneflow as flow


@flow.global_function(flow.function_config())
def test_job():
    batch_size = 64
    color_space = 'RGB'
    with flow.fixed_placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader('/path/to/ImageNet/ofrecord',
                                             batch_size=batch_size,
                                             data_part_num=1,
                                             part_name_suffix_length=5,
                                             random_shuffle=True,
                                             shuffle_after_epoch=True)
        image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, "encoded",
                                                         color_space=color_space)
        label = flow.data.OFRecordRawDecoder(ofrecord, "class/label", shape=(), dtype=flow.int32)
        rsz = flow.image.Resize(image, resize_x=224, resize_y=224, color_space=color_space)

        rng = flow.random.CoinFlip(batch_size=batch_size)
        normal = flow.image.CropMirrorNormalize(rsz, mirror_blob=rng, color_space=color_space,
                                                mean=[123.68, 116.779, 103.939],
                                                std=[58.393, 57.12, 57.375],
                                                output_dtype=flow.float)
        return normal, label


if __name__ == '__main__':
    images, labels = test_job().get()
    print(images.shape, labels.shape)
```
为了运行上面这段脚本，需要一个ofrecord数据集，您可以[加载与准备OFRecord数据集](../extended_topics/how_to_make_ofdataset.md)或者下载我们准备的一个包含64张图片的ofrecord文件[part-00000](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00001)。完整代码：

上面这段脚本中`/path/to/ImageNet/ofrecord`替换为保存`part-00000`文件的目录，然后运行
```
python of_data_pipeline.py
```
将得到下面的输出：
```
(64, 3, 224, 224) (64,)
```
### 代码解析
OneFlow的数据处理流水线分为两个阶段：**数据加载**和**数据预处理**。

- 数据加载采用的是`ofrecord_reader`，需要指定ofrecord文件所在的目录，和一些其他参数，请参考[ofrecord_reader api](ofrecord_reader.api)
- 数据预处理是一个系列过程，`OFRecordImageDecoderRandomCrop`负责图片解码并随机做了裁剪，`Resize`把裁剪后的图片调整成224x224的大小，`CropMirrorNormalize`把图片进行了正则化。标签部分只需要进行解码`CropMirrorNormalize`。

OneFlow提供了一些数据加载和预处理的算子，详细请参考[数据流水线API](api)。未来会不断丰富和优化这些算子，用户也可以自己定义算子满足特定的需求。

