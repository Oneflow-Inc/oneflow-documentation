# 数据输入
深度学习是一种数据驱动的技术，为了兼顾易用性与效率， OneFlow 提供了两种“喂”数据给神经网络的方法：

- 一种方法，可以直接将 NumPy ndarray 对象作为参数传递给作业函数。也就是说 OneFlow 能够直接使用 NumPy 数据作为输入。
- 另外一种方法是使用 OneFlow 的 [DataLoader](https://oneflow.readthedocs.io/en/master/data.html) 及其相关算子，从文件系统加载特定格式的数据集并做预处理。


直接使用 NumPy 数据的方式简单方便，但仅适合小数据量的情况。因为当数据量过大时，可能在准备 NumPy 数据上遭遇效率瓶颈。因此，这种方式比较适合项目的初始阶段，快速验证和改进算法；

OneFlow 的 DataLoader 内部采用了多线程和数据流水线等技术使得数据加载、数据预处理等效率更高。但是，需要为已经支持的格式[准备数据集](../extended_topics/how_to_make_ofdataset.md)或为 OneFlow 暂时还不支持的格式[开发自己的 DataLoader](../extended_topics/implement_data_loader.md)。因此，推荐在成熟的项目中使用。


## 使用 Numpy 数据作为输入
### 运行例子

在 Oneflow 中，可以直接使用 NumPy 类型的数据作为作业函数的输入，下面是一个完整的例子：

```python
# feed_numpy.py
import numpy as np
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple


@flow.global_function(type="predict")
def test_job(
    images: tp.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((32,), dtype=flow.int32),
) -> Tuple[tp.Numpy, tp.Numpy]:
    # do something with images or labels
    return (images, labels)


if __name__ == "__main__":
    images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
    labels_in = np.random.randint(-10, 10, (32,)).astype(np.int32)
    images, labels = test_job(images_in, labels_in)
    print(images.shape, labels.shape)
```

下载完整代码：[feed_numpy.py](../code/basics_topics/feed_numpy.py) ，然后用 python 执行即可：

```bash
python feed_numpy.py
```
将得到如下结果
```bash
(32, 1, 28, 28) (32,)
```



### 代码解读
在上面的代码中，我们定义了一个作业函数 `test_job()`，其输入为 `images` 和 `labels` ，并且通过注解(注意形参后面是“:”，而不是“=”。)指定了数据的形状与数据类型。

因此，例子中按照作业函数对形状和数据类型的要求随机生成了 NumPy数据：`images_in` 和 `labels_in` ：
```python
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
```

并在调用作业函数时，直接将 NumPy 数据 `images_in` 和 `labels_in` 作为参数传递：
```python
images, labels = test_job(images_in, labels_in)
```

代码中的 `oneflow.typing.Numpy.Placeholder` 是 NumPy `ndarray` 对象的占位符，OneFlow 中还有多种占位符，可以表示更复杂的 NumPy 数据形式。具体可以参考[作业函数的定义与调用](../extended_topics/job_function_define_call.md)。

## 使用 DataLoader 及相关算子
在 [oneflow.data](https://oneflow.readthedocs.io/en/master/data.html) 模块下，有用于加载数据集的 DataLoader 算子以及相关的数据预处理算子。DataLoader 一般以 `data.xxx_reader` 的形式命名，如目前已有的 `data.ofrecord_reader` 和 `data.coco_reader`，分别支持 OneFlow 原生的 `OFRecord` 格式的文件和 COCO 数据集。

此外，在该模块下，还包含有其它数据预处理算子，用于处理 DataLoader 加载后的数据。如下文代码使用的 `data.OFRecordImageDecoderRandomCrop` 用于图片随机裁剪，`data.OFRecordRawDecoder` 用于图片解码。具体使用方法可以查阅 [API 文档](https://oneflow.readthedocs.io/en/master/index.html)。

### 运行例子
以下的例子，读取 `OFRecord` 数据格式文件，处理的是 ImageNet 数据集中的图片。完整代码可以点此下载：[of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

这个脚本，需要一个 OFRecord 数据集，你可以根据[这篇文章](../extended_topics/how_to_make_ofdataset.md)自己制作一个。

或者下载我们已经准备好的 [part-00000](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000) 数据文件，它包含了64张图片。并且，将脚本中的 `path/to/ImageNet/ofrecord` 替换为 `part-00000` 文件 **所在的目录**，然后运行脚本。

以下是使用我们预先准备的数据集运行脚本的例子:

```shell
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000
sed -i "s:path/to/ImageNet/ofrecord:./:" of_data_pipeline.py
python3 of_data_pipeline.py
```

将得到下面的输出：
```
(64, 3, 224, 224) (64,)
```
### 代码解读
使用 OneFlow DataLoader 一般为两个阶段： **数据加载** 和 **数据预处理** 。

脚本中 `flow.data.ofrecord_reader` 负责从文件系统中加载数据到内存。
```python
    ofrecord = flow.data.ofrecord_reader(
        "path/to/ImageNet/ofrecord",
        batch_size=batch_size,
        data_part_num=1,
        part_name_suffix_length=5,
        random_shuffle=True,
        shuffle_after_epoch=True,
    )
```

需要指定 OFRecord 格式文件所在的目录，和一些其他参数，请参考 [data.ofrecord_reader](https://oneflow.readthedocs.io/en/master/data.html#oneflow.data.ofrecord_reader)

DataLoader 的返回值，如果是简单的基本数据类型，那么可以直接作为下游的算子的输入，否则，需要继续调用数据预处理算子，进行预处理。

比如，在以上脚本中：
```python
    image = flow.data.OFRecordImageDecoderRandomCrop(
        ofrecord, "encoded", color_space=color_space
    )
    label = flow.data.OFRecordRawDecoder(
        ofrecord, "class/label", shape=(), dtype=flow.int32
    )
    rsz = flow.image.Resize(
        image, resize_x=224, resize_y=224, color_space=color_space
    )
    rng = flow.random.CoinFlip(batch_size=batch_size)
    normal = flow.image.CropMirrorNormalize(
        rsz,
        mirror_blob=rng,
        color_space=color_space,
        mean=[123.68, 116.779, 103.939],
        std=[58.393, 57.12, 57.375],
        output_dtype=flow.float,
    )
```

`OFRecordImageDecoderRandomCrop` 负责图片解码并随机做了裁剪，`OFRecordRawDecoder` 负责从 ofrecord 对象中直接解码出标签， `image.Resize` 把裁剪后的图片调整成224x224的大小， `CropMirrorNormalize` 把图片进行了正则化。

## 支持更多格式的 DataLoader
OneFlow 提供了一些 DataLoader 和预处理的算子，详细请参考 [oneflow.data](https://oneflow.readthedocs.io/en/master/data.html)。未来会不断丰富和优化这些算子，用户也可以参考 [这篇文章](../extended_topics/implement_data_loader.md) 自定义 DataLoader 满足特定的需求。
