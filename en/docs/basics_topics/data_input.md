# Data input
Machine learning is a data-driven technology. Data loading and preprocessing need to balance both efficiency and scalability. OneFlow provides two ways to load data.

- One is very flexible, using Numpy ndarray as the interface, which means that OneFlow training or prediction tasks can take a set of Numpy data as input.

- Another way is the ` data flow` of OneFlow. This way can only take data of a specific format, such as: OFRecord format data loaded by `ofrecord_reader` (similar to TFRecord)

#### Comparison

the way of numpy ndarray is simple and convenient, but only suitable for small-scale data.Because when the amount data is too large, there may be a time bottleneck in preparing numpy data.Therefore, it is recommended to use this way in the earlier stage of the project when the data structure is not determined;

The way of OneFlow looks a little more complicated, but in fact, it uses technologies such as multithreading and data pipeline to make data loading and subsequent data preprocessing and data augmentation more efficient.Therefore, it is recommended for more mature projects.


## Use numpy as data input
### Example

在Oneflow中，你可以在训练/预测过程中，直接使用 `numpy ndarray` 类型的数据作为输入，下面是一个完整的例子：

```python
# feed_numpy.py
import numpy as np
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple

@flow.global_function(type="predict")
def test_job(images:tp.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
             labels:tp.Numpy.Placeholder((32,), dtype=flow.int32)) -> Tuple[tp.Numpy, tp.Numpy]:
    # do something with images or labels
    return (images, labels)


if __name__ == '__main__':
    images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
    labels_in = np.random.randint(-10, 10, (32,)).astype(np.int32)
    images, labels = test_job(images_in, labels_in)
    print(images.shape, labels.shape)
```

在上面的代码中，我们用 `@flow.global_function` 定义了一个预测作业-- `test_job()`，其输入为 `images` 和 `labels` ，我们可以直接通过 `numpy` 格式的 `images_in` 和 `labels_in` 作为其数据输入。

下载完整代码：[feed_numpy.py](../code/basics_topics/feed_numpy.py) ，然后用 python 执行即可：

```bash
python feed_numpy.py
```
将得到如下结果
```bash
(32, 1, 28, 28) (32,)
```
### The explanation of the code
我们在快速入门的[识别 MNIST 手写体数字](../quick_start/lenet_mnist.md)一文中，已经了解到作业函数分为先 **定义** 再 **调用** 两个基本的步骤。我们来解析，如果要将 numpy 数据作为作业函数的输入，在作业函数的定义和调用阶段分别要如何做。

#### Definition
在作业函数定义时，指定参数类型为 `oneflow.typing` 中的类型作为数据占位符，声明输入变量的形状及数据类型。

```python
def test_job(images:tp.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
             labels:tp.Numpy.Placeholder((32, ), dtype=flow.int32)) -> Tuple[tp.Numpy, tp.Numpy]:
```

如以上代码中，声明了 `images` 和 `labels` 两个传入参数，它们都是 `oneflow.typing.Numpy`的占位符，在调用时，需要传入形状、数据类型一致的 `numpy` 数据。

#### 调用
调用作业函数时，准备好与作业函数中声明的占位符形状、数据类型一致的 `numpy ndarray` 数据，作为参数调用即可。

例子中按照输入的形状和数据类型的要求随机生成了输入：`images_in` 和 `labels_in` ：
```python
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
```

然后把 `images_in` 和 `labels_in` 作为 `test_job` 的输入传入并且进行计算，并返回计算结果保存到 `images` 和 `labels` 里。
```python
  images, labels = test_job(images_in, labels_in)
```

一般我们是在一个训练或者预测作业的循环中调用作业函数，以上简化的例子仅调用了一次作业函数。

关于占位符的其它说明：

* `oneflow.typing.Numpy.Placeholder` 表示的是 `numpy ndarray` 类型的占位符，OneFlow 中还有多种占位符，分别对应 `list of ndarray` 以及更复杂的形式。 具体可以参考[作业函数的定义与调用](../extended_topics/job_function_define_call.md);

* 调用作业函数时，传入的参数和返回的结果，都是 `numpy` 数据，而不是占位符

## Use the data pipeline of OneFlow
OneFlow 数据流水线解耦了数据的加载和数据预处理过程：

* Data loading currently supports two types: `data.ofrecord_reader` and `data.coco_reader`, which respectively support OneFlow's `OFRecord` format files and coco dataset. Readers in other formats can be extended through customization;

* The data preprocessing process adopts a pipeline method, which supports the combination of various data preprocessing operators, and the data preprocessing operators can also be customized and extended.

### Example
下面就给一个完整的例子，这个例子读取的是 `OFRecord` 数据格式文件，处理的是 ImageNet 数据集中的图片。完整代码可以点此下载：[of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

```python
# of_data_pipeline.py
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple

@flow.global_function(type="predict")
def test_job() -> Tuple[tp.Numpy, tp.Numpy]:
    batch_size = 64
    color_space = 'RGB'
    with flow.scope.placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader('path/to/ImageNet/ofrecord',
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
    images, labels = test_job()
    print(images.shape, labels.shape)
```
为了运行上面这段脚本，需要一个 ofrecord 数据集，你可以[加载与准备OFRecord数据集](../extended_topics/how_to_make_ofdataset.md)或者下载我们准备的一个包含64张图片的 ofrecord 文件 [part-00000](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000) 。

上面这段脚本中 `path/to/ImageNet/ofrecord` 替换为保存 `part-00000` 文件 **所在的目录**，然后运行
```
python of_data_pipeline.py
```
将得到下面的输出：
```
(64, 3, 224, 224) (64,)
```
### The explanation of the code
OneFlow的数据处理流水线分为两个阶段： **数据加载** 和 **数据预处理** 。

* 数据加载采用的是 `ofrecord_reader`，需要指定 ofrecord 文件所在的目录，和一些其他参数，请参考 [ofrecord_reader api](https://oneflow-api.readthedocs.io/en/latest/data.html?highlight=ofrecord_reader#oneflow.data.ofrecord_reader)

* Data preprocessing is a multi-stage process. `OFRecordImageDecoderRandomCrop` is for decoding the picture and random cropping, `Resize` adjusts the cropped picture to the size of 224x224 and `CropMirrorNormalize` 0> regularize the picture.标签部分只需要进行解码。

OneFlow提供了一些数据加载和预处理的算子，详细请参考[数据流水线API](https://oneflow-api.readthedocs.io/en/latest/data.html)。未来会不断丰富和优化这些算子，用户也可以自己定义算子满足特定的需求。

