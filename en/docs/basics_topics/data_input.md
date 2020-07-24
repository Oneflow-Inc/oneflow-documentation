# Data input
Machine learning is drive by data. Loading and pretreatment need give attention to both efficiency and scalability. OneFlow support two methods to load data:

- One is very flexible, use numpy ndarray as input, which means the training and predicting task can take a set of numpy data as input.

- Another way is the ` data flow` of OneFlow. Data line only can take specific formats data. For example, `ofrecord_reader` loading data in OFRecord (similar as: TFRecord )

#### Comparison

The numpy ndarray is easier, but it only suitable for small scale of data.When scale of data is too large. May stuck on preparing numpy data.Thus, we recommend use the following methon at the beginning of the project and when the data structure is unclear.

The data flow method in OneFlow may looks more complicated. But it use multithreading and data flow technology made the following data load, data preprocessing and data to enhance more efficient.Thus, we recommend use for experienced projects.


## Use numpy as data input
### For example

In Oneflow, during the process of training or predicting, can directly use numpy ndarray as data input:

```python
# feed_numpy.py
import numpy as np
import oneflow as flow
import oneflow.typing as oft


@flow.global_function(flow.function_config())
def test_job(images:oft.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((32,), dtype=flow.int32)):
    # do something with images or labels
    return images, labels


if __name__ == '__main__':
    images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
    labels_in = np.random.randint(-10, 10, (32,)).astype(np.int32)
    images, labels = test_job(images_in, labels_in).get()
    print(images.shape, labels.shape)
```

In the script above, we use  `@flow.global_function` to define a --test_job(). Its input is images and labels. We using numpy format images and labels as input.

You can download complete script：[feed_numpy.py](../code/basics_topics/feed_numpy.py) and run by:

```bash
python feed_numpy.py
```
We are expecting the following results:
```bash
(32, 1, 28, 28) (32,)
```
### Code explanation
When user need use OneFlow to achieve a deep learning training or predicting task. We need to define a job function then use it.First, `define` then `use `is two basic steps.To achieved using numpy as input, need make some special configurations to `define` and `use` the job function. The following part will explain in details.

#### Definition
The definition part need declare what input do we have and the shape and data type of the input.The following part is where to define the job function.`test_job`  is the name of job function. The example have two inputs: `images` and `labels`. Also have the data shape and type of the input.
```python
def test_job(images:oft.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((32, ), dtype=flow.int32)):
```
#### Using
We need prepared the numpy ndarray first. In example, it generates the `images_in` and `labels_in` according to the input data type and shape.
```python
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
```

Then put  `images_in` and `labels_in` as input in  `test_job`  then calculating. After that, it return the results and stored in  `images` and `labels`.
```python
  images, labels = test_job(images_in, labels_in).get()
```

Normally is be used in the cycle of training and predicting task.This simplified example used the job function at a time.

Two things need be explained in job function:

* In  `oft.Numpy.Placeholder`, the return object is a place holder and also the place holder can be generate by `oft.ListNumpy.Placeholder` in OneFlow. The difference between this two please renference [two type of blob](../extended_topics/consistent_mirrored.md).

* 2 - 任务函数支持多个参数，每个参数都必须是下面几种中的一种：1. 一个`占位符`  2. 一个由`占位符`组成的列表(list)

**总结**：在定义 job 函数的时候把 job 函数的输入定义成占位符的形式，当使用 job 函数的时候输入相应的 numpy 数组对象，这样就把 numpy 数据送入了网络进行训练或者预测。

## 使用OneFlow数据流水线
OneFlow 数据流水线解耦了数据的加载和数据预处理过程：

- 数据的加载目前支持 `data.ofrecord_reader` 和 `data.coco_reader` 两种，分别支持 OneFlow 原生的 `OFRecord` 格式的文件和 coco 数据集，其他格式的 reader 可以通过自定义扩展；

- 数据预处理过程采用的是流水线的方式，支持各种数据预处理算子的组合，数据预处理算子也可以自定义扩展。

### 运行一个例子
下面就给一个完整的例子，这个例子读取的是 `OFRecord` 数据格式文件，处理的是 ImageNet 数据集中的图片。完整代码可以点此下载：[of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

```python
# of_data_pipeline.py
import oneflow as flow


@flow.global_function(flow.function_config())
def test_job():
    batch_size = 64
    color_space = 'RGB'
    with flow.scope.placement("cpu", "0:0"):
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
为了运行上面这段脚本，需要一个 ofrecord 数据集，您可以[加载与准备OFRecord数据集](../extended_topics/how_to_make_ofdataset.md)或者下载我们准备的一个包含64张图片的 ofrecord 文件 [part-00000](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000) 。

上面这段脚本中 `/path/to/ImageNet/ofrecord` 替换为保存 `part-00000` 文件的目录，然后运行
```
python of_data_pipeline.py
```
将得到下面的输出：
```
(64, 3, 224, 224) (64,)
```
### 代码解析
OneFlow的数据处理流水线分为两个阶段： **数据加载** 和 **数据预处理** 。

- 数据加载采用的是 `ofrecord_reader`，需要指定 ofrecord 文件所在的目录，和一些其他参数，请参考 [ofrecord_reader api](../api/data.html?highlight=ofrecord_reader#oneflow.data.ofrecord_reader)

- 数据预处理是一个系列过程，`OFRecordImageDecoderRandomCrop` 负责图片解码并随机做了裁剪，`Resize` 把裁剪后的图片调整成224x224的大小， `CropMirrorNormalize` 把图片进行了正则化。标签部分只需要进行解码 `CropMirrorNormalize` 。

OneFlow提供了一些数据加载和预处理的算子，详细请参考[数据流水线API](../api/data.html)。未来会不断丰富和优化这些算子，用户也可以自己定义算子满足特定的需求。

