# Data input
Machine learning is drived by data. Data loading and preprocess need give attention to both efficiency and scalability. OneFlow supports two methods to load data:

- One is very flexible, use numpy ndarray as input, which means the training or predicting task can take a set of numpy data as input.

- Another way is the ` data pipeline` of OneFlow. Data pipeline only can take specific formats data. For example, `ofrecord_reader` loading data in OFRecord format (similar as: TFRecord )

#### Comparison

The method of loading data by using numpy ndarray is easier, but it only suitable for small scale of data.When scale of data is too large. It may stuck on preparing numpy data.Thus, we recommend using the following methods at the beginning of the project and when the data structure is unclear.

The data pipeline method in OneFlow may looks more complicated. But it use multithreading and data pipeline technology made the following data loading, preprocessing and enhancing more efficient.Thus, we recommend using it for experienced projects.


## Use numpy as data input
### For example

In Oneflow, during the process of training or predicting, can directly use numpy ndarray as data input:

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

In the script above, we use  `@flow.global_function` to define a --test_job(). Its input are images and labels. We using numpy format images and labels as input.

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
def test_job(images:tp.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
             labels:tp.Numpy.Placeholder((32, ), dtype=flow.int32)) -> Tuple[tp.Numpy, tp.Numpy]:
```
#### Using
We need prepare the numpy ndarray first. In example, it generates the `images_in` and `labels_in` according to the input data type and shape.
```python
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
```

Then put  `images_in` and `labels_in` as input in  `test_job`  then calculating. After that, it return the results and stored in  `images` and `labels`.
```python
  images, labels = test_job(images_in, labels_in)
```

Normally is be used in the cycle of training and predicting task.This simplified example used the job function at a time.

Something need to be explained about Placeholder:

* `oneflow.typing.Numpy.Placeholder` represents Numpy ndarray type Placeholder, there are also various placeholder types in OneFlow that correspond to 'list of ndarray' and more complex forms.You can refer to(../extended_topics/consistent_mirrored.md).

* When we call the job function, the parameters and results are `numpy` data, not the placeholders

## Use the data flow of OneFlow
The data flow of OneFlow decoupling the data loading and data preprocess:

- The data loading currently supports  `data.ofrecord_reader` and `data.coco_reader`. Support respective  `OFRecord`  and coco dataset. Reading other types of data can be achieved by custom extensions.

- The preprocessing data is using the data flow method. Supports the combination of various data preprocessing operators. And it also can be custom extensions.

### For example
The following example read `OFRecord` data to preprocess the images in ImageNet dataset: 
The complete code can be downloaded here[of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

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
In order to run the above script, we need a ofrecord dataset. We can [ load and prepare OFRecord dataset](../extended_topics/how_to_make_ofdataset.md) or we have a package which have 64 images called  [part-00000](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000) .

Replace the `/path/to/ImageNet/ofrecord` to  `part-00000` in the above code, and then run it.
```
python of_data_pipeline.py
```
We are expecting the following result:
```
(64, 3, 224, 224) (64,)
```
### Script Explanation
There are two stage in data processing in OneFlow: **loading data** and **preprocessing data**.

- Data loading uses  `ofrecord_reader`. It need specify the path and other parameters. Please refer to [ofrecord_reader api](https://oneflow-api.readthedocs.io/en/latest/data.html?highlight=ofrecord_reader#oneflow.data.ofrecord_reader)

- Data preprocessing is a multi-stage process. `OFRecordImageDecoderRandomCrop` is for decoding the picture and cropping randomly, `Resize` adjusts the cropped picture to the size of 224x224 and `CropMirrorNormalize` regularize the picture.The label part only need decoding.

OneFlow provides some operators for data loading and preprocessing. Please refer to [Data Pipeline Api](https://oneflow-api.readthedocs.io/en/latest/data.html) for details.These operators will be continuously enriched and optimized in the future, users can also define their customized operators to meet specific needs.

