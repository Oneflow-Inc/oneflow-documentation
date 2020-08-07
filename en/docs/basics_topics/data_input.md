# Data input
Machine learning is driven by data. Data loading and preprocessing require both efficiency and scalability. OneFlow supports two methods to load data:

* One is very flexible: use numpy ndarray as input, which means the training or predicting task can take a set of numpy data as input;

* Another way is `data pipeline` of OneFlow. Data pipeline can only take data in specific formats. For example, `ofrecord_reader` loading data in OFRecord format (similar as: TFRecord )

#### Comparison

The method of loading data by using numpy ndarray is easier, but it only suitable for small scale of data. When the amount of data is too large, the framework may stuck on preparing numpy data procedure. Therefore, it is recommended to use this method in the initial stage of the project when the data structure is unclear.

The data pipelining method in OneFlow looks a little complicated. But it use multithreading and data pipelining technology to enhance data loading and preprocessing which is more efficiency. Therefore, we recommend using it for mature projects.


## Use numpy as data input
### Example
We can directly use numpy ndarray as data input during training or predicting with OneFlow:

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

As the code above shows, we use  `@flow.global_function` to define a job function `test_job()`. Its input are images and labels. We use numpy as input when we call the job function.

You can download code from [feed_numpy.py](../code/basics_topics/feed_numpy.py) and run it by:

```bash
python3 feed_numpy.py
```
Following output are expected:
```bash
(32, 1, 28, 28) (32,)
```

### Code explanation
In article [Recognition of MNIST Handwritten Digits](../quick_start/lenet_mnist.md), we have introduced that there are two two basic steps about using of job function: "define a job function" and "call a job function". We will explain how to use numpy as input of job function in 'define' and 'call' stage.

#### Definition phase
When define a job function, we should specify the annotation of parameters with placeholder type in `oneflow.typing`. Declare data type and data shape of parameters.

```python
def test_job(
    images: tp.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((32,), dtype=flow.int32),
) -> Tuple[tp.Numpy, tp.Numpy]:
    # do something with images or labels
    return (images, labels)
```

In above code, we declare `images` and `labels` two input parameters. They both are the placeholder of `oneflow.typing.Numpy`. When we call the job function, the `numpy` data with same shape and type should be passed in.

#### Call phase
When we call the job function, we need prepare the numpy ndarray first. In this example, we use random data to generate the `images_in` and `labels_in` whose data shape and type are identical to placeholders of job function.
```python
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
```

Then `images_in` and `labels_in` are passed to `test_job` and the job function starts to caculate. 

After that, job function returns the results that stored in  `images` and `labels`.
```python
  images, labels = test_job(images_in, labels_in)
```

We usually call job function looply but the simplified example above calls once only.

Other things needed to be explained about Placeholder:

* `oneflow.typing.Numpy.Placeholder` represents `numpy ndarray` type placeholder, there are also various placeholder types in OneFlow (eg. "list of ndarray" representation). You can refer to[Call and Definition of Job Function](../extended_topics/job_function_define_call.md) for details

* When we call the job function, the parameters and results are `numpy` data, not the placeholders

## Use data pipelining
The data pipelining of OneFlow decouples the data loading and data preprocessing:

* There are two kinds of data readers `data.ofrecord_reader` and `data.coco_reader` in OneFlow so far which support `OFRecord`  and coco dataset format respectivly. You can implement custom reader for specific data format.

* The data preprocessing process is pipelined, which supports the combination of various data preprocessing operators. And it can also be customized.

### Example
The following example shows how to read `OFRecord` data and preprocess it from ImageNet dataset: 
The code can be downloaded from [of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

```python
# of_data_pipeline.py
import oneflow as flow
import oneflow.typing as tp
from typing import Tuple


@flow.global_function(type="predict")
def test_job() -> Tuple[tp.Numpy, tp.Numpy]:
    batch_size = 64
    color_space = "RGB"
    with flow.scope.placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader(
            "path/to/ImageNet/ofrecord",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            random_shuffle=True,
            shuffle_after_epoch=True,
        )
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
        return normal, label


if __name__ == "__main__":
    images, labels = test_job()
    print(images.shape, labels.shape)
```
A ofrecord dataset is required to run the code above. We can making it by refering to [load and prepare OFRecord dataset](../extended_topics/how_to_make_ofdataset.md) or download the small prepared dataset which containing 64 images [part-00000](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000) .

We should replace the `/path/to/ImageNet/ofrecord` with the path to the folder containing `part-00000` file in the above code, and then run the script.
```
python3 of_data_pipeline.py
```
Following results are expected:
```
(64, 3, 224, 224) (64,)
```
### Code Explanation
There are two stage in data processing in OneFlow: **data loading** and **data preprocessing**.

* The method `ofrecord_reader` is used for data loading which needs parameters to specify path to dataset and others. Please refer to [ofrecord_reader api](https://oneflow-api.readthedocs.io/en/latest/data.html?highlight=ofrecord_reader#oneflow.data.ofrecord_reader) for details.

* Data preprocessing is a multi-stage process. `OFRecordImageDecoderRandomCrop` is for decoding the picture and cropping randomly. `Resize` resizes the cropped picture to 224x224 and `CropMirrorNormalize` regularizes the pictures. The label only need to be decoded.

OneFlow provides some operators for data loading and preprocessing. Please refer to [Data Pipeline Api](https://oneflow-api.readthedocs.io/en/latest/data.html) for details. These operators will be continuously enriched and optimized in the future, users can also define their customized operators to meet specific needs.

