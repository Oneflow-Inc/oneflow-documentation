# Data Input
Machine learning is driven by data. Data loading and preprocessing require both efficiency and scalability. OneFlow supports two methods to load data:

* One way to do this is to pass a Numpy ndarray object as a parameter to the job function directly.

* Another approach is to use [DataLoader](https://oneflow.readthedocs.io/en/master/data.html) of OneFlow and its related operators. It can load and pre-process datasets of a particular format from the file system.

Working directly with Numpy data is easy and convenient but only for small amounts of data. Because when the amount of data is too large, there may be barrier in preparing the Numpy data. Therefore, this approach is more suitable for the initial stages of the project to quickly validate and improve the algorithm.

The DataLoader of OneFlow use techniques such as multi-threading and data pipelining which make data loading, data pre-processing more efficient.However, you need to [prepare dataset](../extended_topics/how_to_make_ofdataset.md) which already supported by Oneflow. Thus we recommend use that in mature projects.


## Use Numpy as Data Input
### Example
We can directly use Numpy ndarray as data input during training or predicting with OneFlow:

```python
# feed_numpy.py
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp
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

You can download code from [feed_numpy.py](../code/basics_topics/feed_numpy.py) and run it by:

```bash
python3 feed_numpy.py
```
Following output are expected:
```bash
(32, 1, 28, 28) (32,)
```

### Code Explanation
In the above code, we defined a job function `test_job()` with `images` and `labels` as inputs and annotate (note that the formal parameter is followed by “:” , not “=”) to specifies the shape and data type of the data.

Thus, the example generates Numpy data randomly (`images_in` and `labels_in`) according to the shape and data type requirements of the job function.
```python
 images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
```
Then directly pass the Numpy data `images_in` and `labels_in` as parameters when the job function is called.
```python
images, labels = test_job(images_in, labels_in)
```
The `oneflow.typing.Numpy.Placeholder` is the placeholder of Numpy `ndarray`. There are also various placeholders in OneFlow that can represent more complex forms of Numpy data. More details please refer to [The Definition and Call of Job Function](../extended_topics/job_function_define_call.md).

## Using DataLoader and Related Operators

Under the [oneflow.data](https://oneflow.readthedocs.io/en/master/data.html) module, there are DataLoader operators for loading datasets and associated data preprocessing operators.DataLoader is usually named as `data.xxx_reader`, such as the existing `data.ofrecord_reader` and `data.coco_reader` which support OneFlow's native `OFRecord` format and COCO dataset.

In addition, there are other data preprocessing operators that are used to process the data after DataLoader has been loaded. The following code uses `data.OFRecordImageDecoderRandomCrop` for random image cropping and `data.OFRecordRawDecoder` for image decoding. You can refer to the [API documentation](https://oneflow.readthedocs.io/en/master/index.html) for more details.

### Examples

The following example reads the `OFRecord` data format file and dealing with images from the ImageNet dataset. The complete code can be downloaded here: [of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py).

This script requires an OFRecord dataset and you can make your own one according to [this article](../extended_topics/how_to_make_ofdataset.md).

Or you can download the [part-00000](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000) that we have prepared for you which contains 64 images. Then replace `path/to/ImageNet/ofrecord` in the script with the directory where the `part-00000` file **is located** and run the script.

The following example is running a script with our pre-prepared dataset:

```
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000
sed -i "s:path/to/ImageNet/ofrecord:./:" of_data_pipeline.py
python3 of_data_pipeline.py
```

The following output are expected:

```
(64, 3, 224, 224) (64,)
```

#### Code Explanation

There are generally two stages in using OneFlow DataLoader: **Load Data** and **Preprocessing Data**.

`flow.data.ofrecord_reader` in the script is responsible for loading data from the file system into memory.


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

To specify the directory where the OFRecord file is located and some other parameters please refer to [data.ofrecord_reader](https://oneflow.readthedocs.io/en/master/data.html#oneflow.data. ofrecord_reader).

If the return value of the DataLoader is a basic data type. Then it can be used directly as an input to the downstream operator. Otherwise the data preprocessing operator needs to be called further for preprocessing.

For example, in the script:

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

`OFRecordImageDecoderRandomCrop` is responsible for randomly cropping the image, `OFRecordRawDecoder` is responsible for decoding the label directly from the ofrecord object. `image.Resize` resizes the cropped image to 224x224 and `CropMirrorNormalize` normalizes the image.

## More Formats Support by DataLoader

OneFlow provides a number of DataLoaders and preprocessing operators, refer to [oneflow.data](https://oneflow.readthedocs.io/en/master/data.html) for details. These operators will be enriched and optimized in the future.
