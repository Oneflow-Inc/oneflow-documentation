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

* The job function supports multiple parameters. But the parameters must be one of the following: a `place holder` and the list of` place holder`.

Summary: define the input of job function as place holder when defining the job function. When the job function receive the corresponding numpy array as input. The numpy array will send to the network for training or predicting.

## Use the data flow of OneFlow
The data flow of OneFlow decoupling the data loading and data pretreatment process:

- The data load is support  `data.ofrecord_reader` and `data.coco_reader` for now. Support respective  `OFRecord`  and coco dataset. Reading other type of data can achieve by custom extensions.

- The preprocessing data is using the data flow method. Support the different preprocessing to calculate their operators. And it also can be custom extensions.

### For example
The following example read   `OFRecord`  data and it is the image of ImageNet dataset.Name: [of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

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
In order to run the above script, we need a ofrecord dataset. We can [ load and prepare OFRecord dataset](../extended_topics/how_to_make_ofdataset.md) or we have a package which have 64 images called  [part-00000](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/part-00000) .

Replace the `/path/to/ImageNet/ofrecord` to  `part-00000` if you want use our package.
```
python of_data_pipeline.py
```
We are expecting the following result:
```
(64, 3, 224, 224) (64,)
```
### Script Explanation
There are two stage in data processing in OneFlow: **loading data** and **preprocessing data**.

- Loading is use  `ofrecord_reader`. It need specify the path and other parameters. More information reference [ofrecord_reader api](../api/data.html?highlight=ofrecord_reader#oneflow.data.ofrecord_reader).

- Data preprocessing is a multi-stage process. `OFRecordImageDecoderRandomCrop` is for decoding the picture and random cropping, `Resize` adjusts the cropped picture to the size of 224x224 and `CropMirrorNormalize` 0> regularize the picture.`CropMirrorNormalize` if for decoding the labeled part.

OneFlow provides some operators for data loading and preprocessing. Please refer to [Data Pipeline Api](../api/data.html) for details.These operators will be continuously enriched and optimized in the future, and users can also define their customized operators to meet specific needs.

