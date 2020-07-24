# Data input
Machine learning is a data-driven technology. Data loading and preprocessing need to balance both efficiency and scalability. OneFlow provides two ways to load data.

- One is very flexible, using Numpy ndarray as the interface, which means that OneFlow training or prediction tasks can take a set of Numpy data as input.

- Another way is the ` data flow` of OneFlow. This way can only take data of a specific format, such as: OFRecord format data loaded by `ofrecord_reader` (similar to TFRecord)

#### Comparison

the way of numpy ndarray is simple and convenient, but only suitable for small-scale data.Because when the amount data is too large, there may be a time bottleneck in preparing numpy data.Therefore, it is recommended to use this way in the earlier stage of the project when the data structure is not determined;

The way of OneFlow looks a little more complicated, but in fact, it uses technologies such as multithreading and data pipeline to make data loading and subsequent data preprocessing and data augmentation more efficient.Therefore, it is recommended for more mature projects.


## Use numpy as data input
### Example

In Oneflow, you can directly use numpy ndarray type data as input during the training/prediction process. The following is a complete example

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

In the script above, we use `@flow.global_function` to define a --test_job(). It takes images and labels as input. We can directly pass images and labels in numpy format as its data input.

You can download the complete script：[feed_numpy.py](../code/basics_topics/feed_numpy.py) and run by Python,

```bash
python feed_numpy.py
```
The following results are expected
```bash
(32, 1, 28, 28) (32,)
```
### The explanation of the code
When a user wants to use OneFlow for a deep learning training or prediction task, they need to define a job/job function (Job Function), and then use the Job function.First, `define` then `use ` are the two basic steps.To achieved using numpy as input, need make some special configurations to `define` and `use` the job function. The following part will explain in details.

#### Definition
Where you define it, you need to declare what inputs are, as well as the shape and data type of the inputs.The following part is where to define the job function.`test_job`  is the name of job function. The example have two inputs: `images` and `labels`. Also have the data shape and type of the input.
```python
def test_job(images:oft.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((32, ), dtype=flow.int32)):
```
#### How to use
We need prepared the numpy ndarray first. In example, it generates the `images_in` and `labels_in` according to the input data type and shape.
```python
  images_in = np.random.uniform(-10, 10, (32, 1, 28, 28)).astype(np.float32)
  labels_in = np.random.randint(-10, 10, (32, )).astype(np.int32)
```

Then pass in `images_in` and `labels_in` as the input of `test_job` and perform calculations, and return the calculation results and save them to `images` and `labels`.
```python
  images, labels = test_job(images_in, labels_in).get()
```

Normally it is used in a training or prediction task cycle. This simplified example uses a job function.

Two points about the parameters of the Job function

* In the example of `oft.Numpy.Placeholder`, the return object is a place holder. The place holder can also be generate by `oft.ListNumpy.Placeholder` in OneFlow. Please refer to [two type of blob](../extended_topics/consistent_mirrored.md) for the difference between the two.

* The job function supports for multiple parameters. and each parameter must be one of the following: a `place holder` and the list (list) of` place holder`.

**Summary**: When defining the job function, define the input of the job function as a placeholder. When using the job function, take the corresponding numpy array object as input, so that the numpy data is sent to the network for training or prediction.

## Use the data pipeline of OneFlow
The data pipeline of OneFlow decoupling the data loading and data preprocessing process:

- Data loading currently supports two types: `data.ofrecord_reader` and `data.coco_reader`, which respectively support OneFlow's `OFRecord` format files and coco dataset. Readers in other formats can be extended through customization;

- The data preprocessing process adopts a pipeline method, which supports the combination of various data preprocessing operators, and the data preprocessing operators can also be customized and extended.

### Example
Here is a complete example. This example reads the `OFRecord` data format file and processes the images in the ImageNet datasetThe complete code can be downloaded here: [of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

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
In order to run the script above, an ofrecord data set is required. You can [load and prepare the Ofrecord data set](../extended_topics/how_to_make_ofdataset.md) or download an ofrecord file containing 64 pictures prepared by us

part-00000</1 ></p> 

Replace the `/path/to/ImageNet/ofrecord` to the path to `part-00000` and run


```
python of_data_pipeline.py
```


The following result is expected


```
(64, 3, 224, 224) (64,)
```



### The explanation of the code

There are two stage in data processing in OneFlow: **loading data** and **preprocessing data**.

- Loading is use  `ofrecord_reader`. It need specify the path and other parameters. More information reference [ofrecord_reader api](../api/data.html?highlight=ofrecord_reader#oneflow.data.ofrecord_reader).

- Data preprocessing is a multi-stage process. `OFRecordImageDecoderRandomCrop` is for decoding the picture and random cropping, `Resize` adjusts the cropped picture to the size of 224x224 and `CropMirrorNormalize` 0> regularize the picture.`CropMirrorNormalize` if for decoding the labeled part.

OneFlow provides some operators for data loading and preprocessing. Please refer to [Data Pipeline Api](../api/data.html) for details.These operators will be continuously enriched and optimized in the future, and users can also define their customized operators to meet specific needs.

