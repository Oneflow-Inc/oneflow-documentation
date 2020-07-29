In [data_input](../basics_topics/data_input.md) section， we know that because of the support like Multi-threading, scheduling of resources in OneFlow, the efficiency of processing data will be highter in OneFlow data-pipeline. Also, we learn about the operation in data-pipeline.

In [OFRecord](ofrecord.md) section, we learn about the storage format of OFRecord files.

In this section, we will focus on the loading and producing of OneFlow's OFRecord dataset. Mainly including:

* The organization of OFRecord dataset

* The multiple ways of loading OFRecord dataset

* The transition of OFRecord dataset to other dataformats

## What is OFRecord dataset
In [OFRecord](ofrecord.md) section. We have introduced the storage format of `OFRecord file `, and we also have known about what is `OFRecord file`

OFRecord dataset is **the collection of OFRecord files **.According to the OneFlow convention file format, we store `multiple OFRecord files` in the same directory, then we will get OFRecord dataset.

By default, The files in OFRecord dataset directory are uniformly named in the way of `part-xxx`, where "xxx" is the file id starting from zero, and there can be two choices of completement or non-completement.

Here is the example of using non-completement name style:
```
mnist_kaggle/train/
├── part-0
├── part-1
├── part-10
├── part-11
├── part-12
├── part-13
├── part-14
├── part-15
├── part-2
├── part-3
├── part-4
├── part-5
├── part-6
├── part-7
├── part-8
└── part-9
```

Here is the example of using completement name style:
```
mnist_kaggle/train/
├── part-00000
├── part-00001
├── part-00002
├── part-00003
├── part-00004
├── part-00005
├── part-00006
├── part-00007
├── part-00008
├── part-00009
├── part-00010
├── part-00011
├── part-00012
├── part-00013
├── part-00014
├── part-00015
```
OneFlow adopts this convention, which is consistent with the default storage filename in `spark`, it is convenient to make and convert to OFRecord data by spark.

Actually, we can specify the filename prefix(`part-`)，whether to complete the filename id, how many bits to complete.We just need to keep the same parameters when loading dataset(described below)

OneFlow provides the API interface to load OFRecord dataset, so that we can enjoy the Multi-threading, pipeline and some other advantages brought by OneFlow framework by specifying the path of dataset directory.

## The method to load OFRecord dataset
We usually use `decode_ofrecord` to load and decode dataset; or use `ofrecord_reader` to load and preprocess dataset.

### `decode_ofrecord`
We can use `flow.data.decode_ofrecord` to load and decode the dataset at the same time. The APIinterface of `decode_ofrecord` is as follow：
```python
def decode_ofrecord(
    ofrecord_dir,
    blobs,
    batch_size=1,
    data_part_num=1,
    part_name_prefix="part-",
    part_name_suffix_length=-1,
    shuffle=False,
    buffer_size=1024,
    name=None,
)
```

Its common parameters and their meanings are as follows

* batch_size： sample number in a mini-batch training

* data_part_num: the number of OFRecord files in dataset

* part_name_prefix: the prefix of OFRecord files in dataset

* part_name_suffix_length：the suffix length of OFRecord files in dataset, like the filename `part-00001`, we should set `part_name_suffix_length` as 5, -1 means there is no complement.

* shuffle：Whether the order of data is randomly shuffled

* buffer_size：the sample number is data-pipeline. For example, when we set it as 1024, which means there are 1024 samples in buffer. Also, when we set the shuffle as True, it only shuffle the 1024 samples in buffer。

The required parameter `ofrecord_dir` is the path of dataset directory, `blobs` is a tuple, in which there is a `Feature`(refer to [OFRecord](ofrecord.md)) that needs to read the dataset. We will introduce how to define the `blobs` parameter as follow.

The complete code: [decode_ofrecord.py](../code/extended_topics/decode_ofrecord.py)

```python
import oneflow as flow

def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config


@flow.global_function(get_train_config())
def train_job():
  images = flow.data.BlobConf("images", 
          shape=(28, 28, 1), 
          dtype=flow.float, 
          codec=flow.data.RawCodec())
  labels = flow.data.BlobConf("labels", 
          shape=(1, 1), 
          dtype=flow.int32, 
          codec=flow.data.RawCodec())

  return flow.data.decode_ofrecord("./dataset/", (images, labels),
                                data_part_num=1,
                                batch_size=3)

def main():
  check_point = flow.train.CheckPoint()
  check_point.init()

  f0, f1 = train_job().get()
  print(f0.ndarray(), f1.ndarray())
  print(f0.shape, f1.shape)

if __name__ == '__main__':
  main()
```

For the above code, load the dataset in [OFRecord](ofrecord.md) - "Write the OFRecord object to a file" section.

After running code, we will get the results as follows：
```text
... [[0.5941235 ]
   [0.27485612]
   [0.4714867 ]
   ... [0.21632855]
   [0.15881447]
   [0.65982276]]]] [[[2]]

 [[3]]

 [[1]]]
(3, 28, 28, 1) (3, 1, 1)
```

As you can see, we use `flow.data.BlobConf` to declare the placeholders corresponding to `Feature` in dataset, the required parameters in `BlobConf` are：
```python
 BlobConf(name, shape, dtype, codec)
```

* name：The Key corresponding to Feature when making OFRecord files;

* shape：The shape corresponding to data, it needs to be consistent with the number of elements in Feature.Like the above `(28, 28, 1)` can be modified to `(14, 28*2, 1)` or `(28, 28)`；

* dtype：The data type, it needs to be consistent with the Feature data type written in dataset;

* codec：The decoder，OneFlow has `RawCodec`、`ImageCodec`、`BytesListCodec` and some other decoders.In the previous example, we use `RawCodec`.

When we get the placeholder by using `BlobConf`, we can get the data in dataset by using `decode_ofrecord`
```python
    flow.data.decode_ofrecord("./dataset/", (images, labels),
                            data_part_num=1,
                            batch_size=3)
```

Through the above examples, we can summarize the basic steps of using `decode_ofrecord` ：

* The placeholder is defined by `BlobConf`, which is used to extract the `Feature` in dataset

* We pass the placeholder defined in the previous step to `decode_ofrecord` by calling `decode_ofrecord`, and set some parameters to get data in dataset

It's convenient to extract the `Feature` in data by using `decode_ofrecord`. However, the types of preprocessing and decoder are limited.For more flexible data preprocessing, including custom user op, it is recommended to use `ofrecord_reader`.

### `ofrecord_reader`
In [data_input](../basics_topics/data_input.md) section, we have shown how to use `ofrecord_reader` this api to load and preprocess OFRecord data:

The complete code: [of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

```python
import oneflow as flow

@flow.global_function(flow.function_config())
def test_job():
  batch_size = 64
  color_space = 'RGB'
  with flow.scope.placement("cpu", "0:0"):
    ofrecord = flow.data.ofrecord_reader('/path/to/ImageNet/ofrecord',
                                         batch_size = batch_size,
                                         data_part_num = 1,
                                         part_name_suffix_length = 5,
                                         random_shuffle = True,
                                         shuffle_after_epoch = True)
    image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, "encoded",
                                                     color_space = color_space)
    label = flow.data.OFRecordRawDecoder(ofrecord, "class/label", shape = (), dtype = flow.int32)
    rsz = flow.image.Resize(image, resize_x = 224, resize_y = 224, color_space = color_space)

    rng = flow.random.CoinFlip(batch_size = batch_size)
    normal = flow.image.CropMirrorNormalize(rsz, mirror_blob = rng, color_space = color_space,
                                            mean = [123.68, 116.779, 103.939],
                                            std = [58.393, 57.12, 57.375],
                                            output_dtype = flow.float)
    return normal, label

if __name__ == '__main__':
  images, labels = test_job().get()
  print(images.shape, labels.shape)
```

The API interface of `ofrecord_reader` is as follow.
```python
def ofrecord_reader(
    ofrecord_dir,
    batch_size=1,
    data_part_num=1,
    part_name_prefix="part-",
    part_name_suffix_length=-1,
    random_shuffle=False,
    shuffle_buffer_size=1024,
    shuffle_after_epoch=False,
    name=None,
)
```

The advantage of using `ofrecord_reader` is we can preprocess data in the way of data processing pipeline, and we can customize preprocessing op with high flexibility and expansibility.

* you can refer to [data_input](../basics_topics/data_input.md) for data pipeline and preprocessing.

* you can refer to [user_op](user_op.md) for customizing op

## The transition between other dataformat data and OFRecord dataset
According to the the storage format of OFRecord file in [OFRecord](ofrecord.md) section and the filename format convention of OFRecord dataset introduced at the begining, we can make OFRecord dataset by ourselves.

To make things easier, we provide Spark's jar package, which is convenient to the interconversion between OFRecord and common data formats (such as TFRecord and JSON).

### The installation and launch of Spark
At first, we should download Spark and Spark-oneflow-connector：

* Download the [spark-2.4.0-bin-hadoop2.7](https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz) from the official website of Spark

* Download jar package at [there](https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/spark-oneflow-connector-assembly-0.1.0_int64.jar), which Spark needs to support the ofrecord file format

Then, unzip the `spark-2.4.0-bin-hadoop2.7.tgz` and configure the environment variable `SPARK_HOME`:
```shell
export SPARK_HOME=path/to/spark-2.4.0-bin-hadoop2.7
```

Here we can launch the pyspark shell with the following command：
```shell
pyspark --master "local[*]"\
 --jars spark-oneflow-connector-assembly-0.1.0_int64.jar\
 --packages org.tensorflow:spark-tensorflow-connector_2.11:1.13.1
```

```text
... Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.4.0
      /_/

Using Python version 3.6.10 (default, May  8 2020 02:54:21)
SparkSession available as 'spark'. >>> 
```

We can complete the interconversion between OFRecord dataset and other data formats in launched pyspark shell.

### Use Spark to view OFRecord dataset
We can view OFRecord data with following code：
```
spark.read.format("ofrecord").load("file:///path/to/ofrecord_file").show()
```
The first 20 data are displayed by default:
```
+--------------------+------+
|              images|labels|
+--------------------+------+
|[0.33967614, 0.87...|     2|
|[0.266905, 0.9730...|     3|
|[0.66661334, 0.67...|     1|
|[0.91943026, 0.89...|     6|
|[0.014844197, 0.0...|     6|
|[0.5366513, 0.748...|     4|
|[0.055148937, 0.7...|     7|
|[0.7814437, 0.228...|     4|
|[0.31193638, 0.55...|     3|
|[0.20034336, 0.24...|     4|
|[0.09441255, 0.07...|     3|
|[0.5177533, 0.397...|     0|
|[0.23703437, 0.44...|     9|
|[0.9425567, 0.859...|     9|
|[0.017339867, 0.0...|     3|
|[0.827106, 0.3122...|     0|
|[0.8641392, 0.194...|     2|
|[0.95585227, 0.29...|     3|
|[0.7508129, 0.464...|     4|
|[0.035597708, 0.3...|     9|
+--------------------+------+
only showing top 20 rows
```


### The interconversion with TFRecord dataset
we can convert TFRecord to OFRecord with the following command：

```python
reader = spark.read.format("tfrecords")
dataframe = reader.load("file:///path/to/tfrecord_file")
writer = dataframe.write.format("ofrecord")
writer.save("file:///path/to/outputdir")
```
In the above code, the `outputdir` directory will be created automatically, we will save ofrecord file in this directory.Make sure the "outputdir" directory does not exist before excuting the command.

In addition, we can use the following command to split the data into multiple ofrecord file in conversion.
```python
reader = spark.read.format("tfrecords")
dataframe = reader.load("file:///path/to/tfrecord_file")
writer = dataframe.repartition(10).write.format("ofrecord")
writer.save("file://path/to/outputdir")
```
After the above command is executed, 10 ofrecord files of `part-xxx` format will be generated in "outputdir" directory.

The process of converting OFRecord file to TFRecord file is similar. we just need to swap read/write side `format`:
```python
reader = spark.read.format("ofrecord")
dataframe = reader.load("file:///path/to/ofrecord_file")
writer = dataframe.write.format("tfrecords")
writer.save("file:///path/to/outputdir")
```

### The interconversion with JSON format
we can convert JSON to OFRecord with the following command：
```python
dataframe = spark.read.json("file:///path/to/json_file")
writer = dataframe.write.format("ofrecord")
writer.save("file:///path/to/outputdir")
```

The following command will convert OFRecord data to JSON file：
```python
reader = spark.read.format("ofrecord")
dataframe = reader.load("file:///path/to/ofrecord_file")
dataframe.write.json("file://path/to/outputdir")
```

### Other script and tools
除了以上介绍的方法，使得其它数据格式与 OFRecord 数据格式进行转化外，在[oneflow_toolkit](https://github.com/Oneflow-Inc/oneflow_toolkit/tree/master/ofrecord)仓库下，还有各种与 OFRecord有关的脚本：

* 为 BERT 模型准备 OFRecord 数据集的脚本

* 下载 ImageNet 数据集并转 OFRecord 数据集的工具

* 下载 MNIST 数据集并转 OFRecord 数据集的工具

* 将微软 Coco 转 OFRecord 数据集的工具

