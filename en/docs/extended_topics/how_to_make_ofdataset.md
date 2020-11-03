
In [data input] (... /basics_topics/data_input.md) we learned that it is usually more efficient to load data using DataLoader and related operators. We also learned how to use DataLoader and related operators.

In article [OFRecord](ofrecord.md), we learn about the storage format of OFRecord files.

In this article, we will focus on the loading and generating of OneFlow's OFRecord dataset, which mainly includes:

* The hierarchy of OFRecord dataset

* Multiple ways of loading OFRecord dataset

* The transition between OFRecord dataset and other data formats

## What is OFRecord Dataset

In article [OFRecord](ofrecord.md), we introduce what `OFRecord file ` is and the storage format of `OFRecord file`.

OFRecord dataset is **the collection of OFRecord files**. The collection of mutiple files that named by OneFlow convention, and that stored in the same directory, is an OFRecord dataset.

By default, The files in OFRecord dataset directory are uniformly named in the way of `part-xxx`, where "xxx" is the file id starting from zero, and there can be choices about padding or non-padding.

These are the examples of using non-padding name style:

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

These are the examples of using padding name style:

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

OneFlow adopts this convention, which is consistent with the default storage filename in `spark`, so it is convenient to prepare OFRecord data by spark.

Actually, we can specify the filename prefix `part-`, whether we pad the filename id and how many bits to pad. We just need to keep the same parameters when loading dataset, which will be described below.

OneFlow provides the API interface to load OFRecord dataset by specifying the path of dataset directory, so that we can have the multi-threading, pipelining and some other advantages brought by OneFlow framework.

## The Method to Load OFRecord Dataset

We use `ofrecord_reader` to load and preprocess dataset. 

In article [Data Input](../basics_topics/data_input.md), we show how to use `ofrecord_reader` API to load OFRecord data and preprocess it. 

Code: [of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

The prototype of `ofrecord_reader` is as follows：

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

- `ofrecord_dir` is the directory which stored the dataset
- `batchsize` assign the batch size in each epoch
- `data_part_num` assign the number of ofrecord data format file in the directory which stored the dataset. It will raise an error if the parameter is greater than the number of the existed files
- `part_name_prefix` assign the filename prefix of ofrecord files. Oneflow locates the ofrecord files according to the prefix + index in the dataset directory
- `part_name_suffix_length` assigns the padding of ofrecord file index, -1 represents no padding
- `random_shuffle` assign whether shuffle the sample order randomly when reading data
- `shuffle_buffer_size` assign the buffer size when reading data
- `shuffle_after_epoch` assign whether shuffle the sample order after each epoch

The benefit of using `ofrecord_reader` is that `ofrecord_reader` acts as a normal operator which participates in OneFlow composition optimization and enjoys OneFlow flowline acceleration.
For flexibility and extensibility of the code, we can define a preprocessing OP for `ofrecord_reader` to deal with specific data formats which are coupled with operational logic (e.g. decoding, decompression and etc.).
- For more information on DataLoader and related operator usage refer to [Data input](../basics_topics/data_input.md) .
-  For more information on customized OP refer to [User op](./user_op.md).

## The transition between other data format data and OFRecord dataset

According to the storage format of OFRecord file in article [OFRecord](ofrecord.md) and the filename format convention of OFRecord dataset introduced at the beginning, we can prepare OFRecord dataset by ourselves.

To prepare dataset easier, we provide jar package from Spark, which is convenient to the interconversion between OFRecord and common data formats (such as TFRecord and JSON).

### The installation and launch of Spark

At first, we should download Spark and Spark-oneflow-connector：

* Download the [spark-2.4.0-bin-hadoop2.7](https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz) from the official website of Spark

* Download jar package [here](https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/spark-oneflow-connector-assembly-0.1.0_int64.jar), which is needed by Spark to support the ofrecord file format

Then unzip the `spark-2.4.0-bin-hadoop2.7.tgz` and configure the environment variable `SPARK_HOME`:

```shell
export SPARK_HOME=path/to/spark-2.4.0-bin-hadoop2.7
```

We can launch the pyspark shell with the following command：

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

We can complete the data conversion between OFRecord dataset and other formats in pyspark shell.

### Use Spark to view OFRecord dataset

We can view OFRecord data with following code：

```
spark.read.format("ofrecord").load("file:///path/to/ofrecord_file").show()
```

The first 20 rows are displayed by default:

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

In the above code, the `outputdir` directory will be created automatically, and ofrecord files will be saved into this directory. Make sure that the "outputdir" directory does not exist before executing the command.

In addition, we can use the following command to split data into multiple ofrecord files.

```python
reader = spark.read.format("tfrecords")
dataframe = reader.load("file:///path/to/tfrecord_file")
writer = dataframe.repartition(10).write.format("ofrecord")
writer.save("file://path/to/outputdir")
```

After executing the above commands, 10 ofrecord files of `part-xxx` format will be generated in "outputdir" directory.

The process of converting OFRecord file to TFRecord file is similar. we just need to change the `format` of read/write side:

```python
reader = spark.read.format("ofrecord")
dataframe = reader.load("file:///path/to/ofrecord_file")
writer = dataframe.write.format("tfrecords")
writer.save("file:///path/to/outputdir")
```

### The interconversion with JSON format

We can convert JSON to OFRecord with the following command：

```python
dataframe = spark.read.json("file:///path/to/json_file")
writer = dataframe.write.format("ofrecord")
writer.save("file:///path/to/outputdir")
```

The following command will convert OFRecord data to JSON files：

```python
reader = spark.read.format("ofrecord")
dataframe = reader.load("file:///path/to/ofrecord_file")
dataframe.write.json("file://path/to/outputdir")
```

