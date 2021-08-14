在[数据输入](../basics_topics/data_input.md)一文中我们知道了使用 DataLoader 及相关算子加载数据，往往效率更高，并且学习了如何使用 DataLoader 及相关算子。

在 [OFrecord 数据格式](ofrecord.md)中，我们学习了 OFRecord 文件的存储格式。

本文，将围绕 OneFlow 的 OFRecord 数据集的加载与制作展开，主要包括：

* OFRecord 数据集的组织形式

* 加载 OFRecord 数据集的多种方式

* OFRecord 数据集与其它数据格式的相互转化

## 什么是OFRecord数据集
在 [OFrecord 数据格式](ofrecord.md)中我们已经介绍过 `OFRecord 文件` 的存储格式，知道了什么是 `OFRecord文件`。

OFRecord 数据集是 **OFRecord 文件的集合** 。将多个 `OFRecord文件`，按照 OneFlow 约定的文件名格式，存放在同一个目录中，就得到了 OFRecord 数据集。

默认情况下，OFRecord 数据集目录中的文件，统一以 `part-xxx` 的方式命名，其中的 "xxx" 是从0开始的文件编号，有补齐和不补齐两种选择。

以下是没有采用补齐的命名风格示例：
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

以下是有补齐的命名风格：
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
OneFlow 采用此约定，与 `spark` 的默认存储的文件名一致，方便使用 spark 制作与转化 OFRecord 数据。

实际上，文件名前缀(`part-`)、文件名编号是否补齐、按多少位补齐，均可以自行指定，只需要在加载数据集(下文会介绍)时，保持相关参数一致即可。

OneFlow 提供了加载 OFRecord 数据集的接口，使得我们只要指定数据集目录的路径，就可以享受 OneFlow 框架所带来的多线程、数据流水线等优势。

## 加载OFRecord数据集的方法
我们使用 `ofrecord_reader` 加载并预处理数据集。

在[数据输入](../basics_topics/data_input.md)一文中，我们已经展示了如何使用 `ofrecord_reader` 接口加载 OFRecord 数据，并进行数据预处理。

代码见：[of_data_pipeline.py](../code/basics_topics/of_data_pipeline.py)

`ofrecord_reader` 的接口如下：
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

* `ofrecord_dir` 指定存放数据集的目录路径

* `batch_size` 指定每轮读取的 batch 大小

* `data_part_num` 指定数据集目录中一共有多少个 ofrecord 格式的文件，如果这个数字大于真实存在的文件数，会报错

* `part_name_prefix` 指定 ofrecord 文件的文件名前缀， OneFlow 根据前缀+序号在数据集目录中定位 ofrecord 文件

* `part_name_suffix_length` 指定 ofrecord 文件的序号的对齐长度，-1表示不用对齐

* `random_shuffle` 表示读取时是否需要随机打乱样本顺序

* `shuffle_buffer_size` 指定了读取样本的缓冲区大小

* `shuffle_after_epoch` 表示每轮读取完后是否需要重新打乱样本顺序

使用 `ofrecord_reader` 的好处在于， `ofrecord_reader` 作为一个普通算子，参与 OneFlow 构图优化，并享有 OneFlow 流水线加速。

对于与业务逻辑耦合的特定操作（如解码、解压等），我们还可以为 `ofrecord_reader` 定义预处理 op，让程序拥有很高的灵活性和扩展性。

- 关于 DataLoader 及相关算子使用可以参考[数据输入](../basics_topics/data_input.md#dataloader)

## 其它格式数据与 OFRecord 数据集的相互转化
参考[OFrecord数据格式](ofrecord.md)中 OFRecord 文件的存储格式及本文开头介绍的 OFRecord 数据集的文件名格式约定，我们完全可以自己制作 OFRecord 数据集。

不过为了更加方便，我们提供了 Spark 的 jar 包，方便 OFRecord 与常见数据格式(如 TFRecord、json)进行相互转化。

### spark 的安装与启动
首先，下载 spark 及 spark-oneflow-connector：

* 在 spark 官网下载[spark-2.4.7-bin-hadoop2.7.tgz](https://archive.apache.org/dist/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz)

* 在[这里](https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/spark-oneflow-connector-assembly-0.1.0_int64.jar)下载 jar 包，spark 需要它来支持 ofrecord 格式

接着，解压 `spark-2.4.7-bin-hadoop2.7.tgz`，并配置环境变量 `SPARK_HOME`:
```
export SPARK_HOME=path/to/spark-2.4.7-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$PATH
```

然后，通过以下命令启动 pyspark shell：
```
pyspark --master "local[*]"\
 --jars spark-oneflow-connector-assembly-0.1.0_int64.jar\
 --packages org.tensorflow:spark-tensorflow-connector_2.11:1.13.1
```

```text
...
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.4.7
      /_/

Using Python version 3.6.10 (default, Mar 25 2020 18:53:43)
SparkSession available as 'spark'.
>>>
```

在启动的 pyspark shell 中，我们可以完成 OFRecord 数据集与其它数据格式的相互转化。

### 使用 spark 查看 OFRecord 数据集
使用以下命令可以查看 OFRecord 数据：
```
spark.read.format("ofrecord").load("file:///path/to/ofrecord_file").show()
```
默认显示前20条数据:
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


### 与 TFRecord 数据集的相互转化
以下命令可以将 TFRecord 转化为 OFRecrod：

```python
reader = spark.read.format("tfrecords")
dataframe = reader.load("file:///path/to/tfrecord_file")
writer = dataframe.write.format("ofrecord")
writer.save("file:///path/to/outputdir")
```
以上代码中的 `outputdir` 目录会被自动创建，并在其中保存 ofrecord 文件。在执行命令前应保证 outputdir 目录不存在。

此外，还可以使用以下命令，在转化的同时，将数据切分为多个 ofrecord 文件：
```python
reader = spark.read.format("tfrecords")
dataframe = reader.load("file:///path/to/tfrecord_file")
writer = dataframe.repartition(10).write.format("ofrecord")
writer.save("file://path/to/outputdir")
```
以上命令执行后，在 outputdir 目录下会产生10个 `part-xxx` 格式的ofrecord文件。

将 OFRecord 文件转为 TFRecord 文件的过程类似，交换读/写方的 `format` 即可：
```python
reader = spark.read.format("ofrecord")
dataframe = reader.load("file:///path/to/ofrecord_file")
writer = dataframe.write.format("tfrecords")
writer.save("file:///path/to/outputdir")
```

### 与 JSON 格式的相互转化
以下命令可以将 JSON 格式数据转为 OFRecord 数据集:
```python
dataframe = spark.read.json("file:///path/to/json_file")
writer = dataframe.write.format("ofrecord")
writer.save("file:///path/to/outputdir")
```

以下命令将 OFRecord 数据转为 JSON 文件：
```python
reader = spark.read.format("ofrecord")
dataframe = reader.load("file:///path/to/ofrecord_file")
dataframe.write.json("file://path/to/outputdir")
```
