深度学习应用需要复杂的多阶段数据预处理流水线，数据加载是流水线的第一步，OneFlow支持多种格式数据的加载，其中`OFRecord`格式是OneFlow原生的数据格式。

`OFRecord`的格式定义参考了TensorFlow的[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)，熟悉`TFRecord`的用户，可以很快上手OneFlow的`OFRecord`。

本文将介绍：

* OFRecord使用的数据类型

* 如何将数据转化为OFRecord对象并序列化

* OFRecord文件格式

掌握它们后，有助于我们学习[制作与导入OFRecord数据集](making_load_ofdataset.md)。

## OFRecord相关数据类型
OneFlow内部采用[Protocol Buffers](https://developers.google.com/protocol-buffers/)描述`OFRecord`的序列化格式。相关的`.proto`文件在`oneflow/core/record/record.proto`中，具体定义如下：

```
syntax = "proto2";
package oneflow;

message BytesList {
  repeated bytes value = 1;
}

message FloatList {
  repeated float value = 1 [packed = true];
}

message DoubleList {
  repeated double value = 1 [packed = true];
}

message Int32List {
  repeated int32 value = 1 [packed = true];
}

message Int64List {
  repeated int64 value = 1 [packed = true];
}

message Feature {
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    DoubleList double_list = 3;
    Int32List int32_list = 4;
    Int64List int64_list = 5;
  }
}

message OFRecord {
  map<string, Feature> feature = 1;
}
```

我们先对以上的重要数据类型进行解释：

* OFRecord: OFRecord的实例化对象，可用于存储所有需要序列化的数据。它由任意多个string->Feature的键值对组成；

* Feature: Feature可存储BytesList、FloatList、DoubleList、Int32List、Int64List各类型中的任意一种；

* OFRecord、Feature、XXXList等类型，均由`Protocol Buffers`生成对应的同名接口，使得我们可以在Python层面构造对应对象。

## 转化数据为Feature格式

我们可以通过调用`ofrecord.xxxList`及`ofrecord.Feature`将数据转为`Feature`格式，但是为了更加方便，我们需要对`protocol buffers`生成的接口进行简单封装：

```python
import oneflow.core.record.record_pb2 as ofrecord

def int32_feature(value):
    if not isinstance(value, (list, tuple)):
      value = [value]
    return ofrecord.Feature(int32_list=ofrecord.Int32List(value=value))


def int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int64_list=ofrecord.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, (list, tuple)):
      value = [value]
    return ofrecord.Feature(float_list=ofrecord.FloatList(value=value))


def double_feature(value):
    if not isinstance(value, (list, tuple)):
      value = [value]
    return ofrecord.Feature(double_list=ofrecord.DoubleList(value=value))


def bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not six.PY2:
        if isinstance(value[0], str):
            value = [x.encode() for x in value]
    return ofrecord.Feature(bytes_list=ofrecord.BytesList(value=value))
```

## 创建OFRecord对象并序列化

在下例子中，我们将创建有2个feature的OFRecord对象，并且调用它的`SerializeToString`方法序列化。

```python
    obserations = 28*28
    #...
    image = [random.random() for x in range(0,obserations)]
    label = [random.randint(0,9)]

    topack = {
        'images': float_feature(image),
        'labels': int64_feature(label),
    }

    ofrecord_features = ofrecord.OFRecord(feature=topack)
    serilizedBytes = ofrecord_features.SerializeToString()
```

通过以上例子，我们可以总结序列化数据的步骤：

* 将需要序列化的数据，通过调用`ofrecord.Feature`及`ofrecord.XXXList`转为`Feature`对象；

* 将上一步得到的各个Feature对象，以`string->Feature`键值对的形式，存放在Python字典中；

* 调用`ofrecord.OFRecord`创建`OFRecord`对象

* 调用OFRecord对象的`SerializeToString`方法得到序列化结果

序列化的结果，可以存为ofrecord格式的文件。

## OFRecord格式的文件

将OFRecord对象序列化后按OneFlow约定的格式存文件，就得到`OFRecord文件`。

1个OFRecord文件中可存储多个OFRecord对象，OFRecord文件可用于`OneFlow数据流水线`，具体操作可见[制作与导入OFRecord数据集](making_load_ofdataset.md)

OneFlow约定，对于 **每个** OFRecord对象，用以下格式存储：

```
uint64 length
byte   data[length]
```

即头8个字节存入数据长度，然后存入序列化数据本身。

```python
length = ofrecord_features.ByteSize()
f.write(struct.pack("q", length))
f.write(serilizedBytes)
```

## 完整代码
以下完整代码展示如何生成OFRecord文件，并调用`protobuf`生成的`OFRecord`接口手工读取OFRecord文件中的数据。

实际上，OneFlow提供了`flow.data.decode_ofrecord`等接口，可以更方便地提取OFRecord文件（数据集）中的内容。详细内容请参见[制作与导入OFRecord数据集](making_load_ofdataset.md)。

### 将OFRecord对象写入文件
以下代码，模拟了3个样本，每个样本为`28*28`的图片，并且包含对应标签。将三个样本转化为OFRecord对象后，按照OneFlow约定格式，存入文件。

```python
import oneflow.core.record.record_pb2 as ofrecord
import six
import random
import struct

def int32_feature(value):
    if not isinstance(value, (list, tuple)):
      value = [value]
    return ofrecord.Feature(int32_list=ofrecord.Int32List(value=value))


def int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int64_list=ofrecord.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, (list, tuple)):
      value = [value]
    return ofrecord.Feature(float_list=ofrecord.FloatList(value=value))


def double_feature(value):
    if not isinstance(value, (list, tuple)):
      value = [value]
    return ofrecord.Feature(double_list=ofrecord.DoubleList(value=value))


def bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not six.PY2:
        if isinstance(value[0], str):
            value = [x.encode() for x in value]
    return ofrecord.Feature(bytes_list=ofrecord.BytesList(value=value))

obserations = 28*28

f = open("./dataset/part-0", "wb")

for loop in range(0, 3):
    image = [random.random() for x in range(0,obserations)]
    label = [random.randint(0,9)]

    topack = {
        'images': float_feature(image),
        'labels': int64_feature(label),
    }

    ofrecord_features = ofrecord.OFRecord(feature=topack)
    serilizedBytes = ofrecord_features.SerializeToString()

    length = ofrecord_features.ByteSize()

    f.write(struct.pack("q", length))
    f.write(serilizedBytes)

f.close()
```

### 从OFRecord文件中读取数据
以下代码，读取上例中生成的`OFRecord`文件，调用`FromString`方法反序列化得到`OFRecord`对象，并最终显示数据：
```python
import oneflow.core.record.record_pb2 as ofrecord
import struct

with open("./dataset/part-0", "rb") as f:
    for loop in range(0,3):
        length = struct.unpack("q", f.read(8))
        serilizedBytes = f.read(length[0])
        ofrecord_features = ofrecord.OFRecord.FromString(serilizedBytes)
        
        image = ofrecord_features.feature["images"].float_list.value
        label = ofrecord_features.feature["labels"].int64_list.value

        print(image, label, end="\n\n")
```



