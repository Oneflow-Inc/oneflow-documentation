深度学习应用需要复杂的多阶段数据预处理流水线，数据加载是流水线的第一步，OneFlow支持多种格式数据的加载，其中`OFRecord`格式是OneFlow原生的数据格式。

`OFRecord`的格式定义参考了TensorFlow的[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)，熟悉`TFRecord`的用户，可以很快上手OneFlow的`OFRecord`。

本文将介绍：

* OFRecord使用的数据类型

* 如何将数据转化为OFRecord对象并序列化

* OFRecord文件的写入

* OFRecord文件的读取

文末附有写入/读取OFRecord文件的完整代码。

掌握它们后，有助于我们学习 **制作与导入OFRecord数据集**。

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

* OFRecord、Feature、XXXList等类型，均有`Protocol Buffers`生成对应的同名接口，使得我们可以在Python层面构造相关对象。

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

在下例子中，我们将创建有4个feature的OFRecord对象，并且调用它的`SerializeToString`方法序列化。

```python
feature0 = [True, True, False, False, True]
feature1 = [random.randint(1, 100) for x in range(0,5)]
feature2 = [b'cat', b'dog', b'chicken', b'horse', b'goat']
feature3 = [random.random() for x in range(0, 5)]

topack = {
      'feature0': int64_feature(feature0),
      'feature1': int64_feature(feature1),
      'feature2': bytes_feature(feature2),
      'feature3': float_feature(feature3),
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
OneFlow约定，采用以下格式存储OFRecord对象的序列化结果：

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

## OFRecord格式的文件的读取
我们已经知道，OneFlow的OFRecord文件的格式为头8字节为数据长度，之后为OFRecord对象序列化后的数据。因此，我们可以按照该约定，从`OFRecord`文件中加载数据：

```python
with open("example.ofrecords", "rb") as f:
    length = struct.unpack("q", f.read(8))
    serilizedBytes = f.read(length[0])
    ofrecord_features = ofrecord.OFRecord.FromString(serilizedBytes)
    f0 = ofrecord_features.feature["feature0"].int64_list.value
    f1 = ofrecord_features.feature["feature1"].int64_list.value
    f2 = ofrecord_features.feature["feature2"].bytes_list.value
    f3 = ofrecord_features.feature["feature3"].float_list.value

    print(f0)
    print(f1)
    print(f2)
    print(f3)
```
以上，先通过`FromString`反序列化得到`OFRecord`类型的对象`ofrecord_features`。
然后依次取出`OFRecord`中的各个feature。具体请参阅[Protocol Buffers](https://developers.google.com/protocol-buffers/)及`oneflow/core/record/record.proto`。

## 完整代码

### 序列化数据并存文件
本例代码，展示了如何将需要保存的数据转为`OFRecord`对象，并且序列化后存文件。

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

feature0 = [True, True, False, False, True]
feature1 = [random.randint(1, 100) for x in range(0,5)]
feature2 = [b'cat', b'dog', b'chicken', b'horse', b'goat']
feature3 = [random.random() for x in range(0, 5)]

topack = {
      'feature0': int64_feature(feature0),
      'feature1': int64_feature(feature1),
      'feature2': bytes_feature(feature2),
      'feature3': float_feature(feature3),
}

ofrecord_features = ofrecord.OFRecord(feature=topack)

serilizedBytes = ofrecord_features.SerializeToString()

f = open("example.ofrecords", "wb")
length = ofrecord_features.ByteSize()
f.write(struct.pack("q", length))
f.write(serilizedBytes)
f.close()
```

### 从ofrecord文件中读取数据并反序列化
以下代码与以上序列化OFrecord对象并写文件对应。
我们读取文件内容，并反序列化得到f0~f3。

```python
import oneflow.core.record.record_pb2 as ofrecord
import six
import random
import struct

with open("example.ofrecords", "rb") as f:
    length = struct.unpack("q", f.read(8))
    serilizedBytes = f.read(length[0])
    ofrecord_features = ofrecord.OFRecord.FromString(serilizedBytes)
    f0 = ofrecord_features.feature["feature0"].int64_list.value
    f1 = ofrecord_features.feature["feature1"].int64_list.value
    f2 = ofrecord_features.feature["feature2"].bytes_list.value
    f3 = ofrecord_features.feature["feature3"].float_list.value

    print(f0)
    print(f1)
    print(f2)
    print(f3)
```
