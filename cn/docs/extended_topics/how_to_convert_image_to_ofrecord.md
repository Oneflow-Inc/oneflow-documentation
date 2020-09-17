# 制作OFRecord数据集

在 [OFrecord数据格式](./ofrecord.md) 和 [加载与准备 OFRecord 数据集](./how_to_make_ofdataset.md) 中，我们分别学习了 OFRecord 文件的存储格式以及如何加载 OFRecord 数据集。

本文，我们将围绕 OneFlow 的 OFRecord 数据集的编解码方式与制作展开，包括：

- OFRecord Reader 的解码方式
- OFRecord 的编码流程
- 制作基于 Mnist 手写数字数据集的 OFRecord 数据集

## OFRecord编解码方式

OneFlow 内部的解码算子是采用 [OpenCV](https://opencv.org/) 来对数据进行解码的。相关的 `.cpp` 文件在 [oneflow.user.kernels.ofrecord_decoder_kernels.cpp](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/kernels/ofrecord_decoder_kernels.cpp) 

代码中的 `Feature` 就是在 [OFrecord数据格式](./ofrecord.md) 一章中提到的 OFRecord 内部用 [Protocol Buffers](https://developers.google.com/protocol-buffers/) 定义的序列化格式。

整体解码流程概括如下：

- 根据对应的 `name` 读取 OFRecord 内对应的 `Feature`

- 然后读取 `Feature` 内的 `BytesList` 类型数据，并进行一系列**合法性检查** 
- 通过调用 OpenCV 库的 `imdecode` 方法对读取到的**字节流数据**进行解码，**转换成原始图片数据**

- 对图片进行对应的后处理



## 将图片数据转化为OFRecord

了解了 OFRecord 的解码流程后，我们可以对**整个流程进行反推**，从而对图片数据进行编码转化为 OFRecord 数据集。

- 调用 `imencode` 将原始图片数据编码成**字节流数据**，并进行序列化
- 转换成 OFRecord 的 `Feature`，并进行序列化

下面我们看两段具体的代码

首先是我们对读取进来的图片数据进行编码 

```python
def encode_img_file(filename, ext=".jpg"):
    img = cv2.imread(filename)
    encoded_data = cv2.imencode(ext, img)[1]
    return encoded_data.tostring()
```

然后转化成 `Feature` 的形式，并进行序列化，写入到文件中。

注意，每次写入前需要将**Feature的数据长度**也给写入。

```python
def ndarry2ofrecords(dsfile, dataname, encoded_data, labelname, encoded_label):
    topack = {dataname: bytes_feature(encoded_data), 
              labelname: int32_feature(encoded_label)}
    ofrecord_features = ofrecord.OFRecord(feature=topack)
    serilizedBytes = ofrecord_features.SerializeToString()
    length = ofrecord_features.ByteSize()
    dsfile.write(struct.pack("q", length))
    dsfile.write(serilizedBytes)
```

### 代码解析

- 先用 `bytes_feature` 将图片转化为 `Feature` 格式
- 调用 `ofrecord.OFRecord` 以及 `SerializeToString` 进行序列化操作
- 调用 `ofrecord_features.ByteSize()` 获取数据长度
- 将数据长度 `length` 以及数据 `serilizedBytes` 写入到文件中

另外我们建议将数据写进**多个 part 文件**中， OneFlow Reader 读取的时候会使用**多线程加速**，当数据存储在多个文件时，**读取效率会大大提升**。

## 完整代码

### 制作基于 Mnist 手写数字数据集的 OFRecord 文件

我们使用**Mnist手写数字数据集**来完整制作一个OFRecord格式文件，Mnist 数据集下载地址为 [Mnist数据集](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)

下载至 `img_to_ofrecord/mnist` 目录下并解压后，整个代码目录构造如下 

```
img_to_ofrecord
├── dataset
├── mnist
	├── train_set
	├── test_set
	├── train_label
		├── label.txt
├── img2ofrecord.py
├── lenet_train.py
```

其中 `mnist` 目录存放原始 Mnist 数据集以及标签文件 `label.txt` ，`dataset` 目录将用于存放制作的 OFRecord 文件，而 `img2ofrecord.py` 是将手写数字数据集转换成 OFRecord 格式文件的脚本，`lenet_train.py` 则是读取我们制作好的 OFRecord 数据集，使用 LeNet 模型进行训练。 

完整代码：[img2ofrecord.py](NULL)

```python
# img2ofrecord.py
import cv2
import oneflow.core.record.record_pb2 as ofrecord
import six
import struct
import os
import argparse


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


def encode_img_file(filename, ext=".jpg"):
    img = cv2.imread(filename)
    encoded_data = cv2.imencode(ext, img)[1]
    return encoded_data.tostring()


def ndarray2ofrecords(dsfile, dataname, encoded_data, labelname, encoded_label):
    topack = {dataname: bytes_feature(encoded_data),
              labelname: int32_feature(encoded_label)}
    ofrecord_features = ofrecord.OFRecord(feature=topack)
    serilizedBytes = ofrecord_features.SerializeToString()
    length = ofrecord_features.ByteSize()
    dsfile.write(struct.pack("q", length))
    dsfile.write(serilizedBytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_root',
        type=str,
        default='./mnist/train_set',
        help='the directory of images')
    parser.add_argument(
        '--part_num',
        type=int,
        default='6',
        help='the amount of OFRecord data part')
    parser.add_argument(
        '--label_dir',
        type=str,
        default='./mnist/train_label/label.txt',
        help='the directory of labels')
    parser.add_argument(
        '--img_format',
        type=str,
        default='.png',
        help='the encode format of images')
    args = parser.parse_args()
    imgs_root = args.image_root
    part_num = args.part_num
    label_dir = args.label_dir
    img_format = args.img_format

    print("The image root is: ", imgs_root)
    print("The amount of OFRecord data part is: ", part_num)
    print("The directory of Labels is: ", label_dir)
    print("The image format is: ", img_format)
    print("Start Processing......")

    part_cnt = 0
    file_cnt = 0
    # Read the labels
    with open(label_dir, 'r') as label_file:
        labels = label_file.readlines()

    imgfilenames = os.listdir(imgs_root)
    file_total_cnt = len(imgfilenames)

    for i, file in enumerate(imgfilenames):
        ofrecord_filename = r"./dataset/part-{}".format(part_cnt)
        label = int(labels[i].strip('\n'))  # delete the '\n' in labels
        with open(ofrecord_filename, 'ab') as f:
            imgfile = os.path.join(imgs_root, file)
            encoded_data = encode_img_file(imgfile, img_format)
            ndarray2ofrecords(f, "images", encoded_data, "labels", label)
            # print("{} feature saved".format(imgfile))
            file_cnt += 1
            if file_cnt == file_total_cnt // part_num:
                file_cnt = 0
                part_cnt += 1

    print("Process image successfully !!!")
```

- 我们读取6万张训练图片，并分别调用 `encode_img_file`, `imgfile2label`, `ndarray2ofrecords`，来完成图像，标签的编码，并将数据写入到文件中。
- 我们通过命令行参数 `image_root`，`part_num`，`label_dir` 可以分别指定图片路径，数据切分个数，标签路径。

我们运行该脚本，并指定数据切分成10个分段，输出如下

```shell
$ python img2ofrecord.py --part_num=10 --img_format=.png
The image root is:  ./mnist/train_set
The amount of OFRecord data part is:  10
The directory of Labels is:  ./mnist/train_label/label.txt
The image format is:  .png
Start Processing......
......
./mnist/train_set/00058991_1.png feature saved
./mnist/train_set/00058992_4.png feature saved
./mnist/train_set/00058993_8.png feature saved
./mnist/train_set/00058994_4.png feature saved
./mnist/train_set/00058995_1.png feature saved
......
Process image successfully !!!
```

### 使用自制的 OFRecord 数据集进行训练

我们运行目录下的 [lenet_train.py](./img_to_ofrecord/lenet_train.py)，它将读取我们刚制作好的 OFRecord 数据集，在 Lenet 模型上进行训练

该训练脚本输出如下

```
[5.24852]
[0.13135958]
[0.0759443]
[0.07838672]
[0.07410873]
[0.03932165]
......
```

至此，我们成功完成了数据集制作，读取，训练整个流程了。

**赶快动手制作你的 OFRecord 数据集并用 OneFlow 进行训练吧！**

