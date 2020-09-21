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

- 然后读取 `Feature` 内的 `BytesList` 类型数据，并进行一系列 **合法性检查** 
- 通过调用 OpenCV 库的 `imdecode` 方法对读取到的 **字节流数据** 进行解码， **转换成原始图片数据** 

- 对图片进行对应的后处理



## 将图片数据转化为OFRecord

了解了 OFRecord 的解码流程后，我们可以对 **整个流程进行反推** ，从而对图片数据进行编码转化为 OFRecord 数据集。

- 调用 `imencode` 将原始图片数据编码成 **字节流数据** ，并进行序列化
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

注意，每次写入前需要将 **Feature的数据长度** 也给写入。

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

另外我们建议将数据写进 **多个 part 文件** 中， OneFlow Reader 读取的时候会使用 **多线程加速** ，当数据存储在多个文件时， **读取效率会大大提升** 。

## 完整代码

### 制作基于 MNIST 手写数字数据集的 OFRecord 文件

我们使用 **MNIST手写数字数据集** 来完整制作一个OFRecord格式文件（这里我们仅取50张图片作为示例)，MNIST 示例数据集以及标签文件的下载地址为 [MNIST数据集](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)

下载至 `img_to_ofrecord/images` 目录下并解压后，整个代码目录构造如下 

```
img_to_ofrecord
├── images
	├── train_set
	├── train_label
		├── label.txt
├── img2ofrecord.py
├── lenet_train.py
```

其中 `images` 目录存放原始示例训练数据集以及标签文件 `label.txt` ，而 `img2ofrecord.py` 是将手写数字数据集转换成 OFRecord 格式文件的脚本，`lenet_train.py` 则是读取我们制作好的 OFRecord 数据集，使用 LeNet 模型进行训练。 

完整代码：[img2ofrecord.py](../code/extended_topics/img_to_ofrecord/img2ofrecord.py)

```python
# img2ofrecord.py
import cv2
import oneflow.core.record.record_pb2 as ofrecord
import six
import struct
import os
import argparse
import json

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_root',
        type=str,
        default='./images/train_set',
        help='The directory of images')
    parser.add_argument(
        '--part_num',
        type=int,
        default='5',
        help='The amount of OFRecord partitions')
    parser.add_argument(
        '--label_dir',
        type=str,
        default='./images/train_label/label.txt',
        help='The directory of labels')
    parser.add_argument(
        '--img_format',
        type=str,
        default='.png',
        help='The encode format of images')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./dataset/',
        help='The save directory of OFRecord patitions')
    args = parser.parse_args()
    return args 


def printConfig(imgs_root, part_num, label_dir, img_format, save_dir): 
    print("The image root is: ", imgs_root)
    print("The amount of OFRecord data part is: ", part_num)
    print("The directory of Labels is: ", label_dir)
    print("The image format is: ", img_format)
    print("The OFRecord save directory is: ", save_dir)
    print("Start Processing......")

if __name__ == "__main__":
    args = parse_args()
    imgs_root = args.image_root
    part_num = args.part_num
    label_dir = args.label_dir
    img_format = args.img_format
    save_dir = args.save_dir

    os.mkdir(save_dir) # Make Save Directory
    printConfig(imgs_root, part_num, label_dir, img_format, save_dir)

    part_cnt = 0
    # Read the labels
    with open(label_dir, 'r') as label_file:
        imgs_labels = label_file.readlines()

    file_total_cnt = len(imgs_labels)
    assert file_total_cnt > part_num, "The amount of Files should be larger than part_num"
    per_part_amount = file_total_cnt // part_num

    for cnt, img_label in enumerate(imgs_labels):
        if cnt !=0 and cnt % per_part_amount == 0: 
            part_cnt += 1
        prefix_filename = os.path.join(save_dir, "part-{}")
        ofrecord_filename = prefix_filename.format(part_cnt)
        with open(ofrecord_filename, 'ab') as f:
            data = json.loads(img_label.strip('\n'))
            for img, label in data.items():
                encoded_data = encode_img_file(img, img_format)
                ndarray2ofrecords(f, "images", encoded_data, "labels", label)
                print("{} feature saved".format(img))

    print("Process image successfully !!!")
```

- 我们读取50张示例训练图片，并分别调用 `encode_img_file`, `imgfile2label`, `ndarray2ofrecords`，来完成图像，标签的编码，并将数据写入到文件中。
- 我们通过命令行参数 `image_root`，`part_num`，`label_dir` 可以分别指定图片路径，数据切分个数，标签路径。

我们运行该脚本，并指定数据切分成5个分段，输出如下

```shell
$ python img2ofrecord.py --part_num=5 --save_dir=./dataset/ --img_format=.png
The image root is:  ./images/train_set
The amount of OFRecord data part is:  5
The directory of Labels is:  ./images/train_label/label.txt
The image format is:  .png
The OFRecord save directory is:  ./dataset/
Start Processing......
./images/train_set/00000000_5.png feature saved
./images/train_set/00000001_0.png feature saved
./images/train_set/00000002_4.png feature saved
./images/train_set/00000003_1.png feature saved
.......
Process image successfully !!!
```

### 使用自制的 OFRecord 数据集进行训练

我们运行目录下的 [lenet_train.py](../code/extended_topics/img_to_ofrecord/lenet_train.py)，它将读取我们刚制作好的 OFRecord 数据集，在 Lenet 模型上进行训练

该训练脚本输出如下

```
[6.778578]
[2.0212684]
[1.3814741]
[0.47514156]
[0.13277876]
[0.16388433]
[0.03788032]
[0.01225162]
......
```

至此，我们成功完成了数据集制作，读取，训练整个流程了。

**赶快动手制作你的 OFRecord 数据集并用 OneFlow 进行训练吧！**

