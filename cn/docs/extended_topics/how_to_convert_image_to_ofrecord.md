# 制作OFRecord数据集

在 [OFrecord数据格式](./ofrecord.md) 和 [加载与准备 OFRecord 数据集](./how_to_make_ofdataset.md) 中，我们分别学习了 OFRecord 文件的存储格式以及如何加载 OFRecord 数据集。

本文，我们将围绕 OneFlow 的 OFRecord 数据集的编解码方式与制作展开，包括：

- OFRecord Reader 的解码方式
- OFRecord 的编码流程
- 制作基于 Mnist 手写数字数据集的 OFRecord 数据集

## OFRecord解码方式

OneFlow 内部的解码算子是采用 [OpenCV](https://opencv.org/) 来对数据进行解码的。相关的 `.cpp` 文件在 [oneflow.user.kernels.ofrecord_decoder_kernels.cpp](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/kernels/ofrecord_decoder_kernels.cpp) 

代码中的 `Feature` 就是在 [OFrecord数据格式](./ofrecord.md) 一章中提到的 OFRecord 内部用 [Protocol Buffers](https://developers.google.com/protocol-buffers/) 定义的序列化格式。

整体解码流程概括如下：

- 根据对应的 `name` 读取 OFRecord 内对应的 `Feature`

- 然后读取 `Feature` 内的 `BytesList` 类型数据，并进行一系列 **合法性检查** 
- 通过调用 OpenCV 库的 `imdecode` 方法对读取到的 **字节流数据** 进行解码， **转换成原始图片数据** 

- 对图片进行对应的后处理

## 将图片数据转化为OFRecord

了解了 OFRecord 的解码流程后，我们可以对 **整个流程进行反推** ，从而对图片数据进行编码转化为 OFRecord 数据集。

我们的编码流程如下：

- 调用 `imencode` 将原始图片数据编码成 **字节流数据** ，并进行序列化
- 转换成 OFRecord 的 `Feature`，并进行序列化

目前，OneFlow 图片编解码支持的格式与 OpenCV 的一致，可参见 [cv::ImwriteFlags](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac)，包括：

- JPEG，一种最常见的有损编码格式，可参考[JPEG](http://www.wikiwand.com/en/JPEG)
- PNG，一种常见的无损位图编码格式，可参考 [Portable Network Graphics](http://www.wikiwand.com/en/Portable_Network_Graphics)
- TIFF，一种可扩展的压缩编码格式，可参考 [Tagged Image File Format](http://www.wikiwand.com/en/TIFF)

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
def ndarray2ofrecords(dsfile, dataname, encoded_data, labelname, encoded_label):
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

我们使用 **MNIST手写数字数据集** 来完整制作一个OFRecord格式文件（这里我们仅取50张图片作为示例)，MNIST 示例数据集以及标签文件的下载地址为 [MNIST数据集](https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/images.zip)

下载至 `img_to_ofrecord` 目录下并解压后，整个代码目录构造如下 

```
img_to_ofrecord
├── images
	├── train_set
		├── 00000000_5.png
		├── 00000001_0.png
		├── 00000002_4.png
		......
	├── train_label
		├── label.txt
├── img2ofrecord.py
├── lenet_train.py
```

- `images` 目录存放原始示例训练数据集以及标签文件 `label.txt` 
- `img2ofrecord.py` 是将手写数字数据集转换成 OFRecord 格式文件的脚本
- `lenet_train.py` 则是读取我们制作好的 OFRecord 数据集，使用 LeNet 模型进行训练。 

完整代码：[img2ofrecord.py](../code/extended_topics/img_to_ofrecord/img2ofrecord.py)

- 我们读取50张示例训练图片，并分别调用 `encode_img_file`, `imgfile2label`, `ndarray2ofrecords`，来完成图像，标签的编码，并将数据写入到文件中。
- 我们通过命令行参数 `image_root`，`part_num`，`label_dir` 可以分别指定图片路径，数据切分个数，标签路径。

我们运行该脚本，并指定数据切分成5个分段，输出如下

```shell
$ python img2ofrecord.py --part_num=5 --save_dir=./dataset/ --img_format=.png --image_root=./images/train_set/
The image root is:  ./images/train_set/
The amount of OFRecord data part is:  5
The directory of Labels is:  ./images/train_label/label.txt
The image format is:  .png
The OFRecord save directory is:  ./dataset/
Start Processing......
./images/train_set/00000030_3.png feature saved
./images/train_set/00000034_0.png feature saved
./images/train_set/00000026_4.png feature saved
./images/train_set/00000043_9.png feature saved
......
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

