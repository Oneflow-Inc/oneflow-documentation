# 将图片文件制作为 OFRecord 数据集

在 [OFRecord 数据格式](./ofrecord.md) 和 [加载与准备 OFRecord 数据集](./how_to_make_ofdataset.md) 中，我们分别学习了如何将其它数据集格式转为 OFRecord 数据集，以及如何加载 OFRecord 数据集。

本文，我们将介绍如何将图片文件制作为 OFRecord 数据集，并提供了相关的制作脚本，方便用户直接使用或者在此基础上修改。内容包括：

- 制作基于 MNIST 手写数字数据集的 OFRecord 数据集

- OFRecord Reader 的编码方式
- 在自制的 OFRecord 数据集上进行训练

### 制作基于 MNIST 手写数字数据集的 OFRecord 文件

我们使用 **MNIST手写数字数据集** 来完整制作一个 OFRecord 格式文件

这里我们仅取50张图片作为示例，相关脚本和数据集的下载地址为 [img2ofrecord](https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/img2ofrecord.zip)

- 下载相关压缩包并解压

```shell
$ wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/img2ofrecord.zip
$ unzip img2ofrecord.zip
```

- 进入到对应目录，并运行 OFRecord 制作脚本 `img2ofrecord.py`

```shell
$ cd ./img_to_ofrecord
$ python img2ofrecord.py --part_num=5 --save_dir=./dataset/ --img_format=.png --image_root=./images/train_set/
```

- 脚本运行过程中，将输出以下内容

```shell
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

至此我们 OFRecord 文件制作完毕，并保存在 `./dataset` 目录下

### 代码解读

整个代码目录构造如下 

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

- `images` 目录存放原始示例训练数据集以及标签文件

这里我们的标签文件是以 `json` 格式存储的，形式如下：

```shell
{"00000030_3.png": 3}
{"00000034_0.png": 0}
{"00000026_4.png": 4}
{"00000043_9.png": 9}
{"00000047_5.png": 5}
{"00000003_1.png": 1}
......
```

- `img2ofrecord.py` 是将手写数字数据集转换成 OFRecord 格式文件的脚本
- `lenet_train.py` 则是读取我们制作好的 OFRecord 数据集，使用 LeNet 模型进行训练。 

处理图片文件并转换为 OFRecord 格式的脚本为 `img2ofrecord.py`，其命令行选项如下：

- `image_root` 指定图片的根目录路径
- `part_num` 指定生成 OFRecord 文件个数，如果该数目大于总图片数目，会报错
- `label_dir` 指定标签的目录路径
- `img_format` 指定图片的格式
- `save_dir` 指定 OFRecord 文件保存的目录

## OFRecord 的编码流程

与 OFRecord 文件编码的相关逻辑也在 `img2ofrecord.py` 内，其编码流程如下：

1. 对读取进来的图片数据进行编码 

```python
def encode_img_file(filename, ext=".jpg"):
    img = cv2.imread(filename)
    encoded_data = cv2.imencode(ext, img)[1]
    return encoded_data.tostring()
```

这里的 `ext` 是图片编码格式，目前，OneFlow 图片编解码支持的格式与 OpenCV 的一致，可参见 [cv::ImwriteFlags](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac)，包括：

- JPEG，一种最常见的有损编码格式，可参考 [JPEG](http://www.wikiwand.com/en/JPEG)
- PNG，一种常见的无损位图编码格式，可参考 [Portable Network Graphics](http://www.wikiwand.com/en/Portable_Network_Graphics)
- TIFF，一种可扩展的压缩编码格式，可参考 [Tagged Image File Format](http://www.wikiwand.com/en/TIFF)

2. 转化成 `Feature` 的形式，进行序列化，并将数据长度写入到文件中

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

## 使用自制的 OFRecord 数据集进行训练

我们运行目录下的 [lenet_train.py](../code/extended_topics/img_to_ofrecord/lenet_train.py)，它将读取我们刚制作好的 OFRecord 数据集，在 Lenet 模型上进行训练

该训练脚本输出如下：

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

