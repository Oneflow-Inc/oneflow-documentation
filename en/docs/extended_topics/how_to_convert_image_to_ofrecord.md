# Convert the image files to OFRecord datasets

In  [OFRecord Data Format](./ofrecord.md) and  [Loading and Preparing OFRecord Dataset](./how_to_make_ofdataset.md), We learned how to convert other dataset formats to OFRecord datasets separately and how to load OFRecord datasets.

In this article, we will explain how to make image files into OFRecord datasets. Also we provide relevant script for users to use directly or make modification base on that. Which includes: 

- OFRecord datasets based on MNIST Handwritten Digits.

- How OFRecord Reader is encoded.
- Training on OFRecord dataset.

### OFRecord datasets based on MNIST Handwritten Digits.

We use MNIST Handwritten Digits dataset to completely produce an OFRecord format file.

we only take 50 pictures for demonstration. The relevant script and datasets please refer to [img2ofrecord](https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/img2ofrecord.zip).

- Download and unzip the relevant zip file

```shell
$ wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/oneflow-tutorial-attachments/img2ofrecord.zip
$ unzip img2ofrecord.zip
```

- Go to the corresponding directory and run OFRecord production script `img2ofrecord.py`

```shell
$ cd ./img_to_ofrecord
$ python img2ofrecord.py --part_num=5 --save_dir=./dataset/ --img_format=.png --image_root=./images/train_set/
```

- The following output will display as the script runs.

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

Thus far, we have created the OFRecord file and saved under `./dataset`.

### Code explanation

The structure of entire code directory shows as follow: 

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

- `images` directory holds the original training dataset and label files.

The label file is stored as `json` here in following format：

```shell
{"00000030_3.png": 3}
{"00000034_0.png": 0}
{"00000026_4.png": 4}
{"00000043_9.png": 9}
{"00000047_5.png": 5}
{"00000003_1.png": 1}
......
```

- `img2ofrecord.py` is to covert hand written dataset to OFRecord file or script.
- `lenet_train.py` is to load the OFRecord dataset we just made and use LeNet model for training. 

Scripts that process image files and convert them to OFRecord format are`img2ofrecord.py`. The command options are as follows：

- `image_root` specify the root directory of the image.
- `part_num` specify the number of OFRecord files to generate. An error is reported if the number is greater than the total number of images.
- `label_dir` specify the directory of the label.
- `img_format` specify the format of the image.
- `save_dir` specify the directory where the OFRecord file will be saved.

## How OFRecord Reader is encoded

The logic associated with the encoding of OFRecord files is in `img2ofrecord.py`. The encoding process is as follows：

1. Encoding the incoming image data.

```python
def encode_img_file(filename, ext=".jpg"):
    img = cv2.imread(filename)
    encoded_data = cv2.imencode(ext, img)[1]
    return encoded_data.tostring()
```

The `ext` is the image encoding format. Currently, OneFlow support the same format of encode and decode for image files as OpenCV. Please refer to [cv::ImwriteFlags](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga292d81be8d76901bff7988d18d2b42ac). Which include：

- JPEG, one of the most common lossy code formats. Please refer to  [JPEG](http://www.wikiwand.com/en/JPEG).
- PNG, a common lossless bitmap encoding format. Please refer to [Portable Network Graphics](http://www.wikiwand.com/en/Portable_Network_Graphics).
- TIFF, a extensible compressed encoding format. Please refer to [Tagged Image File Format](http://www.wikiwand.com/en/TIFF).

2. Covert to `Feature` format then do serialization and record the length of the data into a file.

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

## Training on OFRecord dataset

We run [lenet_train.py](../code/extended_topics/img_to_ofrecord/lenet_train.py) in the dictionary. It will read the OFRecord dataset that we have just created and training by the Lenet model.

The outputs of training script should like below:

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

At this point, we have successfully completed the entire process of creating, reading, and training the dataset.

**Get started on your OFRecord dataset and train with OneFlow!** 

