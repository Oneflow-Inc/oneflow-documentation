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
