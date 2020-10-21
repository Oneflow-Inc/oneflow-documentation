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
        default='./images/train_set/',
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

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)  # Make Save Directory
    printConfig(imgs_root, part_num, label_dir, img_format, save_dir)

    part_cnt = 0
    # Read the labels
    with open(label_dir, 'r') as label_file:
        imgs_labels = label_file.readlines()

    file_total_cnt = len(imgs_labels)
    assert file_total_cnt > part_num, "The amount of Files should be larger than part_num"
    per_part_amount = file_total_cnt // part_num

    for cnt, img_label in enumerate(imgs_labels):
        if cnt != 0 and cnt % per_part_amount == 0:
            part_cnt += 1
        prefix_filename = os.path.join(save_dir, "part-{}")
        ofrecord_filename = prefix_filename.format(part_cnt)
        with open(ofrecord_filename, 'ab') as f:
            data = json.loads(img_label.strip('\n'))
            for img, label in data.items():
                img_full_dir = os.path.join(imgs_root, img)
                encoded_data = encode_img_file(img_full_dir, img_format)
                ndarray2ofrecords(f, "images", encoded_data, "labels", label)
                print("{} feature saved".format(img_full_dir))

    print("Process image successfully !!!")
