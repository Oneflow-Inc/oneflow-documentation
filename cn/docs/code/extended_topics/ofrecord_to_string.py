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


obserations = 28 * 28

f = open("./dataset/part-0", "wb")

for loop in range(0, 3):
    image = [random.random() for x in range(0, obserations)]
    label = [random.randint(0, 9)]

    topack = {
        "images": float_feature(image),
        "labels": int64_feature(label),
    }

    ofrecord_features = ofrecord.OFRecord(feature=topack)
    serilizedBytes = ofrecord_features.SerializeToString()

    length = ofrecord_features.ByteSize()

    f.write(struct.pack("q", length))
    f.write(serilizedBytes)

print("Done!")
f.close()
