import oneflow.core.record.record_pb2 as ofrecord
import struct

with open("./dataset/part-0", "rb") as f:
    for loop in range(0, 3):
        length = struct.unpack("q", f.read(8))
        serializedBytes = f.read(length[0])
        ofrecord_features = ofrecord.OFRecord.FromString(serializedBytes)

        image = ofrecord_features.feature["images"].float_list.value
        label = ofrecord_features.feature["labels"].int64_list.value

        print(image, label, end="\n\n")
