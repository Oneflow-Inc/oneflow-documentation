# of_data_pipeline.py
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp
from typing import Tuple


@flow.global_function(type="predict")
def test_job() -> Tuple[tp.Numpy, tp.Numpy]:
    batch_size = 64
    color_space = "RGB"
    with flow.scope.placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader(
            "./",
            batch_size=batch_size,
            data_part_num=1,
            part_name_suffix_length=5,
            random_shuffle=True,
            shuffle_after_epoch=True,
        )
        image = flow.data.OFRecordImageDecoderRandomCrop(
            ofrecord, "encoded", color_space=color_space
        )
        label = flow.data.OFRecordRawDecoder(
            ofrecord, "class/label", shape=(), dtype=flow.int32
        )
        rsz = flow.image.Resize(
            image, resize_x=224, resize_y=224, color_space=color_space
        )

        rng = flow.random.CoinFlip(batch_size=batch_size)
        normal = flow.image.CropMirrorNormalize(
            rsz,
            mirror_blob=rng,
            color_space=color_space,
            mean=[123.68, 116.779, 103.939],
            std=[58.393, 57.12, 57.375],
            output_dtype=flow.float,
        )
        return normal, label


if __name__ == "__main__":
    images, labels = test_job()
    print(images.shape, labels.shape)
