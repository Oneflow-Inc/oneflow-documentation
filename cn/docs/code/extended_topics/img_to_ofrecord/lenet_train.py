# lenet_train.py
import oneflow as flow
import oneflow.typing as tp


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv1",
        kernel_initializer=initializer,
    )
    pool1 = flow.nn.max_pool2d(
        conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
    )
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(
        conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
    )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")


def ofrecord_decode():
    batch_size = 100
    color_space = "GRAY"
    ofrecord = flow.data.ofrecord_reader(
        "./dataset/",
        batch_size=batch_size,
        # You should set the corresponding data_part_num
        data_part_num=10,
        part_name_suffix_length=-1,
        random_shuffle=True,
        shuffle_after_epoch=True,
    )
    image = flow.data.OFRecordImageDecoderRandomCrop(
        ofrecord, "images", color_space=color_space, random_area=(0.95, 1.0), random_aspect_ratio=(0.99, 1.0)
    )
    labels = flow.data.OFRecordRawDecoder(
        ofrecord, "labels", shape=(1,), dtype=flow.int32
    )
    rsz, scale, new_size = flow.image.Resize(
        image, target_size=(28, 28), channels=1
    )

    normal = flow.image.CropMirrorNormalize(
        rsz,
        color_space=color_space,
        mean=[0.0],
        std=[255.0],
        output_dtype=flow.float,
    )

    return normal, labels


@flow.global_function(type="train")
def train_job() -> tp.Numpy:
    images, labels = ofrecord_decode()

    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )

    loss = flow.math.reduce_mean(loss)
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)

    return loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()
    for epoch in range(100 * 600):
        loss = train_job()
        if epoch % 50 == 0: print(loss)
