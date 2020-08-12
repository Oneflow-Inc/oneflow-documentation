import numpy as np
import oneflow as flow
from PIL import Image
import sys
import os
import oneflow.typing as oft

BATCH_SIZE = 1


def usage():
    usageHint = """
usage:
        python {0} <image file>
    eg: 
        python {0} {1}
    """.format(
        os.path.basename(sys.argv[0]), os.path.join(".", "9.png")
    )
    print(usageHint)


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


@flow.global_function(type="predict")
def eval_job(
    images: oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> oft.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
    return logits


def load_image(file):
    im = Image.open(file).convert("L")
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im - 128.0) / 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im


def main():
    if len(sys.argv) != 2:
        usage()
        return

    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")

    image = load_image(sys.argv[1])
    logits = eval_job(image, np.zeros((1,)).astype(np.int32))

    prediction = np.argmax(logits, 1)
    print("prediction: {}".format(prediction[0]))


if __name__ == "__main__":
    main()
