import time

import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp

flow.config.enable_legacy_model_io(False)
BATCH_SIZE = 100


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
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding="SAME", name="pool1")
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding="SAME", name="pool2")
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
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)


g_total = 0
g_correct = 0


def acc(arguments: Tuple[tp.Numpy, tp.Numpy]):
    global g_total
    global g_correct

    labels = arguments[0]
    logits = arguments[1]
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


def main():
    flow.load_variables(flow.checkpoint.get("./async_lenet_models"))
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )


    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            if i == 20:
                start = time.time()
            eval_job(images, labels)(acc)

    print("async predict elapsed time: ", time.time() - start)
    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))


if __name__ == "__main__":
    main()
